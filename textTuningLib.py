'''
Support for "tuning scripts" for text machine learning projects.

Idea:
Separate from this module, there are tuning scripts that define Pipelines and
parameter options to try and evaluate. (see TuningTemplate.py)

As much as possible, all code is here for these scripts, and only the
pipeline definition & parameter options to evaluate are in the tuning scripts.

So the code here is coupled with TuningTemplate.py.

A big part of this module is the implementation of various tuning
reports used to analyze the tuning runs.

Assumes:
* Pipelines are for binary classification
* Pipelines have named steps: 'vectorizer', 'classifier' with their
*   obvious meanings (may have other steps too)
* The text sample data is in sklearn load_files directory structure
*   or in files parsable by sampleDataLib.py
* We are scoring Pipeline parameter runs via an F-Score
*   (beta is a parameter to this library)
* We can import sampleDataLib that encapsulates knowledge of the specific
*   document type/set we are dealing with (e.g., how to parse data files
*   and break records into documents, etc.)
* probably other things...

Evaluating Pipeline Parameters:
Two Approaches are supported
    (1) k-fold cross validation using GridSearchCV.
    (2) you provide a validation set
    For either approach you can provide either
	* a sample set that will be randomly split into a training
	    and test set.
	* separate training and test sets
    We'll evaluate the different parameter combinations via cross validation (1)
    or on the validation set provided (2) to select the best scoring parameters.
    Then the pipeline with best parameter combination is used to predict the
    test set and its results are reported.

Other functionality:
* Many options/parameters for this process are specified in a config file
    and/or command line arguments (to the tuning script using this module)
    that this module processes.

* Support for generating random seeds or using fixed specified seeds.

* Supports use of a pipeline step, FeatureDocCounter (defined in
    sklearnHelperLib), which you can put into a pipeline just after the
    vectorizer step, and it will give you access to
    the number of documents each feature appears in.

Reports...
If the 'classifier' supports getting weighted coefficients via
classifier.coef_, then the output from here can include a TopFeaturesReport

Coding Convention:
    Use camelCase for all the names here, but
    sklearn typically_uses_names with underscores. So _ for sklearn names.
'''
import sys
import time
import re
import string
import os
import os.path
import argparse
import ConfigParser

import sklearnHelperLib as skHelper

import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, precision_score,\
			recall_score, classification_report, confusion_matrix

# extend path up multiple parent dirs, hoping we can import sampleDataLib
sys.path = ['/'.join(dots) for dots in [['..']*i for i in range(1,4)]] + \
		sys.path
import sampleDataLib

SSTART = "### "			# report output section start delimiter

############################################################
# Common command line parameter handling for the tuning scripts
############################################################

def parseCmdLine():
    """
    Handles cmdline args and config file, returning dict that combines them.
    The Keys for the returned dict are easily discernable in the code below.
	for cmd line args, they are the 'dest' argument in add_argument()
	for non-cmd line args, they keys are set directly.
    """
    cp = ConfigParser.ConfigParser()
    cp.optionxform = str # make keys case sensitive

    # generate a path up multiple parent directories to search for config file
    cl = ['/'.join(l)+'/config.cfg' for l in [['.']]+[['..']*i for i in range(1,6)]]
    cp.read(cl)

    # config file params that are defaults for command line options
    TRAINING_DATA     = cp.get     ("DEFAULT", "TRAINING_DATA")
    TUNING_INDEX_FILE = cp.get     ("MODEL_TUNING", "TUNING_INDEX_FILE")

    basename = os.path.basename(sys.argv[0])
    OUTPUT_FILE_PREFIX = os.path.splitext(basename)[0]

    # command line parameters
    parser = argparse.ArgumentParser( \
    description='Run a tuning experiment script.')

    parser.add_argument('-d', '--trainpath', dest='trainDataPath',
            default=TRAINING_DATA,
            help='pathname where training data live. Default: "%s"' \
                    % TRAINING_DATA)

    parser.add_argument('--valpath', dest='valDataPath', default=None,
            help='pathname where validation data live. Default: None')

    parser.add_argument('--testpath', dest='testDataPath', default=None,
            help='pathname where test data live. Default: None')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='verbose: print longer tuning report.')

    parser.add_argument('--gsverbose', dest='gsVerbose', action='store_true',
                        help='have Gridsearch report details.')

    parser.add_argument('--rsplit', dest='randForSplit', default=None, type=int,
                        help="integer random seed for test_train_split.")

    parser.add_argument('--rclassifier', dest='randForClassifier',
			default=None, type=int,
                        help="integer random seed for classifier.")

    parser.add_argument('-i', '--index', dest='wIndex', action='store_true',
			default=False,
                        help='write to index file.')

    parser.add_argument('--noindex', dest='wIndex', action='store_false',
			default=False,
                        help="don't write to index file (default).")

    parser.add_argument('--indexfile', dest='tuningIndexFile', 
			default=TUNING_INDEX_FILE,
                        help='index file name. Default: %s' % \
							TUNING_INDEX_FILE)
    parser.add_argument('-p', '--predict', dest='wPredictions',
			action='store_true', default=False,
                        help='write predictions for test & training sets')

    parser.add_argument('--nopredict', dest='wPredictions',
			action='store_false', default=False,
	    help="don't write predictions for test & training sets (default)")

    parser.add_argument('--outprefix', dest='outputFilePrefix', 
			default=OUTPUT_FILE_PREFIX,
	    help='output file prefix (features & preds). Default: %s' % \
						    OUTPUT_FILE_PREFIX)
    parser.add_argument('--features', dest='writeFeatures',
			action='store_true', default=False,
			help="write feature file" )

    args =  parser.parse_args()

    # config params that are not cmdline args (yet)
    args.gridSearchBeta  = cp.getint  ("MODEL_TUNING", "GRIDSEARCH_BETA")
    args.compareBeta     = cp.getint  ("MODEL_TUNING", "COMPARE_BETA")
    args.testSplit       = cp.getfloat("MODEL_TUNING", "TEST_SPLIT")
    args.numCV           = cp.getint  ("MODEL_TUNING", "NUM_CV")
#    args.featureFile     = cp.get     ("MODEL_TUNING", "FEATURE_OUTPUT_FILE")
    args.yClassNames     = eval(cp.get("CLASS_NAMES",  "y_class_names"))
    args.yClassToScore   = cp.getint  ("CLASS_NAMES",  "y_class_to_score")
    args.rptClassNames   = eval(cp.get("CLASS_NAMES",  "rpt_class_names"))
    args.rptClassMapping = eval(cp.get("CLASS_NAMES",  "rpt_class_mapping"))
    args.rptNum      = cp.getint("CLASS_NAMES", "rpt_classification_report_num")

    return args
# ---------------------------

args = parseCmdLine()	# make args available to this code and importers

############################################################
# Main class - encapsulates the tuning process
############################################################

class TextPipelineTuningHelper (object):

    def __init__(self,
	pipeline,
	pipelineParameters,
	randomSeeds={'randForSplit':1},	# random seeds. Assume all are not None
	):
	self.pipeline           = pipeline
	self.pipelineParameters = pipelineParameters
	self.randomSeeds        = randomSeeds
	self.randForSplit       = randomSeeds['randForSplit']	# required seed

	# JIM: the idea was that only this constructor accesses args,
	#      but does it really make sense to copy all these to self. ?
	self.trainDataPath      = args.trainDataPath
	self.valDataPath        = args.valDataPath
	self.testDataPath       = args.testDataPath
	self.testSplit          = args.testSplit
	self.gridSearchBeta     = args.gridSearchBeta
	self.numCV              = args.numCV

	self.tuningIndexFile    = args.tuningIndexFile
	self.wIndex             = args.wIndex
	self.wPredictions       = args.wPredictions
	self.outputFilePrefix   = args.outputFilePrefix
	self.writeFeatures      = args.writeFeatures
	self.featureFile	= args.outputFilePrefix + "_features.txt"
	self.compareBeta        = args.compareBeta
	self.verbose            = args.verbose
	self.gsVerbose          = args.gsVerbose

	self.yClassNames        = args.yClassNames
	self.yClassToScore      = args.yClassToScore
	self.rptClassNames      = args.rptClassNames
	self.rptClassMapping    = args.rptClassMapping
	self.rptNum             = args.rptNum

	self.startTimeStr = getFormattedTime()
	self.startTime    = time.time()
	self.scorer = make_scorer(fbeta_score, beta=self.gridSearchBeta,
					  pos_label=self.yClassToScore)
    #---------------------

    def loadTrainValTestSets(self):
	"""
	Get the training, test, & validation sets (val is optional).
	This means setting three parallel lists:
	    self.sampleNames_xxx	names (IDs) of the docs
	    self.docs_xxx		the docs themselves (string)
	    self.y_xxx			the class name index (0, 1)
	    				y_xxx is an np.array
	    For xxx in "train", "test", and "val"
	"""

	####### training set
	trainSet = DocumentSet().load(self.trainDataPath)

	self.sampleNames_train = trainSet.getSampleNames()
	self.docs_train        = trainSet.getDocs()
	self.y_train           = trainSet.getYvalues()

	####### test set
	if self.testDataPath:
	    testSet = DocumentSet().load(self.testDataPath)
	else:		# no specified test set, split random set from training	
	    testSet = trainSet.split(splitSize=self.testSplit,
						randomSeed=self.randForSplit)
	self.sampleNames_test = testSet.getSampleNames()
	self.docs_test        = testSet.getDocs()
	self.y_test           = testSet.getYvalues()

	####### Validation set
	if self.valDataPath:
	    self.haveValSet = True
	    valSet = DocumentSet().load(self.valDataPath)

	    self.sampleNames_val = valSet.getSampleNames()
	    self.docs_val        = valSet.getDocs()
	    self.y_val           = valSet.getYvalues()
	else:
	    self.haveValSet = False

    #---------------------

    def getGridSearchParams(self):
	"""
	Figure out what doc set, y values, cv value to use for the GridSearchCV.

	Return docs, y, cv

	If we don't have a specified validation set,
	    Have GridSearchCV use k-fold cross validation on the training set
	    Return docs_train, y_train, and the int cv value (num of folds)

	If we do have a specified validation set, this is more subtle.
	    Since we have a val set, we don't want to use k-fold, but
	    we still want to use GridSearchCV to train/compare each permutation
	    (so we want to use the GridSearch w/o the CV)

	    Why use GridSearchCV to do this?
	    Because it is coded to try the diff permutations in parallel.

	    We'll combine the training set w/ the validation set into
	    one document list,
	    And tell GridSearchCV how to separate the training docs from the
	    validation docs in one split.

	    This is done by providing a cv value that is
	    a list (iterator) of pairs of doc indexes

	    [ ( [training doc indexes], [val doc indexes] ),  ]
	    
	    In our case, we only want one pair of index lists since we only
	    have one "split" into training and validation docs.

	    See the cv parameter here:
	    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

	    It is not very well documented, but by chasing down the source
	    code, I figured out that the structure above is what the cv list
	    (iterator) needs to be.

	    Maybe there is a better way to do this, but I haven't found it.
	"""
	if self.haveValSet: 
	    docs_gs = self.docs_train + self.docs_val
	    y_gs    = np.concatenate( (self.y_train, self.y_val) )

	    lenTrain = len(self.docs_train)
	    lenVal   = len(self.docs_val)
	    cv = [ (range(lenTrain), range(lenTrain, lenTrain+lenVal) ), ]

	else:				# no val set, use k-fold
	    docs_gs         = self.docs_train
	    y_gs            = self.y_train
	    cv              = self.numCV

	return docs_gs, y_gs, cv
    #---------------------

    def fit(self):
	'''
	run the GridSearchCV
	FIXME: seems like we might often be running this on a fixed training
	    and validation set without multiple parameter options
	    - say on a newly preprocessed set of data.
	    So we really don't need to do the gridsearch itself.
	    Should think about refactoring to skip the gridsearch itself
	    in these cases.
	'''
	# using _train _test variable names as is the custom in sklearn.
	# "y_" are the correct classifications (labels) for the corresponding
	#   samples
	# _gs = grid search set

	self.loadTrainValTestSets()

	docs_gs, y_gs, cv = self.getGridSearchParams()

	self.gs = GridSearchCV( self.pipeline,
				self.pipelineParameters,
				scoring= self.scorer,
				cv=      cv,
				refit=   True,
				verbose= self.gsVerbose,
				n_jobs=  4,
				)
	self.gs.fit( docs_gs, y_gs )

	# bestEstimator with the best params evaluated on val set or cv
	# Note, this has been trained on train + val set or whole train set
	#  (without any cv)
	self.bestEstimator  = self.gs.best_estimator_

	self.bestVectorizer = self.bestEstimator.named_steps['vectorizer']
	self.bestClassifier = self.bestEstimator.named_steps['classifier']
	self.featureEvaluator = self.bestEstimator.named_steps.get( \
						    'featureEvaluator',None)
	if self.featureEvaluator == None: self.featureValues = None
	else: self.featureValues = self.featureEvaluator.getValues()

	# run estimator on the training, test sets so we can compare
	self.verboseWrite("Predicting on training set\n")
	self.y_predicted_train = self.bestEstimator.predict(self.docs_train)

	self.verboseWrite("Predicting on test set\n")
	self.y_predicted_test  = self.bestEstimator.predict(self.docs_test)

	# run on validation set so we can see how it does & compare to test
	if self.haveValSet:
	    # if val preds score close to training set, seems like overfit
	    # otherwise, should be close to test set or somewhat intermediate
	    self.verboseWrite("Predicting on validation set\n")
	    self.y_predicted_val  = self.bestEstimator.predict(self.docs_val)

	self.verboseWrite("Done with predictions\n")

	return self		# customary for fit() methods
    # ---------------------------

    def verboseWrite(self, msg):
	if self.gsVerbose:
	    sys.stderr.write(getFormattedTime() + " " + msg)
	    sys.stderr.flush()
    # ---------------------------

    def getReports(self,
		    nFeaturesReport=10,		# in features report
		    nFalsePosNegReport=5,	# for false pos/neg rpt
		    nTopFeatures=20,		# num of top weighted to rpt
	):
	if self.wIndex:
	    self.writeIndexFile(self.tuningIndexFile, self.compareBeta)
	if self.wPredictions:
	    self.writePredictions()
	if self.writeFeatures:
	    writeFeatures(self.bestVectorizer, self.bestClassifier,
				    self.featureFile, values=self.featureValues)
	output = self.getReportStart()

	output += getFormattedMetrics("Training Set", self.y_train,
				    self.y_predicted_train, self.compareBeta,
				    rptClassNames=self.rptClassNames,
				    rptClassMapping=self.rptClassMapping,
				    rptNum=self.rptNum,
				    yClassNames=self.yClassNames,
				    yClassToScore=self.yClassToScore,
				    )
	if self.haveValSet:
	    output += getFormattedMetrics("Validation Set", self.y_val,
					self.y_predicted_val, self.compareBeta,
					rptClassNames=self.rptClassNames,
					rptClassMapping=self.rptClassMapping,
					rptNum=self.rptNum,
					yClassNames=self.yClassNames,
					yClassToScore=self.yClassToScore,
					)
	output += getFormattedMetrics("Test Set", self.y_test,
				    self.y_predicted_test, self.compareBeta,
				    rptClassNames=self.rptClassNames,
				    rptClassMapping=self.rptClassMapping,
				    rptNum=self.rptNum,
				    yClassNames=self.yClassNames,
				    yClassToScore=self.yClassToScore,
				    )
	output += getBestParamsReport(self.gs, self.pipelineParameters)
	output += getGridSearchReport(self.gs, self.pipelineParameters)

	if self.verbose: 
	    features = skHelper.getOrderedFeatures( self.bestVectorizer,
						    self.bestClassifier)
	    output += getTopFeaturesReport( features, nTopFeatures) 

	    output += getVectorizerReport(self.bestVectorizer,
					    nFeatures=nFeaturesReport)

	    # false positives/negatives report.
	    # JIM: Should this be for validation set or test or both?
	    falsePos,falseNeg = skHelper.getFalsePosNeg( self.y_test,
					self.y_predicted_test,
					self.sampleNames_test,
					positiveClass=self.yClassToScore)

	    output += getFalsePosNegReport( "Test set", falsePos, falseNeg,
					    num=nFalsePosNegReport)

	    output += self.getSampleSummaryReport()

	output += self.getReportEnd()
	return output
    # ---------------------------

    def getReportStart(self):

	output = SSTART + "Start Time %s  %s" % (self.startTimeStr, sys.argv[0])
	if self.wIndex: output += "\tindex file: %s" % self.tuningIndexFile
	output += "\n"
	output += "Training data path:   %s\tGridSearch Beta: %d\n" % \
				    (self.trainDataPath, self.gridSearchBeta)
	output += "Validation data path: %s\n" % (str(self.valDataPath))
	output += "Test data path:       %s\n" % (str(self.testDataPath))
	output += getRandomSeedReport(self.randomSeeds)
	output += "\n"
	return output
    # ---------------------------

    def getReportEnd(self):
	return SSTART + "End Time %s. Total %9.2f seconds\n" % \
	    (getFormattedTime(), time.time() - self.startTime)
    # ---------------------------

    def writeIndexFile(self, tuningIndexFile, compareBeta):
	'''
	Handle writing a one-line summary of this run to an index file
	'''
	y_true = self.y_test
	y_predicted = self.y_predicted_test

	if len(sys.argv) > 0: tuningFile = sys.argv[0]
	else: tuningFile = ''

	with open(tuningIndexFile, 'a') as fp:
	    fp.write("%s\tPRF%d,F1\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%s\n" % \
	    (self.startTimeStr,
	    compareBeta,
	    precision_score(y_true, y_predicted, pos_label=self.yClassToScore),
	    recall_score(   y_true, y_predicted, pos_label=self.yClassToScore),
	    fbeta_score(    y_true, y_predicted, compareBeta,
						pos_label=self.yClassToScore), 
	    fbeta_score(    y_true, y_predicted, 1,
						pos_label=self.yClassToScore), 
	    tuningFile,
	    ) )
    # ---------------------------

    def writePredictions(self,):
	'''
	Write files with predictions from training set and test set
	'''
	writePredictionFile( \
	    self.outputFilePrefix + "_train_pred.txt",
	    self.bestEstimator,
	    self.docs_train,
	    self.sampleNames_train,
	    self.y_train,
	    self.y_predicted_train,
	    classNames=self.yClassNames,
	    positiveClass=self.yClassToScore,
	    )
	writePredictionFile( \
	    self.outputFilePrefix + "_test_pred.txt",
	    self.bestEstimator,
	    self.docs_test,
	    self.sampleNames_test,
	    self.y_test,
	    self.y_predicted_test,
	    classNames=self.yClassNames,
	    positiveClass=self.yClassToScore,
	    )
    # ---------------------------

    def getSampleSummaryReport(self,):
	"""
	Return string that summarizes the subsets of samples (training,
	    validation, test)
	"""
	# -----------------
	def formatter(title, y):
	    n = len(y)
	    pos = list(y).count(self.yClassToScore)
	    neg = n - pos
	    output = "%-20s: %12d %12d %12d %11.0f%%\n" % \
			( title[:20], n, pos, neg,  100.0 * (float(pos)/n) )
	    return output
	# -----------------

	output = SSTART + "Sample set sizes\n"
	# Header line
	output += "%-20s: %12s %12s %12s %12s\n" % \
		    ( ' ', 'Samples', 'Positive', 'Negative', '% Positive')

	# One line for each document set
	if self.haveValSet:
	    output += formatter('Validation Set', self.y_val)
	output += formatter('Training Set', self.y_train)
	output += formatter('Test Set',     self.y_test)
	output += "TestSplit: %4.2f\n" % self.testSplit
	return output

# ---------------------------
# end class TextPipelineTuningHelper
# ---------------------------

class DocumentSet (object):
    """
    IS:     a set of documents along with their y_values and sampleNames
    HAS:    parallel lists: documents, y_values, sampleNames along with
	      information about where the docs were loaded from
    DOES:   Load from a file or directory structure using sklearn load_files(),
	    Split into two doc sets, 
	    Convert y_values to and from lists and np.array
    FIXME: maybe this should live in sampleDataLib.py ??
	    (it should not live in sklearnHelperLib since it requires import
	    of sampleDataLib)
    """
    def __init__(self,
		docs=[],	# list of document (strings)
		y=[],		# list of ints 0,1 or np.array w/ 0,1
		sampleNames=[],	# list of names, (strings)
	):
	"""
	Assumes: y is as above.
	"""
	if type(y) == type([]): self.y = np.array(y)	# story as np.array
	else: self.y = y

	self.docs        = docs
	self.sampleNames = sampleNames
	self.pathName    = None		# where loaded from
	self.splitSize   = None		# if we do split, fraction to new set
	self.randomSeed  = None		# if we do split, use for random subset

    # ---------------------------
    def getDocs(self):		return self.docs
    def getSampleNames(self):	return self.sampleNames
    def getPathName(self):	return self.pathName
    def getSplitSize(self):	return self.splitSize
    def getRandomSeed(self):	return self.randomSeed
    def getYvalues(self):	return self.y		# as np.array
    def getYvaluesAsList(self):	return self.y.tolist()	# as list
    # ---------------------------

    def split(self,
		splitSize=None,		# if float, fraction for new DocumentSet
					# if int, size of new DocumentSet
					# if None, use train_test_split default
		randomSeed=None,	# int, use this seed for random
					#   selection of split subset
					# None means use a random value
	):
	"""
	Split self into two DocumentSets.
	Return new DocumentSet of random subset with splitSize docs.
	(remove the random subset of docs from self)
	"""
	self.splitSize  = splitSize
	self.randomSeed = randomSeed

	self.sampleNames, sampleNames_new,	\
	self.docs,        docs_new,		\
	self.y,           y_new = train_test_split( \
					    self.sampleNames,
					    self.docs,
					    self.y,
					    test_size=self.splitSize,
					    random_state=self.randomSeed)

	return DocumentSet(docs=docs_new, y=y_new, sampleNames=sampleNames_new)
    # ---------------------------

    def load(self, path):
	"""
	Load self from the specified directory path.
	Path can be a
	    Directory name, in which case we assume there are subdirectories
		to be loaded by sklearn load_files()
	    Filename, in which case we assume it is a file of records that
		can be loaded by sampleDataLib.SampleRecord().
	Return self
	"""
	self.path = os.path.realpath(path)
	if os.path.isdir(self.path):
	    self.loadFromDir(self.path)
	else:
	    self.loadFromFile(self.path)

	return self
    # ---------------------------

    def loadFromDir(self, path):

	dataSet          = load_files( path )
	self.docs        = dataSet.data
	self.y           = dataSet.target
	self.sampleNames = self.fileNamesToSampleNames(dataSet.filenames)

	return self
    # ---------------------------

    def fileNamesToSampleNames(self, filenames):
	'''
	Convert list of filenames into sampleNames:
	    file basenames without any file extension.
	If this conversion doesn't work for you, subclass and override
	    this method.
	'''
	return [ os.path.splitext(os.path.basename(fn))[0] for fn in filenames ]
    # ---------------------------

    def loadFromFile(self, path):
	self.docs = []
	self.y = []
	self.sampleNames = []

        rcds = open(path,'r').read().split(sampleDataLib.RECORDSEP)

	del rcds[0]		# header line
        del rcds[-1] 		# empty string after end of split

	for rcd in rcds:
	    sr = sampleDataLib.SampleRecord(rcd)
	    self.docs.append       (sr.getDocument())
	    self.y.append          (sr.getKnownYvalue())
	    self.sampleNames.append(sr.getSampleName())

	self.y = np.array(self.y)
	return self
# end class DocumentSet ---------------------------

############################################################
# Functions to format output reports and other things.
# These are functions that concievably could be useful outside on their own.
# SO these do not use args or config variables defined above.
############################################################
def writePredictionFile( \
    outputFile,		# file name (string) or file object (e.g., stdout)
    pipeline,		# trained pipeline (vectorizer+classifier) for predict
    docs,		# [strings] set of (predicted) docs to write predictions
    sampleNames,	# [strings] sample names for the predicted docs
    y_true,		# [ 0|1 ] true classifications for the docs
    y_predicted,	# [ 0|1 ] predicted classifications for the docs
    classNames=['no','yes'], # class labels
    positiveClass=1,	# index in classNames considered the positive class
    ):
    '''
    Write a prediction file, with confidence values if available.
    Prediction file has a line for each doc,
	samplename, y_true, y_predicted, FP/FN, [confidence, abs value]
    '''
    predTypes = [skHelper.predictionType(t, p, positiveClass=positiveClass) \
					    for t,p in zip(y_true,y_predicted)]
    
    trueNames = [ classNames[y] for y in y_true ]
    predNames = [ classNames[y] for y in y_predicted ]
    
    if hasattr(pipeline, "decision_function"):	# include confidences
	confidences = pipeline.decision_function(docs).tolist()
	absConf = map(abs, confidences)
	predictions = \
	 zip(sampleNames, trueNames, predNames, predTypes, confidences, absConf)

	# sort by absolute value of confidence
	selConf = lambda x: x[5]	# select confidence value 
	predictions = sorted(predictions, key=selConf, reverse=True)

	header = "ID\tTrue Class\tPred Class\tFP/FN\tConfidence\tAbs Value\n"
	template = "%s\t%s\t%s\t%s\t%5.3f\t%5.3f\n"
    else:			# no confidence values available
	predictions = zip(sampleNames, trueNames, predNames, predTypes)

	header = "ID\tTrue Class\tPred Class\FP/FN\n"
	template = "%s\t%s\t%s\t%s\n"

    if type(outputFile) == type(''): fp = open(outputFile, 'w')
    else: fp = outputFile

    fp.write(header)
    for p in predictions:
	fp.write(template % p)

    return
# ---------------------------

def writeFeatures( vectorizer,	# fitted vectorizer from a pipeline
		    classifier,	# fitted classifier from the pipeline
		    outputFile,	# filename (string) or file obj (e.g., stdout)
		    values=None,# list of feature values, one for each feature
				# This is some number per feature based on
				#   analysis of the features. Typical example:
				#   number of documents each feature appears in
				#   (but could be anything)
				# Values are printed with %d. This could
				#  become a problem....
    ):
    '''
    Write the full list of feature names to the file
    if values = [ ], should be numeric list parallel to feature names
	we will print these after the feature name
    if classifier can provide feature coefficients, we will print these too
    All "|" delimited.
    Assumes:  vectorizer has get_feature_names() method
    '''
    delimiter = '|'
    featureNames = vectorizer.get_feature_names()

    hasValues = values != None

    hasCoef = hasattr(classifier, 'coef_')
    if hasCoef: coefficients = classifier.coef_[0].tolist()

    hasImportance = hasattr(classifier, 'feature_importances_')
    if hasImportance: importances = classifier.feature_importances_.tolist()

    if type(outputFile) == type(''): fp = open(outputFile, 'w')
    else: fp = outputFile

    for i,f in enumerate(featureNames):
	fp.write(f)
	if hasValues:     fp.write(delimiter + '%d' % values[i])
	if hasCoef:       fp.write(delimiter + '%+5.4f' % coefficients[i])
	if hasImportance: fp.write(delimiter + '%+5.4f' % importances[i])
	fp.write('\n')
    return 
# ----------------------------

def getBestParamsReport( \
    gs,	    	# sklearn.model_selection.GridsearchCV that has been .fit()
    parameters  # dict of parameters used in the gridsearch
    ):
    output = SSTART +'Best Pipeline Parameters:\n'
    for pName in sorted(parameters.keys()):
	output += "%s: %r\n" % ( pName, gs.best_params_[pName] )

    output += "\n"
    return output
# ---------------------------

def getGridSearchReport( \
    gs,	    	# sklearn.model_selection.GridsearchCV that has been .fit()
    parameters  # dict of parameters used in the gridsearch
    ):
    output = SSTART + 'GridSearch Pipeline:\n'
    for stepName, obj in gs.best_estimator_.named_steps.items():
	output += "%s:\n%s\n\n" % (stepName, obj)

    output += SSTART + 'Parameter Options Tried:\n'
    for key in sorted(parameters.keys()):
	output += "%s:%s\n" % (key, str(parameters[key])) 

    output += "\n"
    return output
# ---------------------------

def getVectorizerReport(vectorizer, nFeatures=10):
    '''
    Format a report on a fitted vectorizer, return string
    '''
    featureNames = vectorizer.get_feature_names()
    midFeature   = len(featureNames)/2

    output =  SSTART + "Vectorizer:   Number of Features: %d\n" \
    						% len(featureNames)
    output += "First %d features: %s\n\n" % (nFeatures,
		format(featureNames[:nFeatures]) )
    output += "Middle %d features: %s\n\n" % (nFeatures,
		format(featureNames[ midFeature : midFeature+nFeatures]) )
    output += "Last %d features: %s\n\n" % (nFeatures,
		format(featureNames[-nFeatures:]) )
    return output
# ---------------------------

def getFalsePosNegReport( \
	title,		# string title, typically  "Train" or "Test"
	falsePositives,	# list of (sample names) of the falsePositives
	falseNegatives,	# ... 
	num=5		# number of false pos/negs to display
	):
    '''
    Report on the false positives and false negatives in a test set
    '''

    output = SSTART+"False positives for %s: %d\n" % (title,len(falsePositives))
    for name in falsePositives[:num]:
	output += "%s\n" % name

    output += "\n"
    output += SSTART+"False negatives for %s: %d\n" % (title,len(falseNegatives))
    for name in falseNegatives[:num]:
	output += "%s\n" % name

    output += "\n"
    return output
# ---------------------------

def getFormattedMetrics( \
	title,		# string title, typically  "Train" or "Test"
	y_true,		# true category assignments
	y_predicted,	# predicted assignments
	beta,		# for the fbeta_score
	rptClassNames=['yes', 'no'],
			# class labels for report outputs, in desired order
	rptClassMapping=[1,0],
			# rptClassMapping[y_val] = corresp name in rptClassNames
	rptNum=1,	# num classes to show in classification_report
			#   will be the 1st rptNum names in rptClassNames
	yClassNames=['no', 'yes'],
			# class labels for the actual y_values
	yClassToScore=1	# index of actual class in yClassNames to score
	):
    '''
    Return formated metrics report for a set of predictions.
    y_true and y_predicted are lists of integer category indexes (y_vals).
    Assumes we are using a fbeta score, not a good thing in the long term
    '''
    # concat title string onto all the target class names so
    #  they are easier to differentiate in multiple reports (and you can
    #  grep for them)
    target_names = [ "%s %s" % (title[:5], x) for x in rptClassNames ]

    output = SSTART + "Metrics: %s\n" % title
    output += "%s\n" % (classification_report( \
			y_true, y_predicted,
			target_names=target_names,
			labels=rptClassMapping[:rptNum], )
		    )
    output += "%s F%d: %7.5f (%s)\n\n" % \
	    (title[:5], beta,
	    fbeta_score(y_true,y_predicted,beta, pos_label=yClassToScore ),
	    yClassNames[yClassToScore],
	    )
    output += "%s\n" % getFormattedCM(y_true, y_predicted)

    return output
# ---------------------------

def getFormattedCM( \
    y_true,		# true category assignments for test set
    y_predicted,	# predicted assignments
    rptClassNames=['yes', 'no'],
		    # class labels for report outputs, in desired order
    rptClassMapping=[1,0],
		    # rptClassMapping[y_val] = corresp name in rptClassNames
    ):
    '''
    Return (minorly) formated confusion matrix
    FIXME: this could be greatly improved
    '''
    output = "%s\n%s\n" % ( \
	    str(rptClassNames),
	    str(confusion_matrix(y_true, y_predicted, labels=rptClassMapping)))
    return output
#  ---------------------------

def getTopFeaturesReport(  \
    orderedFeatures,	# features: [ ('feature name', coef), ...]
    num=20,		# number of features w/ highest & lowest coefs to rpt
    ):
    '''
    Return report of the features w/ the highest (positive) and lowest
    (negative) coefficients.
    Assumes num < len(orderedFeatures).
    '''
    if len(orderedFeatures) == 0:		# no coefs
	output =  SSTART + "Top positive features - not available\n"
	output += SSTART + "Top negative features - not available\n"
	output += "\n"
	return output

    topPos = orderedFeatures[:num]
    topNeg = orderedFeatures[len(orderedFeatures)-num:]

    output = SSTART + "Top positive features (%d)\n" % len(topPos)
    for f,c in topPos:
	output += "%+5.4f\t%s\n" % (c,f)
    output += "\n"

    output += SSTART + "Top negative features (%d)\n" % len(topNeg)
    for f,c in topNeg:
	output += "%+5.4f\t%s\n" % (c,f)
    output += "\n"

    return output
# ---------------------------

def getFormattedTime():
    return time.strftime("%Y/%m/%d-%H-%M-%S")
# ---------------------------


############################################################
# Random seed support:
# For various methods, random seeds are used
#   e.g., for train_test_split() the seed is used to decide which samples
#         make it into which set.
# However, we want to record/report the random seeds used so we can
#     reproduce results when desired.
#     So we use these routines to always provide and report a random seed.
#     If a seed is provided, we use it, if not, we generate one here.
#
# getRandomSeeds() takes a dictionary of seeds, and generates random seeds
#     for any key that doesn't already have a numeric seed
# getRandomSeedReport() formats a seed dictionary in a standard way
#     for reporting.
############################################################

def getRandomSeeds( seedDict    # dict: {'seedname' : number or None }
    ):
    '''
    Set a random integer for each key in seedDict that is None
    '''
    for k in seedDict.keys():
        if seedDict[k] == None: seedDict[k] = np.random.randint(1000)

    return seedDict
# ---------------------------

def getRandomSeedReport( seedDict ):
    output = "Random Seeds:\t"
    for k in sorted(seedDict.keys()):
        output += "%s=%d   " % (k, seedDict[k])
    return output
# ---------------------------

if __name__ == "__main__":   ############### ad hoc test code ##############3
    if True:	# DocumentSet testing
	print "docs 1"
	d = DocumentSet()

	d.load('Data/smallset/figureText.txt')
	print len(d.getDocs())
	print d.getSampleNames()[:5]
	print d.getYvalues()[:5]
	print d.getYvaluesAsList()[:5]

	d2 = d.split(splitSize=0.2)
	print "docs 1 after split"
	print len(d.getDocs())
	print d.getSampleNames()[:5]
	print "docs 2"
	print len(d2.getDocs())
	print d2.getSampleNames()[:5]

	print "docs 3"
	d3 = DocumentSet()
	print d3.load('Data/smallset').getSampleNames()[:5]
