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

* Provides a pipeline step, FeatureDocCounter, which you can put into a
    pipeline just after the vectorizer step, and it will give you access to
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
from sklearn.base import TransformerMixin, BaseEstimator
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
    JIM: check that this list is still accurate
    shared among the ModelTuning scripts
    Handles cmdline args and config file, returning dict that combines them.
    Keys:
    trainDataPath:	dir path to training data
    verbose:		boolean - print longer tuning report
    randForSplit:	int
    randForClassifier:	int
    tuningIndexFile:	index file name to write to
    wIndex:		boolean - write indexfile
    wPredictions:	boolean - write predictions file
    predFilePrefix:	filename prefix for the prediction output files
    gridSearchBeta:	int
    compareBeta:	int
    testSplit:		float
    gridSearchCV:	int
    yClassNames:	list of class names corresponding to Y values
    yClassToScore:	index in yClassNames for class to use for f-score
    repClassNames:	list of class names ordered as we want to report them
    rptClassMapping: 	mapping of y_vals to classes in repClassNames 
    rptNum:      	how many of rptClassNames to show in classi'cation rpt
    """
    cp = ConfigParser.ConfigParser()
    cp.optionxform = str # make keys case sensitive

    # generate a path up multiple parent directories to search for config file
    cl = ['/'.join(l)+'/config.cfg' for l in [['.']]+[['..']*i for i in range(1,6)]]
    cp.read(cl)

    # config file params that are defaults for command line options
    TRAINING_DATA     = cp.get     ("DEFAULT", "TRAINING_DATA")
    TUNING_INDEX_FILE = cp.get     ("MODEL_TUNING", "TUNING_INDEX_FILE")
    FEATURE_OUTPUT_FILE = cp.get   ("MODEL_TUNING", "FEATURE_OUTPUT_FILE")
    PRED_OUTPUT_FILE_PREFIX = cp.get("MODEL_TUNING", "PRED_OUTPUT_FILE_PREFIX")

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

    parser.add_argument('--predfiles', dest='predFilePrefix', 
			default=PRED_OUTPUT_FILE_PREFIX,
	    help='prefix for prediction output filenames. Default: %s' % \
						    PRED_OUTPUT_FILE_PREFIX)
    parser.add_argument('--features', dest='writeFeatures',
			action='store_true', default=False,
	    help="write full feature set to %s" % FEATURE_OUTPUT_FILE)

    args =  parser.parse_args()

    # config params that are not cmdline args (yet)
    args.gridSearchBeta  = cp.getint  ("MODEL_TUNING", "GRIDSEARCH_BETA")
    args.compareBeta     = cp.getint  ("MODEL_TUNING", "COMPARE_BETA")
    args.testSplit       = cp.getfloat("MODEL_TUNING", "TEST_SPLIT")
    args.numCV           = cp.getint  ("MODEL_TUNING", "NUM_CV")
    args.featureFile     = cp.get     ("MODEL_TUNING", "FEATURE_OUTPUT_FILE")
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
	# JIM: does it really make sense to copy all these to self. ?
	#      the idea was that only this constructor access args.
	self.pipeline           = pipeline
	self.pipelineParameters = pipelineParameters
	self.randomSeeds        = randomSeeds
	self.randForSplit       = randomSeeds['randForSplit']	# required seed

	self.trainDataPath      = args.trainDataPath
	self.valDataPath        = args.valDataPath
	self.testDataPath       = args.testDataPath
	self.testSplit          = args.testSplit
	self.gridSearchBeta     = args.gridSearchBeta
	self.numCV              = args.numCV

	self.tuningIndexFile    = args.tuningIndexFile
	self.wIndex             = args.wIndex
	self.wPredictions       = args.wPredictions
	self.predFilePrefix     = args.predFilePrefix
	self.writeFeatures      = args.writeFeatures
	self.featureFile	= args.featureFile
	self.compareBeta        = args.compareBeta
	self.verbose            = args.verbose
	self.gsVerbose          = args.gsVerbose

	self.yClassNames        = args.yClassNames
	self.yClassToScore      = args.yClassToScore
	self.rptClassNames      = args.rptClassNames
	self.rptClassMapping    = args.rptClassMapping
	self.rptNum             = args.rptNum
	self.time = getFormattedTime()

	self.scorer = make_scorer(fbeta_score, beta=self.gridSearchBeta,
					  pos_label=self.yClassToScore)
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
	"""
	docSets = DocumentSetLoader(self.trainDataPath, self.valDataPath,
				    self.testDataPath,
				    testSplit=self.testSplit,
				    randomSeed=self.randForSplit)

	self.sampleNames_train = docSets.trainingSet.sampleNames
	self.docs_train        = docSets.trainingSet.docs
	self.y_train           = docSets.trainingSet.y

	self.sampleNames_test = docSets.testSet.sampleNames
	self.docs_test        = docSets.testSet.docs
	self.y_test           = docSets.testSet.y

	if docSets.validationSet == None: 	# no val set, use k-fold
	    self.haveValSet = False
	    docs_gs         = self.docs_train
	    y_gs            = self.y_train
	    cv              = 2  # JIM: small cv for testing self.numCV
	else:
	    self.haveValSet      = True
	    self.sampleNames_val = docSets.validationSet.sampleNames
	    self.docs_val        = docSets.validationSet.docs
	    self.y_val           = docSets.validationSet.y

	    docs_gs = self.docs_train + self.docs_val
	    y_gs    = np.concatenate( (self.y_train, self.y_val) )

	    lenTrain = len(self.docs_train)
	    lenVal   = len(self.docs_val)
	    cv = [ (range(lenTrain), range(lenTrain, lenTrain+lenVal) ), ]

	return docs_gs, y_gs, cv
    #---------------------

    def fit(self):
	'''
	run the GridSearchCV
	'''
	# using _train _test variable names as is the custom in sklearn.
	# "y_" are the correct classifications (labels) for the corresponding
	#   samples
	# _gs = grid search set
	docs_gs, y_gs, cv = self.getGridSearchParams()

	self.gs = GridSearchCV( self.pipeline,
				self.pipelineParameters,
				scoring= self.scorer,
				cv=      cv,
				verbose= self.gsVerbose,
				n_jobs=  -1,
				)
	self.gs.fit( docs_gs, y_gs )

	self.bestEstimator  = self.gs.best_estimator_

	# Now that we've found the estimator that scores best on val set,
	# retrain the bestEstimator on the full docs_gs set
	# (this is full training set or training set + validation set)
	self.bestEstimator.fit( docs_gs, y_gs)

	self.bestVectorizer = self.bestEstimator.named_steps['vectorizer']
	self.bestClassifier = self.bestEstimator.named_steps['classifier']
	self.featureEvaluator = self.bestEstimator.named_steps.get( \
						    'featureEvaluator',None)
	if self.featureEvaluator == None: self.featureValues = None
	else: self.featureValues = self.featureEvaluator.getValues()

	# run estimator on the training, val, test sets so we can compare
	self.y_predicted_train = self.bestEstimator.predict(self.docs_train)

	if self.haveValSet:
	    self.y_predicted_val  = self.bestEstimator.predict(self.docs_val)

	self.y_predicted_test  = self.bestEstimator.predict(self.docs_test)

	return self		# customary for fit() methods
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
	    output += getTopFeaturesReport( \
		    getOrderedFeatures(self.bestVectorizer,self.bestClassifier),
		    nTopFeatures) 

	    output += getVectorizerReport(self.bestVectorizer,
					    nFeatures=nFeaturesReport)

	    # false positives/negatives report.
	    # Should this be for validation set or test or both?
	    falsePos,falseNeg = skHelper.getFalsePosNeg( self.y_test,
					self.y_predicted_test,
					self.sampleNames_test,
					positiveClass=self.yClassToScore)

	    output += getFalsePosNegReport( "Test set", falsePos, falseNeg,
					    num=nFalsePosNegReport)
#	    FIXME:  this report is going to have to be rethought
#		right now, we don't have self.dataSet to use
#	    output += getTrainTestSplitReport(self.dataSet.target,self.y_train,
#						self.y_test, self.testSplit)

	output += self.getReportEnd()
	return output
    # ---------------------------


    def getReportStart(self):

	output = SSTART + "Start Time %s  %s" % (self.time, sys.argv[0])
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
	return SSTART + "End Time %s\n" % getFormattedTime()
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
	    (self.time,
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
	    self.predFilePrefix + "_train.out",
	    self.bestEstimator,
	    self.docs_train,
	    self.sampleNames_train,
	    self.y_train,
	    self.y_predicted_train,
	    classNames=self.yClassNames,
	    positiveClass=self.yClassToScore,
	    )
	# JIM: include validation set predictions too?
	writePredictionFile( \
	    self.predFilePrefix + "_test.out",
	    self.bestEstimator,
	    self.docs_test,
	    self.sampleNames_test,
	    self.y_test,
	    self.y_predicted_test,
	    classNames=self.yClassNames,
	    positiveClass=self.yClassToScore,
	    )
# ---------------------------
# end class TextPipelineTuningHelper
# ---------------------------
# ---------------------------

# FIXME: the next two classes are oddly structured. Need to rethink
# Probably: change DocumentSet to take a path parameter and load the
#           doc set. (that's it. just load the document set)
#           Then DocumentSetLoader should be responsible for splitting
#		training and test sets.

class DocumentSet (object):
    # a data structure with 3 parallel lists:  docs, y, sampleNames
    def __init__(self, docs=[], y=[], sampleNames=[]):
	self.docs = docs
	self.y = y
	self.sampleNames = sampleNames
# ---------------------------

class DocumentSetLoader (object):
    """
    Load training, validation, and test data sets from files or directories.
    Assumes the trainingSetPath is the path to a non-empty dataset.
    Validation set may be empty (path = None).
    If test set path is None, a random test set will be pulled out of the
	training set using testSplit and randomSeed
	(so upon instantiation, we will always have a trainingSet and testSet)
    You can access these sets like:
	foo = DocumentSetLoader( trainingSetPath, valSetPath, testSetPath)
	# foo.trainingSet.docs
	# foo.trainingSet.y
	# foo.trainingSet.sampleNames
	# foo.testSet.docs
	# ...
    FIXME: need to document this class and methods better
    """
    def __init__(self,
		trainingSetPath,
		validationSetPath=None,
		testSetPath=None,
		testSplit=0.20,
		randomSeed=None,
		):
	self.validationSet = self.getDocSet(validationSetPath)
	self.trainingSet   = self.getDocSet(trainingSetPath)
	self.testSet       = self.getDocSet(testSetPath)

	if self.testSet == None:
	    # split training set into random training/test sets

		# sample names
		# documents (strings) themselves
		# correct classifications (labels) for the samples
	    sampleNames_train, sampleNames_test,	\
	    docs_train,        docs_test,		\
	    y_train,           y_test = train_test_split( \
						self.trainingSet.sampleNames,
						self.trainingSet.docs,
						self.trainingSet.y,
						test_size=testSplit,
						random_state=randomSeed)
	    self.trainingSet = DocumentSet(docs_train,y_train,sampleNames_train)
	    self.testSet = DocumentSet(docs_test, y_test, sampleNames_test)
    # ---------------------------

    def getDocSet(self, path):
	if path == None:  return None
	path = os.path.realpath(path)
	if os.path.isdir(path):
	    docSet = self.getDocSetFromDir(path)
	else:
	    docSet = self.getDocSetFromFile(path)
	return docSet
    # ---------------------------

    def getDocSetFromDir(self, path):
	dataSet = load_files( path )
	return DocumentSet( dataSet.data, dataSet.target, 
				self.computeSampleNames( dataSet.filenames) )
    # ---------------------------

    def computeSampleNames(self, filenames):
	''' Convert list of filenames into Samplenames '''
	return [ os.path.basename(fn) for fn in filenames ]
    # ---------------------------

    def getDocSetFromFile(self, path):
	# read sample file
	docs = []		# docs in the sample file
	y = []			# their y values (0 or 1)
	sampleNames = []	# their sample names

        rcds = open(path,'r').read().split(sampleDataLib.RECORDSEP)

	del rcds[0]		# header line
        del rcds[-1] 		# empty string after end of split

	for rcd in rcds:
	    sr = sampleDataLib.SampleRecord(rcd)
	    docs.append(sr.getDocument())
	    y.append   (sr.getKnownYvalue())
	    sampleNames.append(sr.getSampleName())

	return DocumentSet( docs, np.array(y), sampleNames)
	#return DocumentSet( docs, y, sampleNames)
    # ---------------------------

# end class DocumentSetLoader ---------------------------


class FeatureDocCounter(BaseEstimator, TransformerMixin):
    """
    An sklearn Estimator that gives you access to the number of documents each
        feature occurs in.
    Put this "Estimator" into your pipeline after the vectorizer,
    then Access this list of counts via getValues()
    """
    def transform(self, X):
	'''don't actually transform X, just gather the counts.'''
        # print type(X)
        # I wish I understood scipy.sparse matrices and/or numpy.matrix,
        #   but these magic words seem to work (after trial and error)
        self.docCounts = X.sum(axis=0,dtype=int).tolist()[0]
        return X

    def fit(self, X, y=None, **fit_params):
	'''nothing to actually fit'''
        return self

    def getValues(self):
        """ Return list of counts, one integer for each feature,
            in feature order.
            Each count is the number of docs the feature occurs in.
        """
        return self.docCounts
# end FeatureDocCounter ----------

############################################################
# Functions to format output reports and other things.
# These are functions that concievably could be useful outside on their own.
# SO these do not use args or config variables defined above.
############################################################
def writePredictionFile( \
    fileName,		# file to write to
    estimator,		# the trained model to use
    docs,		# the documents to predict
    sampleNames,	# sample names for those docs
    y_true,		# true labels/classifications for those docs 0|1
    y_predicted,	# predicted labels/classifications for those docs 0|1
    classNames=['no','yes'],
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

    if hasattr(estimator, "decision_function"):
	conf = estimator.decision_function(docs).tolist()
	absConf = map(abs, conf)
	predictions = \
	    zip(sampleNames, trueNames, predNames, predTypes, conf, absConf)

	# sort by absolute value of confidence
	selConf = lambda x: x[5]	# select confidence value 
	predictions = sorted(predictions, key=selConf, reverse=True)

	header = "ID\tTrue Class\tPred Class\tFP/FN\tConfidence\tAbs Value\n"
	template = "%s\t%s\t%s\t%s\t%5.3f\t%5.3f\n"
    else:			# no confidence values available
	predictions = zip(sampleNames, trueNames, predNames, predTypes)

	header = "ID\tTrue Class\tPred Class\FP/FN\n"
	template = "%s\t%s\t%s\t%s\n"

    with open(fileName, 'w') as fp:
	fp.write(header)
	for p in predictions:
	    fp.write(template % p)
    return
# ---------------------------

def getTrainTestSplitReport( \
	y_all,		# the y_values for the entire sample set
	y_train,	# ... for the training set
	y_test,		# ... for the test set
	testSplit	# float, fraction of samples to use for the test set
	):
    '''
    Report on the sizes and makeup of the training and test sets
    FIXME:  this is very yucky code...
    '''
    output = SSTART+ 'Train Test Split Report, test %% = %4.2f\n' % (testSplit)
    output += \
    "All Samples: %6d\tTraining Samples: %6d\tTest Samples: %6d\n" \
		    % (len(y_all), len(y_train), len(y_test))
    nYesAll = y_all.tolist().count(1)
    nYesTra = y_train.tolist().count(1)
    nYesTes = y_test.tolist().count(1)
    output += \
    "Yes count:   %6d\tYes count:        %6d\tYes count:    %6d\n" \
		    % (nYesAll, nYesTra, nYesTes)
    output += \
    "No  count:   %6d\tNo  count:        %6d\tNo  count:    %6d\n" \
		    % (y_all.tolist().count(0),
		       y_train.tolist().count(0),
		       y_test.tolist().count(0)  )
    output += \
    "Percent Yes:    %2.2d%%\tPercent Yes:         %2.2d%%\tPercent Yes:     %2.2d%%\n" \
		    % (100 * nYesAll/len(y_all),
		       100 * nYesTra/len(y_train),
		       100 * nYesTes/len(y_test) )
    return output
# ---------------------------

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
    output += "%s F%d: %5.3f (%s)\n\n" % \
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

def getOrderedFeatures( vectorizer,	# fitted vectorizer from a pipeline
			classifier	# trained classifier from a pipeline
    ):
    '''
    Return list of pairs, [ (feature, coef), (feature, coef), ... ]
	ordered from highest coef to lowest.
    Assumes:  vectorizer has get_feature_names() method
    '''
    if not hasattr(classifier, 'coef_'): # not all have coef's
	return []

    coefficients = classifier.coef_[0].tolist()
    featureNames = vectorizer.get_feature_names()
    
    pairList = zip(featureNames, coefficients)

    selCoef = lambda x: x[1]	# select the coefficient in the pair
    return sorted(pairList, key=selCoef, reverse=True)
# ----------------------------

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

def writeFeatures( vectorizer,	# fitted vectorizer from a pipeline
		    classifier,	# fitted classifier from the pipeline
		    fileName,	# to write to
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

    with open(fileName, 'w') as fp:
	for i,f in enumerate(featureNames):
	    fp.write(f)
	    if hasValues: fp.write(delimiter + '%d' % values[i])
	    if hasCoef:   fp.write(delimiter + '%+5.4f' % coefficients[i])
	    fp.write('\n')
    return 
# ----------------------------

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

if __name__ == "__main__":
    # ad hoc test code
    if True:	# DocumentSetLoader testing
	d = DocumentSetLoader('Data/smallset/figureText.txt',
				validationSetPath='data/smallSet',
				testSetPath='data/smallSet',
				testSplit=0.1, randomSeed=None)
	print len(d.trainingSet.docs)
	print d.trainingSet.y[:5]
	print d.trainingSet.sampleNames[:5]
	print len(d.testSet.docs)
	print d.testSet.y[:5]
	print d.testSet.sampleNames[:5]
	print type(d.testSet.y)
	if d.validationSet != None:
	    print len(d.validationSet.docs)
	    print d.validationSet.y[:5]
	    print d.validationSet.sampleNames[:5]
