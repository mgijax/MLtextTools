'''
Support for "tuning scripts" for text machine learning projects.

Separate from this module, there are tuning scripts that define Pipelines and
parameter options to try/compare via GridSearchCV.

The idea is that, as much as possible, all code is here for these scripts, and
only the Pipelines and parameters to try are in the tuning scripts.

So the code here is coupled with TuningTemplate.py.

The biggest part of this module is the implementation of various tuning
reports used to analyze the tuning runs.

Assumes:
* Tuning a binary text classification pipeline (maybe this assumption can go?)
* with class labels yes and no
* The text sample data is in sklearn load_files directory structure
* Topmost directory in that folder is specified in config file or cmd line arg
* We are Tuning a Pipeline via GridSearchCV
* The Pipeline has named steps: 'vectorizer', 'classifier' with their
*   obvious meanings (may have other steps too)
* We are scoring GridSearchCV Pipeline parameter runs via an F-Score
*   (beta is a parameter to this library)
* probably other things...
*
If the 'classifier' supports getting weighted coefficients via
classifier.coef_, then the output from here can include a TopFeaturesReport

Convention: trying to use camelCase for all the names here, but
    sklearn typically_uses_names with underscores.
'''
import sys
import time
import re
import string
import os
import os.path
import argparse
import ConfigParser

import sklearnHelperLib as skhelper

import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, precision_score,\
			recall_score, classification_report, confusion_matrix
#-----------------------------------

# in the list of classifications labels for evaluating text data
# FIXME: Can these be replaced by CLASS_NAMES in config file?
LABELS = [ 1,0]		#[ INDEX_OF_YES, INDEX_OF_NO ]
TARGET_NAMES = ['yes', 'no']

# ---------------------------
# Common command line parameter handling for the tuning scripts
# ---------------------------

def parseCmdLine():
    """
    shared among the ModelTuning scripts
    Handles cmdline args and config file, returning dict that combines them.
    Keys:
    trainingData:	dir path to training data
    verbose:		boolean - print longer tuning report
    randForSplit:	int
    randForClassifier:	int
    tuningIndexFile:	index file name to write to
    wIndex:		boolean - write indexfile
    wPredictions:	boolean - write predictions file
    predFilePrefix:	filename prefix for the prediction output files
    indexOfYes:		int
    indexOfNo:		int
    gridSearchBeta:	int
    compareBeta:	int
    testSplit:		float
    gridSearchCV:	int
    """
    cp = ConfigParser.ConfigParser()
    cp.optionxform = str # make keys case sensitive

    # generate a path up multiple parent directories to search for config file
    cl = ['/'.join(l)+'/config.cfg' for l in [['.']]+[['..']*i for i in range(1,6)]]
    cp.read(cl)

    # config file params that are defaults for command line options
    TRAINING_DATA     = cp.get     ("DEFAULT", "TRAINING_DATA")
    TUNING_INDEX_FILE = cp.get     ("MODEL_TUNING", "TUNING_INDEX_FILE")
    PRED_OUTPUT_FILE_PREFIX = cp.get("MODEL_TUNING", "PRED_OUTPUT_FILE_PREFIX")

    parser = argparse.ArgumentParser( \
    description='Run a tuning experiment script.')

    parser.add_argument('-d', '--data', dest='trainingData',
            default=TRAINING_DATA,
            help='Directory where training data files live. Default: "%s"' \
                    % TRAINING_DATA)

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='verbose: print longer tuning report.')

    parser.add_argument('--rsplit', dest='randForSplit',
			default=None, type=int,
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
    args =  parser.parse_args()

    # config params that are not cmdline args (yet)
    args.indexOfYes      = cp.getint  ("DEFAULT",      "INDEX_OF_YES")
    args.indexOfNo       = cp.getint  ("DEFAULT",      "INDEX_OF_NO")
    args.gridSearchBeta  = cp.getint  ("MODEL_TUNING", "GRIDSEARCH_BETA")
    args.compareBeta     = cp.getint  ("MODEL_TUNING", "COMPARE_BETA")
    args.testSplit       = cp.getfloat("MODEL_TUNING", "TEST_SPLIT")
    args.gridSearchCV    = cp.getint  ("MODEL_TUNING", "GRIDSEARCH_CV")
    args.classNames      = eval(cp.get("DEFAULT",      "CLASS_NAMES"))

    return args
# ---------------------------
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
# ---------------------------

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
# Main class
# ---------------------------

class TextPipelineTuningHelper (object):

    def __init__(self,
	pipeline,
	pipelineParameters,
	trainingDataDir,
	testSplit=0.20,		# size of the test set from the dataSet
	gridSearchBeta=4,	# beta to use in the gridsearch
	gridSearchCV=5,		# number of cross validation folds to use
	gsVerbose=1,		# verbose setting for grid search (to stdout)
	indexOfYes=1,		# index of the "yes" label in the dataset
	randomSeeds={'randForSplit':1},	# random seeds. Assume all are not None
	):

	self.pipeline = pipeline
	self.pipelineParameters = pipelineParameters

	self.gridSearchBeta = gridSearchBeta
	self.indexOfYes = indexOfYes
	self.trainingDataDir = trainingDataDir

	scorer = make_scorer(fbeta_score, beta=gridSearchBeta,
							pos_label=indexOfYes)

	self.gs = GridSearchCV(pipeline,
				pipelineParameters,
				scoring=scorer,
				cv=gridSearchCV,
				verbose=gsVerbose,
				n_jobs=-1,
				)
	self.testSplit = testSplit
	self.randomSeeds = randomSeeds
	self.randForSplit = randomSeeds['randForSplit']	# required seed

	self.time = getFormattedTime()
	self.readDataSet()
	self.computeSampleNames()
    #---------------------

    def readDataSet(self):
	self.dataSet = load_files( self.trainingDataDir )
    # ---------------------------

    def computeSampleNames(self):
	'''
	Convert list of filenames into Samplenames
	'''
	self.sampleNames = [ os.path.basename(fn) for fn \
						    in self.dataSet.filenames ]
    # ---------------------------

    def fit(self):
	'''
	run the GridSearchCV
	'''
	# using _train _test variable names as is the custom in sklearn.
	# "y_" are the correct classifications (labels) for the corresponding
	#   samples

	    # sample names
	    # documents (strings) themselves
	    # correct classifications (labels) for the samples
	self.sampleNames_train, self.sampleNames_test,	\
	self.docs_train,        self.docs_test,	\
	self.y_train,           self.y_test = train_test_split( \
					    self.sampleNames,
					    self.dataSet.data,
					    self.dataSet.target,
					    test_size=self.testSplit,
					    random_state=self.randForSplit)

	self.gs.fit( self.docs_train, self.y_train )	# DO THE GRIDSEARCH

	# FIXME: maybe Should implement getter's for these
	self.bestEstimator  = self.gs.best_estimator_
	self.bestVectorizer = self.bestEstimator.named_steps['vectorizer']
	self.bestClassifier = self.bestEstimator.named_steps['classifier']

	# run estimator on both the training set and test set so we can compare
	self.y_predicted_train = self.bestEstimator.predict(self.docs_train)
	self.y_predicted_test  = self.bestEstimator.predict(self.docs_test)

	return self		# customary for fit() methods
    # ---------------------------

    def getFalsePosNeg( self):
	'''
	Return lists of (sample names of) false positives and false negatives
	    in a test set
	'''
	y_true      = self.y_test
	y_predicted = self.y_predicted_test
	sampleNames = self.sampleNames_test

	falsePositives = []
	falseNegatives = []

	for trueY, predY, name in zip(y_true, y_predicted, sampleNames):
	    predType = skhelper.predictionType(trueY, predY)
	    if predType == 'FP':
		falsePositives.append(name)
	    elif predType == 'FN':
		falseNegatives.append(name)

	return falsePositives, falseNegatives
    # ---------------------------

    def getReports(self,
		    nFeaturesReport=10,		# in features report
		    nFalsePosNegReport=5,	# for false pos/neg rpt
		    nTopFeatures=20,		# num of top weighted to rpt
		    wIndex=False,		# write tuningIndex file
		    tuningIndexFile='index.out',
		    wPredictions=False,		# write predictions file
		    predFilePrefix='predictions',
		    compareBeta=4,	# Fbeta to report in tuningindex file
		    verbose=False,
		    classNames=['no','yes'] # map class indexes to names
	):

	if wIndex: self.writeIndexFile(tuningIndexFile, compareBeta)
	if wPredictions: self.writePredictions(predFilePrefix, classNames)

	output = getReportStart(self.time,self.gridSearchBeta,self.randomSeeds,
				self.trainingDataDir, wIndex, tuningIndexFile)

	output += getFormattedMetrics("Training Set", self.y_train,
				    self.y_predicted_train, compareBeta)
	output += getFormattedMetrics("Test Set", self.y_test,
				    self.y_predicted_test, compareBeta)
	output += getBestParamsReport(self.gs, self.pipelineParameters)
	output += getGridSearchReport(self.gs, self.pipelineParameters)

	if verbose: 
	    output += getTopFeaturesReport( \
		    getOrderedFeatures(self.bestVectorizer,self.bestClassifier),
		    nTopFeatures) 

	    output += getVectorizerReport(self.bestVectorizer,
					    nFeatures=nFeaturesReport)

	    falsePos, falseNeg = self.getFalsePosNeg()
	    output += getFalsePosNegReport( falsePos, falseNeg,
						num=nFalsePosNegReport)

	    output += getTrainTestSplitReport(self.dataSet.target,self.y_train,
						self.y_test, self.testSplit)

	output += getReportEnd()
	return output
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
	    precision_score( y_true, y_predicted, pos_label=self.indexOfYes),
	    recall_score( y_true, y_predicted, pos_label=self.indexOfYes),
	    fbeta_score( y_true, y_predicted, compareBeta,
						    pos_label=self.indexOfYes), 
	    fbeta_score( y_true, y_predicted, 1, pos_label=self.indexOfYes), 
	    tuningFile,
	    ) )
# ---------------------------

    def writePredictions(self, predFilePrefix, classNames):
	'''
	Write files with predictions from training set and test set
	'''
	writePredictionFile( \
	    predFilePrefix + "_train.out",
	    self.bestEstimator,
	    self.docs_train,
	    self.sampleNames_train,
	    self.y_train,
	    self.y_predicted_train,
	    classNames=classNames,
	    )
	writePredictionFile( \
	    predFilePrefix + "_test.out",
	    self.bestEstimator,
	    self.docs_test,
	    self.sampleNames_test,
	    self.y_test,
	    self.y_predicted_test,
	    classNames=classNames,
	    )
# ---------------------------
# end class TextPipelineTuningHelper
# ---------------------------

def writePredictionFile( \
    fileName,		# file to write to
    estimator,		# the trained model to use
    docs,		# the documents to predict
    sampleNames,	# sample names for those docs
    y_true,		# true labels/classifications for those docs 0|1
    y_predicted,	# predicted labels/classifications for those docs 0|1
    classNames=['no','yes'],
    ):
    '''
    Write a prediction file, with confidence values if available.
    Prediction file has a line for each doc,
	samplename, y_true, y_predicted, FP/FN, [confidence, abs value]
    '''
    predTypes = [skhelper.predictionType(t,p) for t,p in zip(y_true,y_predicted)]
    
    trueNames = [ classNames[y] for y in y_true ]
    predNames = [ classNames[y] for y in y_predicted ]

    if hasattr(estimator, "decision_function"):
	conf = estimator.decision_function(docs).tolist()
	absConf = map(abs, conf)
	predictions = \
	    zip(sampleNames, trueNames, predNames, predTypes, conf, absConf)

	selConf = lambda x: x[5]	# select confidence value 
	predictions = sorted(predictions, key=selConf)

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

# ---------------------------
# Functions to format output reports
# ---------------------------
SSTART = "### "			# output section start delimiter

def getReportStart(curtime,beta, randomSeeds,dataDir, wIndex, tuningIndexFile):

    output = SSTART + "Start Time %s  %s" % (curtime, sys.argv[0])
    if wIndex: output += "\tindex file: %s" % tuningIndexFile
    output += "\n"
    output += "Data dir: %s,\tGridSearch Beta: %d\n" % (dataDir, beta)
    output += getRandomSeedReport(randomSeeds)
    output += "\n"
    return output
# ---------------------------

def getReportEnd():
    return SSTART + "End Time %s\n" % getFormattedTime()
# ---------------------------

def getTrainTestSplitReport( \
	y_all,
	y_train,
	y_test,
	testSplit
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
    gs,	    # sklearn.model_selection.GridsearchCV that has been .fit()
    parameters  # dict of parameters used in the gridsearch
    ):
    output = SSTART +'Best Pipeline Parameters:\n'
    for pName in sorted(parameters.keys()):
	output += "%s: %r\n" % ( pName, gs.best_params_[pName] )

    output += "\n"
    return output
# ---------------------------

def getGridSearchReport( \
    gs,	    # sklearn.model_selection.GridsearchCV that has been .fit()
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
    falsePositives,	# list of (sample names) of the falsePositives
    falseNegatives,	# ... 
    num=5		# number of false pos/negs to display
    ):
    '''
    Report on the false positives and false negatives in a test set
    '''
    # method to get false positives and falsenegativess lists

    output = SSTART + "False positives: %d\n" % len(falsePositives)
    for name in falsePositives[:num]:
	output += "%s\n" % name

    output += "\n"
    output += SSTART + "False negatives: %d\n" % len(falseNegatives)
    for name in falseNegatives[:num]:
	output += "%s\n" % name

    output += "\n"
    return output
# ---------------------------

def getFormattedMetrics( \
	title,		# string title
	y_true,		# true category assignments
	y_predicted,	# predicted assignments
	beta,		# for the fbeta_score
	indexOfYes=1	# index of the "yes" label
	):
    '''
    Return formated metrics report
    y_true and y_predicted are lists of integer category indexes.
    Assumes we are using a fbeta score, not a good thing in the long term
    '''
    # concat title string onto all the target category names so
    #  they are easier to differentiate in multiple reports (and you can
    #  grep for them)
    target_names = [ "%s %s" % (title[:5], x) for x in TARGET_NAMES ]

    output = SSTART + "Metrics: %s\n" % title
    output += "%s\n" % (classification_report( \
			y_true, y_predicted,
			labels=LABELS[:1], target_names=target_names[:1])
		    ) # only report on first label: "yes"
    output += "%s F%d: %5.3f\n\n" % (title[:5],beta,
				fbeta_score( y_true, y_predicted, beta,
					     pos_label=indexOfYes ) )
    output += "%s\n" % getFormattedCM(y_true, y_predicted)

    return output
# ---------------------------

def getFormattedCM( \
    y_true,	# true category assignments for test set
    y_predicted	# predicted assignments
    ):
    '''
    Return (minorly) formated confusion matrix
    FIXME: this could be greatly improved
    '''
    output = "%s\n%s\n" % ( \
		str( TARGET_NAMES ),
		str( confusion_matrix(y_true, y_predicted, labels=LABELS) ) )
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

def getFormattedTime():
    return time.strftime("%Y/%m/%d-%H-%M-%S")
