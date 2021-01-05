'''
Support for "tuning scripts" for text machine learning projects.

Idea:
Separate from this module, there are tuning scripts that define Pipelines and
parameter options to try and evaluate. (see TuningTemplate.py)

As much as possible, all code is here for these scripts, and only the
pipeline definition & parameter options to evaluate are in the tuning scripts.

So the code here is coupled with TuningTemplate.py.

A big part of this module is the implementation of various tuning
reports used to analyze the tuning runs. Many of the formatting routines
are in tuningReportsLib.py.

Assumes:
1 Pipelines are for binary classification
2 Pipelines have named steps: 'vectorizer', 'classifier' with their
    obvious meanings (may have other steps too)
3 The text sample data is in sklearn load_files directory structure
     (haven't used this approach in a while so it hasn't been tested recently)
    OR in files parsable by a sampleDataLib.py that encapsulates knowledge of
    the specific document type/set we are dealing with (e.g., how to parse
    data files and break records into documents, etc.)
    - the actual filename is a config parameter

    In this latter case, assumes we can import the sampleDataLib

4 We are scoring Pipeline parameter runs via an F-Score
    (beta is a parameter to this library)
5 probably other things...

Evaluating Pipeline Parameters:
Two Approaches are supported/implemented
    (1) you have a known sample set.
        (a) Randomly splits this into a training set and a validation set
        (you could treat the validation set as a test set in the simplest
        cases where you don't expect to iterate and experiment too much).
        (b) Uses k-fold cross validation via GridSearchCV on the training set
        to optimize any pipeline parameters.  (if there is only one set of
        parameters, skips this step and goes to (c))
        (c) Using the optimal parameters, retrains on the whole (non-cv'ed)
        training set (GridSearchCV does this by default).
        (d) Predicts, evaluates, and reports results against the validation set

        This is most appropriate when the validation set and training sets
        come from the same distribution and you don't necessarily want to
        maintain a validation set - maybe for quick and dirty projects.

    (2) you provide a validation set and training set.
        (a) Runs GridSearchCV on the training set to optimize any pipeline
        parameters against your validation set- no cv.
        (if there is only 1 set of parameters, skips this step and goes to (b))
        (b) Using the best pipeline parameters, trains on the training set.
        (c) Predicts, evaluates, and reports results against the validation set.

        This is most appropriate when your validation set comes from a
        different distribution than the training set and/or you want to
        maintain a validation set.

    (3) (not implemented) you provide a validation set and training set.
        (a) Uses k-fold cross validation via GridSearchCV on the training set
        to optimize any pipeline parameters. (if there is only one set of
        parameters, skips this step and goes to (b))
        (b) Using the optimal parameters, retrains on the whole (non-cv'ed)
        training set (GridSearchCV does this by default).
        (c) Predicts, evaluates, and reports results against the validation set.

        This seems intermediate between (1) and (2). Seems kind of weird NOT
        to optimize the pipeline parameters against your validation set, but
        maybe this makes sense sometimes.

    In any of the above approaches, you may optionally provide a separate test
    set to predict using an additional trained model.
    In this case the model uses the optimal parameters from above
    but is trained on the training + validation sets (to use all available
    classified samples).

    You can use these results to compare to the validation set results, but
    you should not use these test results to make decisions about next steps
    to try since this would lead toward tuning your model for the specific
    test set and making your model less likely to generalize to future inputs.


Other functionality:
* Many options/parameters for this process are specified in a config file
    and/or command line arguments (to the tuning script using this module)
    that this module processes.

* Support for generating/reporting random seeds or using fixed specified seeds.
    (for random splits and training seeds)

* Supports use of a custom pipeline step, FeatureDocCounter (defined in
    sklearnHelperLib), which you can put into a pipeline just after the
    vectorizer step, and it will give you access to
    the number of documents each feature appears in.

Outputs:
* log of the tuning run to stdout. This has lots of subsections like
    prediction metrics
    best pipeline parameters
    top pos/neg weighted features (If the 'classifier' supports getting
                                    weighted coefficients)
    vectorizer report
    false positives/negatives
    sample set size report
* (optional) write a 1-line metrics summary line to an index file
        (usually index.out)
* (optional) write features + their weights to a feature file
* (optional) write training, validation set prediction files
* (optional) write pkl file with the trained model. This is the model using the
    optimal parameters from above but is trained on the training + validation
    set.

Configuration and sampleDataLib:
A configparser configuration file named 'tuning.cfg' defines various tuning
parameters.

    Param "SAMPLE_DATA_LIB" specifies how to access the training/validation sets
    and is either some filename.py or None
    - specified in config file
    - default is "sampleDataLib.py"

    (1) if SAMPLE_DATA_LIB is some filename.py (not "None"), we have a
        sampleDataLib that knows how to read classified sample files
        - Needs to be importable on the PYTHONPATH
        - sampleDataLib needs to define 2 classes:
            - The SAMPLE_OBJ_TYPE class that implements the Sample objects
                represented in the input sample files.
                - typically this class will be a subclass of ClassifiedSample
                - This class should implement class methods:
                    getClassNames()   
                    getY_positive()
                    getY_negative()
            - The ClassifiedSampleSet class that knows how to read Sample files

    (2) if SAMPLE_DATA_LIB is "None", we use sklearn's load_files mechanism,
        and the config file defines:
        y_class_names (e.g., ['no', 'yes'])
            The labels matching y_values from the training set:
                y_class_names[y_val]= name
            These should be the classification labels in alpha order
            These match training set directory names used by
                sklearn.datasets.load_files()

        y_class_to_score (e.g., 1)
            The index in y_class_names to score (typically the positive class),
                i.e., compute precision, recall, f-score, etc.
            This class is used in the grid search scoring to select the best
                model.

        rpt_class_names  (e.g., ['yes', 'no'])
            Order + labels we want to report in confusion matrix and other rpts.

        rpt_class_mapping (e.g., [1, 0])
            List of y_values to rpt in confusion matrix and other reports.
            rpt_class_mapping[y_val] maps to rpt_class_names[]

    See parseCmdLine() below for other expected configuration parameters.

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
import pickle
import argparse

import utilsLib
import sklearnHelperLib as skHelper
import tuningReportsLib as trl

from sklearn.base import clone
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, precision_score, \
                                                            recall_score

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
    config = utilsLib.getConfig('tuning.cfg')

    # config file params that are defaults for command line options
    TRAINING_SET      = config.get("TRAINING_DATA", "TRAINING_SET")
    VALIDATION_SET    = config.get("TRAINING_DATA", "VALIDATION_SET")
    TUNING_INDEX_FILE = config.get("MODEL_TUNING",  "TUNING_INDEX_FILE")

    basename = os.path.basename(sys.argv[0])
    OUTPUT_FILE_PREFIX = os.path.splitext(basename)[0]

    # command line parameters
    parser = argparse.ArgumentParser( \
    description='Run a tuning experiment script.')

    parser.add_argument('-d', '--trainpath', dest='trainDataPath',
            default=TRAINING_SET,
            help='pathname where training set lives. Default: "%s"' \
                    % TRAINING_SET)

    parser.add_argument('--valpath', dest='valDataPath',
            default=VALIDATION_SET,
            help='pathname where validation set lives. Default: "%s"' \
                    % VALIDATION_SET)

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
    parser.add_argument('--nopredict', dest='wPredictions',
                        action='store_false', default=False,
            help="don't write validation & training set predictions (default)")

    parser.add_argument('-p', '--predict', dest='wPredictions',
                        action='store_true', default=False,
                        help='write predictions for validation & training sets')

    parser.add_argument('--noconfidence', dest='doConfidences',
                        action='store_false',
                        help="don't include confidence values w/ predictions")

    parser.add_argument('--outprefix', dest='outputFilePrefix', 
                        default=OUTPUT_FILE_PREFIX,
            help='filename prefix for features & preds files. Default: %s' % \
                                                    OUTPUT_FILE_PREFIX)
    parser.add_argument('--features', dest='writeFeatures',
                        action='store_true', default=False,
                        help="write feature file" )
    parser.add_argument('--savemodel', dest='modelFile', default=None,
            help='pickle/save trained model in this file. Default: no save')

    args =  parser.parse_args()

    # config params that are not cmdline args (yet)
    args.sampleDataLibFile = config.get("TRAINING_DATA", "SAMPLE_DATA_LIB",
                                                fallback="sampleDataLib.py")
    args.sampleObjTypeName = config.get("TRAINING_DATA", "SAMPLE_OBJ_TYPE_NAME",
                                                fallback="ClassifiedSample")
    args.yClassNames     = eval(config.get("CLASS_NAMES", "y_class_names",
                                                fallback="['no', 'yes']"))
    args.yClassToScore   = config.getint  ("CLASS_NAMES", "y_class_to_score",
                                                fallback=1)
    args.rptClassNames   = eval(config.get("CLASS_NAMES", "rpt_class_names",
                                                fallback="['yes', 'no']"))
    args.rptClassMapping = eval(config.get("CLASS_NAMES", "rpt_class_mapping",
                                                fallback="[1, 0]"))
    args.rptNum   = config.getint("CLASS_NAMES","rpt_classification_report_num",
                                                fallback=2)

    args.gridSearchBeta  = config.getint(  "MODEL_TUNING", "GRIDSEARCH_BETA",
                                                                fallback=1)
    args.compareBeta     = config.getint(  "MODEL_TUNING", "COMPARE_BETA",
                                                                fallback=1)
    args.validationSplit = config.getfloat("MODEL_TUNING", "VALIDATION_SPLIT",
                                                                fallback=0.20)
    args.numCV           = config.getint(  "MODEL_TUNING", "NUM_CV",
                                                                fallback=5)
    args.numJobs         = config.getint(  "MODEL_TUNING", "NUM_JOBS",
                                                                fallback=1)
    return args
# ---------------------------

args = parseCmdLine()	# make args available to this code and importers

if args.sampleDataLibFile != "None":    # import the sampleDataLib
    sampleDataLib = utilsLib.importPyFile(args.sampleDataLibFile)

############################################################
# Main class - encapsulates the tuning process
############################################################

class TextPipelineTuningHelper (object):

    def __init__(self,
        pipeline,
        pipelineParameters,
        randomSeeds={'randForSplit':1},	# random seeds. Assume all are not None
        note='',
        ):
        self.pipeline           = pipeline
        self.pipelineParameters = pipelineParameters
        self.randomSeeds        = randomSeeds
        self.randForSplit       = randomSeeds['randForSplit']	# required seed
        self.note		= note

        # Only this constructor accesses args
        self.trainDataPath      = os.path.abspath(args.trainDataPath)

        self.valDataPath        = args.valDataPath
        if self.valDataPath == "None":
            self.valDataPath = None
        if self.valDataPath:
            self.valDataPath = os.path.abspath(args.valDataPath)

        self.testDataPath       = args.testDataPath
        if self.testDataPath:
            self.testDataPath = os.path.abspath(args.testDataPath)

        self.validationSplit    = args.validationSplit
        self.gridSearchBeta     = args.gridSearchBeta
        self.numJobs		= args.numJobs
        self.numCV              = args.numCV

        self.tuningIndexFile    = args.tuningIndexFile
        self.wIndex             = args.wIndex
        self.wPredictions       = args.wPredictions
        self.doConfidences	= args.doConfidences
        self.outputFilePrefix   = args.outputFilePrefix
        self.writeFeatures      = args.writeFeatures
        self.modelFile		= args.modelFile
        self.compareBeta        = args.compareBeta
        self.verbose            = args.verbose
        self.gsVerbose          = args.gsVerbose

        self.sampleDataLibFile  = args.sampleDataLibFile
        if self.sampleDataLibFile != "None":    # have a sampleDataLib
            sampleObjType = getattr(sampleDataLib,args.sampleObjTypeName)
            yClassNames   = sampleObjType.getClassNames()

            # get class names, etc. from sampleObjType
            self.yClassNames     = yClassNames
            self.yClassToScore   = sampleObjType.getY_positive()
            self.rptClassNames = [yClassNames[sampleObjType.getY_positive()],
                                  yClassNames[sampleObjType.getY_negative()]]
            self.rptClassMapping = [sampleObjType.getY_positive(),
                                    sampleObjType.getY_negative()]
        else:   # get from config file
            self.yClassNames     = args.yClassNames
            self.yClassToScore   = args.yClassToScore
            self.rptClassNames   = args.rptClassNames
            self.rptClassMapping = args.rptClassMapping

        self.rptNum = args.rptNum

        self.startTimeStr = trl.getFormattedTime()
        self.startTime    = time.time()
        self.scorer = make_scorer(fbeta_score, beta=self.gridSearchBeta,
                                          pos_label=self.yClassToScore)
    #---------------------

    def loadTrainValTestSets(self):
        """
        Get the training, validation, & test sets (test is optional).
        This means setting three parallel lists:
            self.sampleNames_xxx	names (IDs) of the docs
            self.docs_xxx		the docs themselves (string)
            self.y_xxx			the class name index (0, 1)
            				y_xxx is an np.array
            For xxx in "train", "val", "test"
        """
        ####### training set
        self.trainSet = DocumentSet().load(self.trainDataPath)

        ####### Validation set
        if self.valDataPath:
            self.valSet = DocumentSet().load(self.valDataPath)
        else:		# no specified val set, split random set from training	
            self.valSet = self.trainSet.split(splitSize=self.validationSplit,
                                                randomSeed=self.randForSplit)
        ####### test set
        if self.testDataPath:
            self.testSet = DocumentSet().load(self.testDataPath)
        else:
            self.testSet = None
    #---------------------

    def getGridSearchParams(self):
        """
        Figure out what doc set, y values, cv value to use for the GridSearchCV.

        Return docs, y, cv

        If we don't have a specified validation set,
            Have GridSearchCV use k-fold cross validation on the training set
            Return docs_train, y_train, and the int cv value (num of folds)

        If we do have a specified validation set, this is more subtle.
            Since we have a val set, we don't want to use cv, but
            we still want to use GridSearchCV to train on each param permutation
            and evaluate on the val set
            (so we want to use the GridSearch w/o the k-folds)

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
        if self.valDataPath:		# have a specified validation set 
            docs_gs = self.trainSet.getDocs() + self.valSet.getDocs()
            y_gs    = np.concatenate( (self.trainSet.getYvalues(),
                                                    self.valSet.getYvalues()) )

            lenTrain = self.trainSet.getNumDocs()
            lenVal   = self.valSet.getNumDocs()
            cv = [ (list(range(lenTrain)),
                    list(range(lenTrain, lenTrain+lenVal)) ), ]

        else:				# no val set, use k-fold
            docs_gs         = self.trainSet.getDocs()
            y_gs            = self.trainSet.getYvalues()
            cv              = self.numCV

        return docs_gs, y_gs, cv
    #---------------------

    def fit(self):
        '''
        Find the best pipeline params (against validation) and Set:

            self.bestParams
            self.valSetEstimator      # Trained on tr set to predict val set
                # Its subcomponents:
                self.bestVectorizer
                self.bestClassifier
                self.featureEvaluator 	# (if any)
                    self.featureValues  # (output of self.featureEvaluator)
            self.gridSearch		# or None if no grid search is run
            self.testSetEstimator     # Trained tr+val sets to predict test set
                                      # (None if not needed)
        Subtleties:
            If there is only one pipelineParameters permutation, then using
            GridSearchCV() is very inefficient:
                if given a specified validation set,
                    it trains on the full training set
                        (which we need, but don't have access to)
                    and then retrains again on the training + val set,
                        (which we may not need)
                        returning that as the best_estimator_
                if using cv,
                    it unnecessarily runs all the k-folds
                        (w/ same param permutation!)
                    and then retrains on the full training set (which we need)

            So we avoid this by handling the one permutation case separately.
        '''
        self.verboseWrite("Loading sample sets\n")
        self.loadTrainValTestSets()
        self.testSetEstimator = None	# assume we don't need this.

        if skHelper.isOneCombination(self.pipelineParameters):
            self.verboseWrite("Training valSetEstimator on single param set\n")

            self.bestParams = skHelper.convertParamGrid(self.pipelineParameters)
            self.pipeline.set_params( **self.bestParams)

            self.gridSearch = None
            self.valSetEstimator = self.pipeline
            self.valSetEstimator.fit(self.trainSet.getDocs(),
                                                    self.trainSet.getYvalues())

            if self.testSet or self.modelFile:
                self.verboseWrite("Training testSetEstimator\n")
                self.testSetEstimator = clone(self.pipeline)
                docs = self.trainSet.getDocs() + self.valSet.getDocs()
                y    = np.concatenate( (self.trainSet.getYvalues(),
                                                self.valSet.getYvalues()) )
                self.testSetEstimator.fit(docs, y)

            self.verboseWrite("Done training\n")

        else:		# multiple pipeline params to try
            # Using "_train" "_test" variable names as is the custom in sklearn.
            # "y_" are the correct classifications (labels) for the
            #   corresponding samples
            # "_gs" = grid search set

            self.verboseWrite("Starting GridSearch\n")
            docs_gs, y_gs, cv = self.getGridSearchParams()

            gs = GridSearchCV( self.pipeline,
                                self.pipelineParameters,
                                scoring= self.scorer,
                                cv=      cv,
                                refit=   True,	# if false, no best_estimator_
                                verbose= self.gsVerbose,
                                n_jobs=  self.numJobs,
                                )
            gs.fit( docs_gs, y_gs )
            self.verboseWrite("Done Gridsearch\n")

            self.gridSearch = gs
            self.bestParams = gs.best_params_

            # gs.best_estimator_ is Pipeline w/ best params evaluated on
            #   val set or cv
            #   and trained on train + val set or whole train set (all folds)

            if self.valDataPath:	# have specified validation set
                self.testSetEstimator = gs.best_estimator_	# on tr+val

                self.verboseWrite("Training valSetEstimator\n")
                self.valSetEstimator = clone(gs.best_estimator_)
                self.valSetEstimator.fit(self.trainSet.getDocs(), # on tr set
                                        self.trainSet.getYvalues())
            else:			# GridSearch on cv of training set
                self.valSetEstimator = gs.best_estimator_   # on tr set

                if self.testSet or self.modelFile:
                    self.testSetEstimator = clone(gs.best_estimator_)

                    # train on training + val sets
                    self.verboseWrite("Training testSetEstimator\n")
                    docs = self.trainSet.getDocs() + self.valSet.getDocs()
                    y    = np.concatenate( (self.trainSet.getYvalues(),
                                                self.valSet.getYvalues()) )
                    self.testSetEstimator.fit(docs, y)

            self.verboseWrite("Done with best param search and training\n")

        self.bestVectorizer   = self.valSetEstimator.named_steps['vectorizer']
        self.bestClassifier   = self.valSetEstimator.named_steps['classifier']
        self.featureEvaluator = self.valSetEstimator.named_steps.get( \
                                                    'featureEvaluator',None)
        if self.featureEvaluator == None: self.featureValues = None
        else: self.featureValues = self.featureEvaluator.getValues()

        # run estimator on the training, val sets so we can compare
        self.verboseWrite("Predicting training set using valSetEstimator\n")
        self.trainSet.setPredictions( \
                    self.valSetEstimator.predict(self.trainSet.getDocs()) )

        self.verboseWrite("Predicting validation set using valSetEstimator\n")
        self.valSet.setPredictions( \
                    self.valSetEstimator.predict(self.valSet.getDocs()) )

        if self.testSet:		# run on test set too
            self.verboseWrite("Predicting test set using testSetEstimator\n")
            self.testSet.setPredictions( \
                    self.testSetEstimator.predict(self.testSet.getDocs()) )

        self.verboseWrite("Done with predictions\n")

        return self		# customary for fit() methods
    # ---------------------------

    def verboseWrite(self, msg):
        if self.gsVerbose:
            sys.stderr.write(trl.getFormattedTime() + " " + msg)
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
        if self.modelFile:
            self.writeModel()
        if self.writeFeatures:
            featureFile = self.outputFilePrefix + "_features.txt"
            trl.writeFeatures(self.bestVectorizer, self.bestClassifier,
                                         featureFile, values=self.featureValues)
        output = self.getReportStart()

        output += trl.getFormattedMetrics("Training Set",
                                self.trainSet.getYvalues(),
                                self.trainSet.getPredictions(),self.compareBeta,
                                rptClassNames=self.rptClassNames,
                                rptClassMapping=self.rptClassMapping,
                                rptNum=self.rptNum,
                                yClassNames=self.yClassNames,
                                yClassToScore=self.yClassToScore,
                                )
        output += trl.getFormattedMetrics("Validation Set",
                                self.valSet.getYvalues(),
                                self.valSet.getPredictions(),self.compareBeta,
                                rptClassNames=self.rptClassNames,
                                rptClassMapping=self.rptClassMapping,
                                rptNum=self.rptNum,
                                yClassNames=self.yClassNames,
                                yClassToScore=self.yClassToScore,
                                )
        if self.testSet:
            output += trl.getFormattedMetrics("Test Set",
                                self.testSet.getYvalues(),
                                self.testSet.getPredictions(),self.compareBeta,
                                rptClassNames=self.rptClassNames,
                                rptClassMapping=self.rptClassMapping,
                                rptNum=self.rptNum,
                                yClassNames=self.yClassNames,
                                yClassToScore=self.yClassToScore,
                                )
        if self.note:
            output += SSTART + "Note: %s\n" % self.note

        output += trl.getBestParamsReport(self.bestParams, self.valSetEstimator)
        output += trl.getGridSearchReport(self.gridSearch,
                                                        self.pipelineParameters)

        if self.verbose: 
            features = skHelper.getOrderedFeatures( self.bestVectorizer,
                                                    self.bestClassifier)
            output += trl.getTopFeaturesReport( features, nTopFeatures) 

            output += trl.getVectorizerReport(self.bestVectorizer,
                                            nFeatures=nFeaturesReport)

            # false positives/negatives report.
            falsePos,falseNeg = skHelper.getFalsePosNeg( \
                                        self.valSet.getYvalues(),
                                        self.valSet.getPredictions(),
                                        self.valSet.getSampleNames(),
                                        positiveClass=self.yClassToScore)

            output += trl.getFalsePosNegReport( "Validation set", falsePos,
                                            falseNeg, num=nFalsePosNegReport)

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
            (trl.getFormattedTime(), time.time() - self.startTime)
    # ---------------------------

    def writeIndexFile(self, tuningIndexFile, compareBeta):
        '''
        Handle writing a one-line summary of this run to an index file
        '''
        y_true = self.valSet.getYvalues()
        y_predicted = self.valSet.getPredictions()

        if sys.argv: tuningFile = sys.argv[0]
        else: tuningFile = ''

        with open(tuningIndexFile, 'a') as fp:
            fp.write("%s\tF%dPRNPV\t%6.4f\t%6.4f\t%6.4f\t%6.4f\t%s\n" % \
            (self.startTimeStr,
            compareBeta,
            fbeta_score(    y_true, y_predicted, beta=compareBeta,
                                                pos_label=self.yClassToScore), 
            precision_score(y_true, y_predicted, pos_label=self.yClassToScore),
            recall_score(   y_true, y_predicted, pos_label=self.yClassToScore),
            # negative predictive value:
            precision_score(y_true, y_predicted,pos_label=1-self.yClassToScore),
            tuningFile,
            ) )
    # ---------------------------

    def writePredictions(self,):
        self.writePredictionsFile(self.outputFilePrefix + "_train_pred.txt",
                                    self.trainSet,
                                    self.valSetEstimator,)

        self.writePredictionsFile(self.outputFilePrefix + "_val_pred.txt",
                                    self.valSet,
                                    self.valSetEstimator,)

        if self.testSet:
            self.writePredictionsFile(self.outputFilePrefix + "_test_pred.txt",
                                    self.testSet,
                                    self.testSetEstimator,)
    # ---------------------------

    def writePredictionsFile(self, outputFile, documentSet, estimator):

        if type(outputFile) == type(''): fp = open(outputFile, 'w')
        else: fp = outputFile

        formatter = PredictionFormatter(documentSet, estimator,
                classNames=self.yClassNames, positiveClass=self.yClassToScore,
                doConfidences=self.doConfidences)

        fp.write(formatter.getHeaderText())
        for line in formatter.getNextPredictionText():
            fp.write(line)
    # ---------------------------

    def writeModel(self,):
        fp = open(self.modelFile, 'wb')
        pickle.dump(self.testSetEstimator, fp)
        fp.close()
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
        output += formatter('Training Set', self.trainSet.getYvalues())
        output += formatter('Validation Set', self.valSet.getYvalues())
        if self.testSet:
            output += formatter('Test Set', self.testSet.getYvalues())
        output += "ValidationSplit: %4.2f\n" % self.validationSplit
        return output

# ---------------------------
# end class TextPipelineTuningHelper
# ---------------------------

class DocumentSet (object):
    """
    IS:     a set of documents along with their y_values and sampleNames
    HAS:    parallel lists: documents, y_values, sampleNames along with
              information about where the docs were loaded from
    DOES:   Load from a file OR from directory structure using sklearn
                load_files(),
            If loading from a file, a document may have additional
                extraInfoFields beyond ID, classification (y_values),
                and document text.
            Split into two doc sets, 
            Convert y_values to and from lists and np.array
    Pondered defining a Document class and having DocumentSet be a container of
        Document objects.
        Decided instead to have DocumentSet maintain parallel lists (doc
            strings, sampleNames, y_values,etc.) since these lists are what
            are required by the sklearn methods and many of the report
            methods defined below.

        DocumentSet DOES use ClassifiedSampleSet defined in sampleDataLib.py
        that is a collection of Sample objects.

        So you can think of DocumentSet as the bridge between individual samples
        and the parallel lists needed by sklearn
    """
    def __init__(self,
                                # if specified, these must be parallel lists:
                docs=[],	# list of document (strings)
                y=[],		# list of ints 0,1 or np.array w/ 0,1
                sampleNames=[],	# list of names, (strings)
                extraInfoFieldNames=[],	# names of extra info fields for a doc
                extraInfo=None,		# list of [ doc info fields (strings) ]
                                        # or None if no extra fields
        ):
        """
        Assumes: y is as above.
        """
        if type(y) == type([]): self.y = np.array(y)	# store as np.array
        else: self.y = y

        self.docs        = docs
        self.sampleNames = sampleNames
        self.extraInfoFieldNames = extraInfoFieldNames
        self.extraInfo   = extraInfo
        self.predictions = None
        self.pathName    = None		# where loaded from
        self.splitSize   = None		# if we do split, fraction to new set
        self.randomSeed  = None		# if we do split, use for random subset

    # ---------------------------
    def getNumDocs(self):	return len(self.docs)
    def getDocs(self):		return self.docs
    def getSampleNames(self):	return self.sampleNames
    def getYvalues(self):	return self.y		# as np.array
    def getYvaluesAsList(self):	return self.y.tolist()	# as list

    def getExtraInfo(self):	return self.extraInfo
    def getExtraInfoFieldNames(self): return self.extraInfoFieldNames

    def getPredictions(self):	return self.predictions
    def setPredictions(self, preds):  # list of ints 0,1. Parallel to self.docs
        self.predictions = preds

    def getPathName(self):	return self.pathName
    def getSplitSize(self):	return self.splitSize
    def getRandomSeed(self):	return self.randomSeed
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

        # JIM: this doesn't handle self.predictions. Kind assuming any split
        #   would take place before the docSet is predicted.
        if self.extraInfo != None:		# include extraInfo in the split
            self.sampleNames, sampleNames_new,	\
            self.docs,        docs_new,		\
            self.y,           y_new, 		\
            self.extraInfo,   extraInfo_new = train_test_split( \
                                                self.sampleNames,
                                                self.docs,
                                                self.y,
                                                self.extraInfo,
                                                test_size=self.splitSize,
                                                random_state=self.randomSeed)
        else:
            self.sampleNames, sampleNames_new,	\
            self.docs,        docs_new,		\
            self.y,           y_new 		= train_test_split( \
                                                self.sampleNames,
                                                self.docs,
                                                self.y,
                                                test_size=self.splitSize,
                                                random_state=self.randomSeed)
            extraInfo_new = None

        new = DocumentSet(docs=docs_new, y=y_new, sampleNames=sampleNames_new,
                                extraInfoFieldNames=self.extraInfoFieldNames,
                                extraInfo=extraInfo_new)
        return new
    # ---------------------------

    def load(self, path):
        """
        Load self from the specified directory path.
        Path can be a
            Directory name, in which case we assume there are subdirectories
                to be loaded by sklearn load_files()
            Filename, in which case we assume it is a file of records that
                can be loaded by sampleDataLib.ClassifiedSampleSet().
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
        """
        Load from a directory using sklearn load_files()
        NOTE I have not loaded documents this way in a long time, so this has
            not been tested
        """
        dataSet          = load_files( path )
        self.docs        = dataSet.data
        self.y           = dataSet.target
        self.sampleNames = self.fileNamesToSampleNames(dataSet.filenames)
        self.extraInfoFieldNames   = []
        self.extraInfo   = None

        return self
    # ---------------------------

    def fileNamesToSampleNames(self, filenames):
        '''
        Convert list of filenames into sampleNames:
            file basenames without any file extension.
        '''
        return [ os.path.splitext(os.path.basename(fn))[0] for fn in filenames ]
    # ---------------------------

    def loadFromFile(self, path):
        self.docs        = []
        self.y           = []
        self.sampleNames = []

        srSet = sampleDataLib.ClassifiedSampleSet().read(path)

        self.extraInfoFieldNames = srSet.getExtraInfoFieldNames()

        if self.extraInfoFieldNames: self.extraInfo = []
        else: self.extraInfo = None

        for sr in srSet.sampleIterator():
            self.docs.append(sr.getDocument())
            self.y.append(sr.getKnownYvalue())
            self.sampleNames.append(sr.getSampleName())
            if self.extraInfo != None:  self.extraInfo.append(sr.getExtraInfo())

        self.y = np.array(self.y)

        return self
# end class DocumentSet ---------------------------

class PredictionFormatter (object):
    """
    IS  an object that knows how to format predictions from a DocumentSet to
        write to a prediction output file
    HAS  docSet, pipeline, fields to include in prediction output file
    DOES iterator to return formated predictions
    """
    outputFields = [		# field/column names & fmt in prediction output
                {'fn' : 'ID'         , 'format':'%s'},
                {'fn' : 'Pred Class' , 'format':'%s'},
                {'fn' : 'Confidence' , 'format':'%5.3f'},
                {'fn' : 'Abs Value'  , 'format':'%5.3f'},
                {'fn' : 'True Class' , 'format':'%s'},
                {'fn' : 'FP/FN'      , 'format':'%s'},
            ]
    # ---------------------------

    def __init__(self,
                docSet,			# DocumentSet that has predictions
                pipeline,		# the pipeline that made the predictions
                classNames=['no','yes'], # class labels
                positiveClass=1,	# index in classNames considered the
                                        #   positive class
                doConfidences=True,	# Include confidence vals in predictions
                ):
        """
        Assumes docSet already has predictions from docSet.setPredictions().

        Build parallel lists, one item for each doc in docSet:
        self.trueNames[]	- "keep"/"discard"
        self.predNames[]	- same
        self.predTypes[]	- "TP", "FP", ...
        self.confidences[]	- 0's if pipeline doesn't support confidences
        self.absConf[]		- 0's if pipeline doesn't support confidences
        sampleNames[]
        extraInfo[]		- each item is a list of extra values
                                -  (strings) to output for the doc
        """
        self.fieldSep = '|'
        self.docSet = docSet
        self.positiveClass = positiveClass
        self.doConfidences = doConfidences

        y_true      = docSet.getYvalues()
        y_predicted = docSet.getPredictions()

        self.setConfidenceValues(docSet, pipeline)

        # map predictions 0/1 to text names
        self.trueNames = [ classNames[y] for y in y_true ]
        self.predNames = [ classNames[y] for y in y_predicted ]

        # set predicton type "FP", "FN", "TP", "TN
        self.predTypes = [skHelper.predictionType(t, p,
                                    positiveClass=positiveClass) \
                                        for t,p in zip(y_true,y_predicted)]

        # get fieldnames and formats to output
        self.fieldNames = [ f['fn']     for f in self.outputFields ]
        self.formats    = [ f['format'] for f in self.outputFields ]

        # add any extraInfoFields the documents might have
        self.fieldNames += docSet.getExtraInfoFieldNames()
        self.formats += [ '%s' for i in \
                                    range(len(docSet.getExtraInfoFieldNames()))]
    # ---------------------------

    def setConfidenceValues(self, docSet, pipeline):
        """
        Compute "confidence" values (and abs value)
        """
        if not self.doConfidences:
            self.confidences = [ 0.0 for x in range(docSet.getNumDocs()) ]
        else:
            self.confidences = skHelper.getConfidenceValues(pipeline,
                            docSet.getDocs(), positiveClass=self.positiveClass)
            if not self.confidences:
                self.confidences = [ 0.0 for x in range(docSet.getNumDocs()) ]

        self.absConf     = list(map(abs, self.confidences))
    # ---------------------------

    def getHeaderText(self):  return self.fieldSep.join(self.fieldNames) + '\n'

    def getNextPredictionText(self,):
        """
        iterator over the predictions
        """
        formatString = self.fieldSep.join(self.formats) + '\n'
        sampleNames  = self.docSet.getSampleNames()
        extraInfo    = self.docSet.getExtraInfo()

        for i in range(len(sampleNames)):
            pred = [sampleNames[i],
                    self.predNames[i],
                    self.confidences[i],
                    self.absConf[i],
                    self.trueNames[i],
                    self.predTypes[i],
                    ]
            if extraInfo:
                pred += extraInfo[i]

            yield formatString % tuple(pred)
# end class PredictionFormatter ---------------------------

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
    if True:	# DocumentSet  and Prediction Formatter testing
        print("docs 1")
        d = DocumentSet()

        d.load('Data/smallset/valSetFig.txt')
        d.setPredictions([0 for i in range(100)])
        print(len(d.getDocs()))
        print(d.getSampleNames()[:5])
        print(d.getYvalues()[:5])
        print(d.getYvaluesAsList()[:5])

        formatter = PredictionFormatter(d, None)
        sys.stdout.write(formatter.getHeaderText())
        for line in formatter.getNextPredictionText():
            sys.stdout.write(line)

    if True:	# split testing
        d2 = d.split(splitSize=0.2)
        print("docs 1 after split")
        print(len(d.getDocs()))
        print(d.getSampleNames()[:5])

        d.setPredictions([0 for i in range(d.getNumDocs())])
        formatter = PredictionFormatter(d, None)
        sys.stdout.write(formatter.getHeaderText())
        for line in formatter.getNextPredictionText():
            sys.stdout.write(line)

        print("docs 2")
        print(len(d2.getDocs()))
        print(d2.getSampleNames()[:5])

        d2.setPredictions([0 for i in range(d2.getNumDocs())])
        formatter = PredictionFormatter(d2, None)
        sys.stdout.write(formatter.getHeaderText())
        for line in formatter.getNextPredictionText():
            sys.stdout.write(line)

    if True:	# load from directory testing
        print("docs 3")
        d3 = DocumentSet()
        print(d3.load('Data/smallset').getSampleNames()[:5])
