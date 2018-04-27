#!/usr/bin/env python2.7 
# Take a python file defining a Pipeline, train it on a given data set,
#    and pickle it to a file so the model can be used to predict new samples.
#
# The python file should define a variable 'pipelines'.
#   this can be a Pipeline object
#   or a list of Pipelines, in which case pipelines[0] is trained
#
# If the Pipeline has a method to get features and their weights/coefficients,
#   also write a top weighted feature report after training.
#
import sys
import argparse
import pickle

import textTuningLib as tl
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
#-----------------------

NUM_TOP_FEATURES=50	# number of highly weighted features to report
GOOD_PIPELINE_FILE = "goodPipelines.py"
PICKLE_FILE   = "goodModel.pkl"
FEATURE_FILE = "goodModel.features"
#-----------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Train a model and pickle it so it can be used to predict.')

    parser.add_argument('-d', '--data', dest='trainingData',
	    required=True, help='Directory where training data files live.')

    parser.add_argument('-p', '--pipeline', dest='pipelineFile',
	    default=GOOD_PIPELINE_FILE,
	    help='Python (input) file defining pipeline to train. Expects "pipelines" a Pipeline object or a list (trains the 0th). Default: "%s"'\
		    % GOOD_PIPELINE_FILE)

    parser.add_argument('-o', '--output', dest='pickleFile',
	    default=PICKLE_FILE,
	    help='output pickle file for trained model. Default: "%s"' \
		    % PICKLE_FILE)

    parser.add_argument('-f', '--features', dest='featureFile',
	    default=FEATURE_FILE,
	    help='output file for top weighted features. Default: "%s"'\
		    % FEATURE_FILE)

    return parser.parse_args()
#-----------------------

def process():
    args = parseCmdLine()

    pyFile = args.pipelineFile
    if pyFile.endswith('.py'): pyFile = pyFile[:-3]

    pipelineModule = __import__(pyFile)
    pipeline = pipelineModule.pipelines

    if type(pipeline) == type([]):
        pipeline = pipeline[0]

    print "Training on data from '%s'" % args.trainingData
    dataSet = load_files( args.trainingData )
    pipeline.fit(dataSet.data, dataSet.target)	# train on all samples

    with open(args.pickleFile, 'wb') as fp:
	pickle.dump(pipeline, fp)
	print "Trained model written to '%s'" % args.pickleFile

    # print features report
    vectorizer = pipeline.named_steps['vectorizer']
    classifier = pipeline.named_steps['classifier']
    orderedFeatures = tl.getOrderedFeatures(vectorizer,classifier)

    if len(orderedFeatures) == 0: 
	print "No feature weights/coefficients are available for this Pipeline"
    else:
	fp = open(args.featureFile, 'w')
	fp.write(tl.getTopFeaturesReport(orderedFeatures, NUM_TOP_FEATURES))
	print "Top weighted features file written to '%s'" % args.featureFile
#-----------------------
if __name__ == "__main__": process()
