#!/usr/bin/env python3 
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
import os.path
import argparse
import pickle

import sklearnHelperLib as skHelper
import utilsLib
import tuningReportsLib as trl
from sklearn.pipeline import Pipeline

NUM_TOP_FEATURES=50	# number of highly weighted features to report
PIPELINE_FILE = "goodPipelines.py"
OUTPUT_PICKLE_FILE   = "goodModel.pkl"
DEFAULT_SAMPLEDATALIB  = "sampleDataLib"
DEFAULT_SAMPLE_TYPE = "ClassifiedSample"
#-----------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Train a model and pickle it so it can be used to predict.')

    parser.add_argument('inputFiles', nargs='+',
        help='files of samples or -, may be sklearn load_files dirs')

    parser.add_argument('-m', '--model', dest='pipelineFile',
        default=PIPELINE_FILE,
        help='Pipeline source (.py).' +
        'Expects "pipeline" a Pipeline object or list (trains the 0th). ' +
        'May be a Pipeline .pkl file. Default: %s' % PIPELINE_FILE)

    parser.add_argument('-p', '--preprocessor', metavar='PREPROCESSOR',
        dest='preprocessors', action='append', required=False, default=None,
        help='preprocessor, multiples are applied in order. Default is none.' )

    parser.add_argument('-o', '--output', dest='outputPklFile',
        default=OUTPUT_PICKLE_FILE,
        help='output pickle file for trained model. Default: "%s"' \
                % OUTPUT_PICKLE_FILE)

    parser.add_argument('-f', '--features', dest='featureFile', default=None,
        help='output file for top weighted features. Default: None')

    parser.add_argument('--numfeatures', dest='numTopFeatures',
        type=int, default=NUM_TOP_FEATURES,
        help='num of top weighted features to output. Default: %d' % \
                                                            NUM_TOP_FEATURES)

    parser.add_argument('--sampledatalib', dest='sampleDataLib',
        default=DEFAULT_SAMPLEDATALIB,
        help="Module to import that defines python sample class. " + 
                                        "Default: %s" % DEFAULT_SAMPLEDATALIB)

    parser.add_argument('--sampletype', dest='sampleObjTypeName',
        default=DEFAULT_SAMPLE_TYPE,
        help="Sample class name to use if not specified in sample file. " +
                                        "Default: %s" % DEFAULT_SAMPLE_TYPE)

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
        default=True, help="include helpful messages to stdout, default")

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
        help="skip helpful messages to stdout")

    return parser.parse_args()
#-----------------------

args = parseCmdLine()
sampleDataLib = utilsLib.importPyFile(args.sampleDataLib)

#-----------------------
def main():
#-----------------------
    # get default sampleObjType
    if not hasattr(sampleDataLib, args.sampleObjTypeName):
        sys.stderr.write("invalid sample class name '%s'\n" \
                                                    % args.sampleObjTypeName)
        exit(5)
    sampleObjType = getattr(sampleDataLib, args.sampleObjTypeName)

    pipeline = getPipeline()
    trainSet = getTrainingSet(sampleObjType)

    if args.preprocessors:
        verbose("Running preprocessors %s\n" % str(args.preprocessors))
        rejects = trainSet.preprocess(args.preprocessors)
        verbose("...done\n")

    verbose("Training...\n")
    pipeline.fit(trainSet.getDocuments(), trainSet.getKnownYvalues())
    verbose("Done\n")

    with open(args.outputPklFile, 'wb') as fp:
        pickle.dump(pipeline, fp)
    verbose("Trained model written to '%s'\n" % \
                                        os.path.abspath(args.outputPklFile))

    if args.featureFile:
        writeFeaturesFile(pipeline, args.featureFile)
#-----------------------

def writeFeaturesFile(pipeline, fileName):
    vectorizer = pipeline.named_steps['vectorizer']
    classifier = pipeline.named_steps['classifier']
    orderedFeatures = skHelper.getOrderedFeatures(vectorizer, classifier)

    if len(orderedFeatures) == 0: 
        verbose("No feature weights/coefs are available for this Pipeline\n")
    else:
        fp = open(fileName, 'w')
        fp.write(trl.getTopFeaturesReport(orderedFeatures, args.numTopFeatures))
        verbose("Top weighted features written to '%s'\n" % \
                                                os.path.abspath(fileName))
#-----------------------

def getTrainingSet(sampleObjType):
    sampleSet = sampleDataLib.ClassifiedSampleSet(sampleObjType=sampleObjType)
    for fn in args.inputFiles:
        verbose("Reading '%s' ...\n" % os.path.abspath(fn))
        if fn == '-': fn = sys.stdin

        sampleSet.read(fn)
        verbose("Sample type '%s'\n" % sampleSet.getSampleObjType().__name__ )

    verbose("...done %d total documents.\n" % sampleSet.getNumSamples())
    return sampleSet
#-----------------------

def getPipeline():
    fileName = args.pipelineFile
    ext = os.path.splitext(fileName)[1]
    if ext == '.py':
        verbose("Importing model source file '%s'\n" % \
                                                os.path.abspath(fileName))
        pipeline = utilsLib.importPyFile(fileName).pipeline

        if type(pipeline) == type([]):
            pipeline = pipeline[0]

    elif ext == '.pkl':
        verbose("Loading model '%s'\n" % os.path.abspath(fileName))
        with open(fileName, 'rb') as fp:
            pipeline = pickle.load(fp)
        verbose("...done\n")
    else:
        sys.stderr.write("Invalid model file extension: '%s'\n" % ext)
        exit(5)

    # JIM IS THIS NECESSARY?
    # Some classifiers write process info to stdout messing up our
    #   output.
    # "classifier__verbose" assumes the model is a Pipeline
    #   with a classifier step. This seems like a pretty safe
    #   assumption, but if it turns out to be false, we'd
    #   need more logic to figure out the name of the "verbose"
    #   argument.
    pipeline.set_params(classifier__verbose=0)

    return pipeline
#-----------------------
def verbose(text):
    if args.verbose:
        sys.stdout.write(text)
        sys.stdout.flush()
#-----------------------
if __name__ == "__main__": main()
