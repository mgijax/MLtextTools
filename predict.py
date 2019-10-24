#!/usr/bin/env python2.7 
#
# Script to take a samples and predict them using a trained model
#
# Writes out a prediction file and an optional long prediction file.
#
# See sampleDataLib.py for the details of sample records/files formats
#   and the prediction report formats.
#
# Samples to predict may have a known class (or not)
#
# This script is intended to be independent of specific ML projects.
# The details of data samples are intended to be encapsulated in
#   sampleDataLib.py
#
# Author: Jim Kadin
#
import sys
import string
import pickle
import argparse
import sklearnHelperLib as skHelper
#-----------------------------------

PIPELINE_PICKLE_FILE = "model.pkl"
DEFAULT_SAMPLE_TYPE  = "BaseSample"
DEFAULT_OUTPUT_FIELDSEP  = "|"

#-----------------------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
		description='predict samples. Write predictions to stdout')

    parser.add_argument('inputFiles', nargs=argparse.REMAINDER,
	help='files of samples')

    parser.add_argument('-m', '--model', dest='model', action='store',
	required=False, default=PIPELINE_PICKLE_FILE,
    	help='pickled model file. Default: %s' % PIPELINE_PICKLE_FILE)

    parser.add_argument('--sampletype', dest='sampleObjTypeName',
	default=DEFAULT_SAMPLE_TYPE,
	help="Sample class name to use if not specified in sample file. " +
					"Default: %s" % DEFAULT_SAMPLE_TYPE)

    parser.add_argument('--fieldsep', dest='outputFieldSep',
	default=DEFAULT_OUTPUT_FIELDSEP,
	help="prediction output field separator. Default: '%s'" \
						    % DEFAULT_OUTPUT_FIELDSEP)
    parser.add_argument('-p', '--preprocessor', metavar='PREPROCESSOR',
	dest='preprocessors', action='append', required=False, default=None,
	help='preprocessor, multiples are applied in order. Default is none.' )

    parser.add_argument('--long', dest='longFile', action='store',
	required=False, default=None,
    	help='(not implemented) also write long output to this file. Default: no long output')

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
	    required=False, help="skip helpful messages to stderr")

    args = parser.parse_args()

    return args
#----------------------
  
args = parseCmdLine()

def main():

    # extend path up multiple parent dirs, hoping we can import sampleDataLib
    sys.path.extend(['/'.join(dots) for dots in [['..']*i for i in range(1,8)]])
    import sampleDataLib

    #####################
    if not hasattr(sampleDataLib, args.sampleObjTypeName):
	sys.stderr.write("invalid sample class name '%s'" \
						    % args.sampleObjTypeName)
	exit(5)

    sampleObjType = getattr(sampleDataLib, args.sampleObjTypeName)

    #####################
    verbose("Loading model '%s'\n" % args.model)
    with open(args.model, 'rb') as bp:
	model = pickle.load(bp)

    		# Some classifiers write process info to stdout messing up our
		#   output.
		# "classifier__verbose" assumes the model is a Pipeline
		#   with a classifier step. This seems like a pretty safe
		#   assumption, but if it turns out to be false, we'd
		#   need more logic to figure out the name of the "verbose"
		#   argument.
    model.set_params(classifier__verbose=0)
    verbose("...done\n")

    #####################
    sampleSet = sampleDataLib.SampleSet(sampleObjType)

    for fn in args.inputFiles:
	verbose("Reading '%s' ... " % fn)
	if fn == '-': fn = sys.stdin

	sampleSet.read(fn)
	verbose("Sample type '%s'\n" % sampleSet.getSampleObjType().__name__ )

    verbose("...done %d total documents.\n" % sampleSet.getNumSamples())

    #####################
    if args.preprocessors:
	verbose("Running preprocessors %s\n" % str(args.preprocessors))
	sampleSet = preprocess(sampleSet, args.preprocessors)
	verbose("...done\n")

    #####################
    verbose("Predicting\n")

    y_predicted = model.predict(sampleSet.getDocuments(),)
    classNames = sampleSet.getSampleClassNames()
    predictedClasses = [ classNames[y] for y in y_predicted ]

    verbose("...done\n")

    #####################
    verbose("Getting prediction confidence values\n")
    confidences = skHelper.getConfidenceValues(model, sampleSet.getDocuments(),
			    positiveClass=sampleSet.getY_positive(),)
    if not confidences:
	verbose("no confidence values available for this model\n")
	confidences = [ 0.0 for x in range(sampleSet.getNumSamples()) ]
    verbose("...done\n")

    #####################
    verbose("Writing predictions\n")
    writePredictions(model, sampleSet, predictedClasses, confidences)
    verbose("...done\n")
# ---------------------------

def preprocess(sampleSet,	# samples to preprocess
		preprocessors,	# list of preprocessor function names
		):
    # JIM: maybe move this to sklearnHelperLib?

    # Save prev sample ID for printing if we get an exception.
    # Gives us a fighting chance of finding the offending record
    prevSampleName = 'very first sample'

    for rcdnum, sample in enumerate(sampleSet.sampleIterator()):
	try:
	    for pp in preprocessors:
		sample = getattr(sample, pp)()  # preprocessor method on sample

	    prevSampleName = sample.getSampleName()
	except:
	    sys.stderr.write("\nException in %s record %s prevID %s\n\n" % \
						(fn, rcdnum, prevSampleName))
	    raise
    return sampleSet
# ---------------------------

def writePredictions(model,
		    sampleSet,
		    predictedClasses,
		    confidences,
		    ):
    fp = sys.stdout

    # JIM: should we write more info? title? journal?
    # long output not implemented yet
    header = args.outputFieldSep.join(['ID', 'Prediction', 'Confidence'])
    fp.write(header + '\n')
    
    for className, ID, confidence in \
		zip(predictedClasses, sampleSet.getSampleIDs(), confidences):
	l = args.outputFieldSep.join([str(ID), className, "%6.3f" % confidence])
	fp.write(l + '\n')

    return
# ---------------------------
def verbose(text):
    if args.verbose:
	sys.stderr.write(text)
	sys.stderr.flush()
# ---------------------------
main()
