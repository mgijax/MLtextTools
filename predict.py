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
# We assume, however, all samples have the same input structure/columns
# We also assume all samples have been preprocessed (e.g., stemming, etc.),
#   no preprocessing steps are performed here.
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
#-----------------------------------

PIPELINE_PICKLE_FILE   = "goodModel.pkl"
DEFAULT_OUTPUT   = "predictions.tsv"
DEFAULT_OUTPUT_LONG  = "predictions_long.tsv"

#-----------------------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
		description='predict samples/records relevance')

    parser.add_argument('inputFiles', nargs=argparse.REMAINDER,
	help='files of samples')

    parser.add_argument('-m', '--model', dest='model', action='store',
	required=False, default=PIPELINE_PICKLE_FILE,
    	help='pickled model file. Default: %s' % PIPELINE_PICKLE_FILE)

    parser.add_argument('-o', '--output', dest='outputFile', action='store',
	required=False, default=DEFAULT_OUTPUT,
    	help='output file name, predictions + confidences. Default: %s'\
							% DEFAULT_OUTPUT)
    parser.add_argument('--long', dest='writeLong',
        action='store_const', required=False, default=False, const=True,
        help='write long output file too. Default: False')

    parser.add_argument('-l', '--longfile', dest='longFile', action='store',
	required=False, default=DEFAULT_OUTPUT_LONG,
    	help='long output file name: predictions + text... . Default: %s' \
						% DEFAULT_OUTPUT_LONG)
    args = parser.parse_args()

    return args
#----------------------
  
def main():
    args = parseCmdLine()

    # extend path up multiple parent dirs, hoping we can import sampleDataLib
    sys.path.extend(['/'.join(dots) for dots in [['..']*i for i in range(1,8)]])
    import sampleDataLib

    #####################
    # Get Trained Model
    print "Loading model %s" % args.model
    with open(args.model, 'rb') as bp:
	model = pickle.load(bp)

    hasConf = hasattr(model, "decision_function")# confidence values?

    #####################
    # Read file of samples to predict

    samples = []			# sample records
    docs = []				# documents (from samples)
    
    for fn in args.inputFiles:
	print "Reading %s" % fn
	#ip = open(args.inputFile, 'r')

        rcds = open(fn,'r').read().split(sampleDataLib.RECORDSEP)
        del rcds[-1]                   # empty rcd at end after split

	for rcd in rcds[1:]:		# skip header rcd
	    sample = sampleDataLib.SampleRecord(rcd)

	    docs.append(sample.getDocument())
	    samples.append(sample)

    print "...done %d documents." % (len(docs))

    #####################
    # predict!!
    print "Predicting...."
    y_predicted = model.predict(docs)
    if hasConf:
	# parallel list to samples[] and docs[]
        confs = model.decision_function(docs).tolist()
    print "...done"

    #####################
    # Write Output Files
    print "Writing prediction file(s)..."

    fp = open(args.outputFile, 'w')

    reporter = sampleDataLib.PredictionReporter(samples[0],hasConf)
    fp.write(reporter.getPredOutputHeader())

    if args.writeLong:
	lfp = open(args.longFile, 'w')
	lfp.write(reporter.getPredLongOutputHeader())

    for i, (s, y) in enumerate(zip(samples, y_predicted, )):
	if hasConf: conf = confs[i]
	else:       conf = None
	fp.write(reporter.getPredOutput(s, y, confidence=conf))
	if args.writeLong:
	    lfp.write(reporter.getPredLongOutput(s, y, confidence=conf))

    print "...done %d lines written to %s" % (len(samples), args.outputFile)
    if args.writeLong:
	print "...done %d lines written to %s" % (len(samples), args.longFile)

    return
# ---------------------------
main()
