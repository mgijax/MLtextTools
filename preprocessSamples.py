#!/usr/bin/env python2.7 
#
# Takes one or more input files of samples and applies various preprocessors
#   to the samples, and writes preprocessed samples to stdout
#
# Sample Class in sampleDataLib.py is responsible for the sample
#   handling and preprocessors and is intended to encapsulate all the 
#   sample details.
#
# Assumes all input files have the same column structure. see Sample.
#
# Note if no preprocessor steps are specified, this will intelligently cat
# the sample files, collapsing down to 1 header line at the start of the output
#
# This script is intended to be independent of specific ML projects.
#
import sys
import string
import os
import time
import argparse
import sklearnHelperLib as skHelper

DEFAULT_SAMPLE_TYPE  = "BaseSample"
#-----------------------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Apply preprocessor steps to files of samples. Write to stdout')

    parser.add_argument('inputFiles', nargs=argparse.REMAINDER,
    	help='files of samples, "-" for stdin')

    parser.add_argument('--sampletype', dest='sampleObjTypeName',
	default=DEFAULT_SAMPLE_TYPE,
	help="Sample class name to use if not specified in sample file. " +
					    "Default: %s" % DEFAULT_SAMPLE_TYPE)

    parser.add_argument('--omitrejects', dest='omitRejects',
	action='store_true', required=False,
	help="don't write reject samples, default is write")

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
	required=False, help="skip helpful messages to stderr")

    parser.add_argument('-p', '--preprocessor', metavar='PREPROCESSOR',
	dest='preprocessors', action='append', required=False, default=[],
	help='preprocessor, multiples are applied in order. Default is none.' )

    args = parser.parse_args()

    return args
#----------------------

args = parseCmdLine()

#----------------------
# Main prog
#----------------------
def main():

    # extend path up multiple parent dirs, hoping we can import sampleDataLib
    sys.path = ['/'.join(dots) for dots in [['..']*i for i in range(1,8)]] + \
		    sys.path
    import sampleDataLib

    if not hasattr(sampleDataLib, args.sampleObjTypeName):
        sys.stderr.write("invalid sample class name '%s'" \
                                                    % args.sampleObjTypeName)
        exit(5)

    sampleObjType = getattr(sampleDataLib, args.sampleObjTypeName)

    verbose("Preprocessing steps: %s\n" % ' '.join(args.preprocessors)) 
    totNumSamples = 0
    totNumSkipped = 0
    firstFile = True
    startTime = time.time()

    for fn in args.inputFiles:

	verbose("Preprocessing '%s'\n" % fn)
	if fn == '-': fn = sys.stdin

	sampleSet = sampleDataLib.SampleSet(sampleObjType).read(fn)

        if firstFile:
            sampleObjType     = sampleSet.getSampleObjType()
            verbose("Sample type: %s\n" % sampleObjType.__name__)
        else:
            if sampleObjType != sampleSet.getSampleObjType():
                sys.stderr.write( \
                    "Input files have inconsistent sample types: %s & %s\n" % \
                    (sampleObjType.__name__,
                    sampleSet.getSampleObjType().__name__) )
                exit(5)

	numSamples, numSkipped = preprocess(sampleSet, args.preprocessors)

	sampleSet.write( sys.stdout, writeHeader=firstFile,
			    writeMeta=firstFile, omitRejects=args.omitRejects )
	firstFile = False
	totNumSamples += numSamples
	totNumSkipped += numSkipped
	verbose('...done. %d samples, %d skipped\n' % (numSamples, numSkipped))

    verbose("Samples processed: %d \t Samples skipped: %d\n" % \
						(totNumSamples, totNumSkipped))
    verbose( "Total time: %8.3f seconds\n\n" % (time.time()-startTime))
# ---------------------

def preprocess(sampleSet,
		preprocessors,	# list of preprocessor (method) names
		):
    """
    Run the preprocessors on each sample in the sampleSet.
    Return the number of samples processed and the number skipped
    """
    numSamples = 0
    numSkipped = 0

    # save prev sample ID for printing if we get an exception.
    # Gives us a fighting chance of finding the offending record
    prevSampleName = 'very first sample'

    for rcdnum, sample in enumerate(sampleSet.sampleIterator()):
	numSamples += 1
	try:
	    for pp in args.preprocessors:
		sample = getattr(sample, pp)()  # run preproc method on sample

	    if args.omitRejects and sample.isReject():
		verbose( "%s, %s skipped\n" % \
			(sample.getRejectReason(), sample.getSampleName()) )
		numSkipped += 1
	    prevSampleName = sample.getSampleName()
	except:
	    sys.stderr.write("\nException in record %s prevID %s\n\n" % \
						    (rcdnum, prevSampleName))
	    raise
    return numSamples, numSkipped
# ---------------------

def verbose(text):
    if args.verbose:
	sys.stderr.write(text)
	sys.stderr.flush()
# ---------------------
if __name__ == "__main__":
    main()
