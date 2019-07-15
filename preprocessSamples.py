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
#-----------------------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Apply preprocessor steps to files of samples. Write to stdout')

    parser.add_argument('inputFiles', nargs=argparse.REMAINDER,
    	help='files of samples, "-" for stdin')

    parser.add_argument('--omitrejects', dest='omitRejects', action='store_true',
	required=False, help="don't write reject samples, default is write")

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
	required=False, help="skip helpful messages to stderr")

    parser.add_argument('-p', '--preprocessor', metavar='PREPROCESSOR',
	dest='preprocessors', action='append', required=False, default=[],
    	help='preprocessor method name in sampleDataLib, multiples are applied in order. Default is none.' )

    args = parser.parse_args()

    return args
#----------------------

args = parseCmdLine()

#----------------------
# Main prog
#----------------------
def main():

    # extend path up multiple parent dirs, hoping we can import sampleDataLib
    sys.path = ['/'.join(dots) for dots in [['..']*i for i in range(1,4)]] + \
		    sys.path
    import sampleDataLib

    verbose("Preprocessing steps: %s\n" % ' '.join(args.preprocessors)) 
    counts = { 'samples':0, 'skipped':0}
    firstFile = True
    startTime = time.time()

    for fn in args.inputFiles:

	verbose("Reading %s\n" % fn)

	if fn == '-': fn = sys.stdin

	# JIM: Classified vs Unclassified ? will need to deal with this
	sampleSet = sampleDataLib.ClassifiedSampleSet()
	sampleSet.read(fn)

	# save prev sample ID for printing if we get an exception.
	# Gives us a fighting chance of finding the offending record
	prevSampleName = 'very first sample'

	for rcdnum, sample in enumerate(sampleSet.sampleIterator()):
	    counts['samples'] += 1
	    try:
		for pp in args.preprocessors:
		    sample = getattr(sample, pp)()  # preprocessor method on sample

		if args.omitRejects and sample.isReject():
		    verbose( "%s, %s skipped\n" % \
				(sample.getRejectReason(), sample.getSampleName()) )
		    counts['skipped'] += 1
		prevSampleName = sample.getSampleName()
	    except:
		sys.stderr.write( "\nException in %s record %s prevID %s\n\n" % \
				    (fn, rcdnum, prevSampleName)  )
		raise

	sampleSet.write( sys.stdout, writeHeader=firstFile,
						    omitRejects=args.omitRejects )
	firstFile = False

    verbose("Samples processed: %d \t Samples skipped: %d\n" % \
					(counts['samples'], counts['skipped']) )
    verbose( "Total time: %8.3f seconds\n\n" % (time.time()-startTime))
# ---------------------

def verbose(text):
    if args.verbose:
	sys.stderr.write(text)
	sys.stderr.flush()
# ---------------------
if __name__ == "__main__":
    main()
