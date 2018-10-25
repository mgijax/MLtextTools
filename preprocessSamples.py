#!/usr/bin/env python2.7 
#
# preprocessSamples.py
# Takes one or more files of samples and applies various transformations
#   to the samples, and writes transformed samples to stdout
#
# SampleRecord Class in sampleDataLib.py is responsible for the sample
#   handling and transformations and is intended to encapsulate all the 
#   sample details.
#
# Assumes all input files have the same column structure. see SampleRecord.
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
    	help='files of samples')

    parser.add_argument('-p', '--preprocessor', metavar='PREPROCESSOR',
	dest='preprocessors', action='append', required=False, default=[],
    	help='preprocessor method name in sampleDataLib.SampleRecord, multiples are applied in order. Default is none.' )

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
	required=False, help="skip helpful messages to stderr")

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
    recordSep = sampleDataLib.RECORDSEP
    counts = { 'samples':0, 'skipped':0}
    headerLine = None		# None until we read 1st header line in 1st file
    startTime = time.time()

    for fn in args.inputFiles:

	verbose("Reading %s\n" % fn)
	lines = open(fn,'r').read().split(recordSep)
	del lines[-1]			# empty line at end after split

	if not headerLine:		# print header only @ start of 1st file
	    headerLine = lines[0]
	    sys.stdout.write(headerLine + recordSep)

	del lines[0]			# delete header line 

	# save prev sample ID for printing if we get an exception.
	# Give us a fighting chance of finding the offending record!
	prevSampleName = 'very first sample'

	for rcdnum, line in enumerate(lines):
	    try:
		counts['samples'] += 1
		if counts['samples'] % 500 == 0:	# print every 500
		    verbose("%d.." % counts['samples'] )

		sr = sampleDataLib.SampleRecord(line)
		for pp in args.preprocessors:
		    sr = getattr(sr,pp)()	# call preprocessor method on sr

		if sr.isReject():
		    verbose("%s, %s skipped\n" % \
				(sr.getRejectReason(), sr.getSampleName()) )
		    counts['skipped'] += 1
		else: sys.stdout.write(sr.getSampleAsText())
		prevSampleName = sr.getSampleName()
	    except:
		sys.stderr.write( \
		"\nException in %s record %s prevID %s\n\n" % \
				    (fn, rcdnum, prevSampleName)
		)
		raise

    verbose("Samples processed: %d \t Samples skipped: %d\n" % \
					(counts['samples'], counts['skipped']) )
    verbose( "Total time: %8.3f seconds\n\n" % (time.time()-startTime))
# ---------------------

def verbose(text):
    if args.verbose: sys.stderr.write(text)

# ---------------------
if __name__ == "__main__":
    main()
