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

    parser.add_argument('-r', '--recordsep', dest='recordsep',
	action='store', default='\n',
    	help="sample record separator string. Default \\n" )

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
	required=False, help="skip helpful messages to stderr")

    args = parser.parse_args()

    return args
#----------------------

# Main prog
def main():
    args = parseCmdLine()

    # extend path up multiple parent dirs, hoping we can import sampleDataLib
    sys.path.extend(['/'.join(dots) for dots in [['..']*i for i in range(1,8)]])
    from sampleDataLib import SampleRecord

    if args.verbose:
	sys.stderr.write("Preprocessing steps: %s\n" % \
					' '.join(args.preprocessors)) 
    counts = { 'samples':0, 'skipped':0}
    headerLine = None		# None until we read the 1st header line
    startTime = time.time()

    for fn in args.inputFiles:

	if args.verbose:
	    sys.stderr.write("Reading %s\n" % fn)
	lines = open(fn,'r').read().split(args.recordsep)
	del lines[-1]			# empty line at end after split

	if not headerLine:		# print header only @ start of 1st file
	    headerLine = lines[0]
	    sys.stdout.write(headerLine + args.recordsep)

	del lines[0]			# delete header line 
	for line in lines:
	    counts['samples'] += 1
	    if args.verbose and (counts['samples'] % 100 == 0):# print every 100
		sys.stderr.write("%d.." % counts['samples'] )

	    sr = SampleRecord(line)
	    for pp in args.preprocessors:
		sr = getattr(sr,pp)()	# call preprocessor method on sr

	    if sr.isReject():
		if args.verbose:
		    sys.stderr.write("%s, %s skipped\n" % \
			    (sr.getRejectReason(), sr.getSampleName()) )
		counts['skipped'] += 1
	    else: sys.stdout.write(sr.getSampleAsText() + args.recordsep)

    if args.verbose:
	sys.stderr.write("Samples processed: %d \t Samples skipped: %d\n" % \
			    (counts['samples'], counts['skipped']) )
	sys.stderr.write( "Total time: %8.3f seconds\n\n" % \
						    (time.time()-startTime))
# ---------------------
if __name__ == "__main__":
    main()
