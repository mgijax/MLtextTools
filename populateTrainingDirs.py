#!/usr/bin/env python2.7 
#
# populateTrainingDirs.py
# Take one or more delimited files of samples (one sample record per line)
# and split them into individual text files, one for each sample.
# The samples are assumed to have been assigned to a known class (yes or no).
#
# Files (samples) get shoved into two directories, /data/no and /data/yes,
#       that sklearn.datasets.load_files() can read easily.
#
# SampleRecord class in sampleDataLib.py encapsulates the structure of the
#   sample record line format and other sample details
#
# This script is intended to be independent of specific ML projects.
#
import sys
import string
import os
import argparse
#----------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Converts training data files into sklearn directory structure')

    parser.add_argument('inputFiles', nargs=argparse.REMAINDER,
        help='files of samples')

    parser.add_argument('-o', '--outputDir', dest='outputDir', action='store',
	required=False, default='.',
    	help='parent dir for /no and /yes. Default=%s' % '.')

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
        required=False, help="skip helpful messages to stderr")

    args = parser.parse_args()

    return args
#----------------------

def main():
    args = parseCmdLine()

    sys.path.extend(['/'.join(dots) for dots in [['..']*i for i in range(1,8)]])
    import sampleDataLib

    for yesNo in ['yes', 'no']:
	dirname =  os.sep.join( [ args.outputDir, yesNo ] )
	if not os.path.exists(dirname):
	    os.makedirs(dirname)

    counts = { 'yes':0, 'no':0,}

    for fn in args.inputFiles:
	if args.verbose:
            sys.stderr.write("Reading %s\n" % fn)

        lines = open(fn,'r').read().split(sampleDataLib.RECORDSEP)
	del lines[-1]			# empty string at end of split
	del lines[0]			# header line
	for line in lines:

	    sample = sampleDataLib.SampleRecord(line)
	    
	    yesNo = sample.getKnownClassName()
	    counts[yesNo] += 1

	    filename = os.sep.join([args.outputDir,yesNo,
						sample.getSampleName()])
	    with open(filename, 'w') as newFile:
		newFile.write(sample.getDocument()) 

    numFiles = counts['yes'] + counts['no']
    print "Files written to %s: %d" % (args.outputDir, numFiles)
    print "%d yes, %d no" % (counts['yes'], counts['no'])
#
main()
