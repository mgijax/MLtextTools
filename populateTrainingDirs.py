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
    	help='parent dir for /classname dirs. Default=%s' % '.')

    parser.add_argument('-e', '--extension', dest='extension', action='store',
	required=False, default=None,
    	help='file extension for sample files. Default=None')

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
        required=False, help="skip helpful messages to stderr")

    args = parser.parse_args()

    return args
#----------------------

args = parseCmdLine()

def main():

    sys.path.extend(['/'.join(dots) for dots in [['..']*i for i in range(1,4)]])
    import sampleDataLib

    if args.extension:
	if args.extension.startswith('.'): fileExtension = args.extension
	else: fileExtension = '.' + args.extension
    else:
	fileExtension = ''

    counts = {}				# keep counts of samples from each class
    for d in sampleDataLib.CLASS_NAMES:
	counts[d] = 0
	dirname =  os.sep.join( [ args.outputDir, d ] )
	if not os.path.exists(dirname):
	    os.makedirs(dirname)

    for fn in args.inputFiles:
	verbose("Reading %s\n" % fn)

        rcds = open(fn,'r').read().split(sampleDataLib.RECORDSEP)
	del rcds[-1]			# empty string at end of split
	del rcds[0]			# header line
	for i,rcd in enumerate(rcds):

	    if i % 100 == 0: verbose('.')

	    sample = sampleDataLib.SampleRecord(rcd)
	    
	    className = sample.getKnownClassName()
	    counts[className] += 1

	    filename = os.sep.join([args.outputDir,className,
					sample.getSampleName() +fileExtension])
	    with open(filename, 'w') as newFile:
		newFile.write(sample.getDocument()) 

    verbose('\n')
    numFiles = 0
    for c in sampleDataLib.CLASS_NAMES:
	verbose("%d files written to %s\n" % (counts[c], c) )
	numFiles += counts[c]
    verbose("%d total files written under %s\n" % (numFiles,args.outputDir))
#----------------------

def verbose(text):
    if args.verbose:
	sys.stderr.write(text)
#----------------------

main()
