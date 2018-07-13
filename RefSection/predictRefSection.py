#!/usr/bin/env python2.7

#
#  Purpose: predict reference sections in extracted text files
#		output "|" delimited text file summarizing results
#
#  Outputs:
#    write to stdout, for each extracted text file:
#	pubmed ID, doc length,
#	last matched "reference section" indicating term,
#	# chars after that, % of doc after that,
#	count of "mice" before predicted ref section
#	count of "mice" after predicted ref section
#	journal name
#
###########################################################################
import sys
import os
import string
import time
import re
import argparse
import refSectionLib
#from ConfigParser import ConfigParser
#import db

#-----------------------------------

def getArgs():
    parser = argparse.ArgumentParser( \
	description= \
	'''
	Predict Reference Sections from (extracted) text files.
	To stdout, write one line per file describing the predicted ref section.
	''')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
        required=False, help="messages to stderr")

    parser.add_argument('dirNames', nargs=argparse.REMAINDER,
	help= \
	'''
	directory names holding text files to predict.
	Typically these files would be named by PubMed ID.
	''')

    args =  parser.parse_args()

    return args
###################################

FD = '|'	# output field delimiter

outputCols = [ "ID",
		"Doc length",
		"Last term",
		"Num chars after",
		"Percent after",
		"Num mice before",
		"Num mice after",
		"Journal", 
	    ]

def analyzeFiles ():

    args = getArgs()

    refRm = refSectionLib.RefSectionRemover()

    if False:	# testing
	for s in ['Acknowledgements', 'Lit Cited', 'stuff 123?']:
	    print almostCaseInsensitiveRegex(s)
	exit(0)
    if False:	# testing
	print buildRegex(refKeyWords)
	exit(0)

    miceRegex = re.compile( r'\bmice\b', flags=re.IGNORECASE)

    numRefMiceOnly = 0		# num of papers with "mice" only in refs

    startTime = time.time()

    sys.stdout.write( FD.join(outputCols) + '\n')
    numProcessed = 0
    for dn in args.dirNames:

	if dn.endswith(os.sep): 		# path ends in '/'
	    journal = dn.split(os.sep)[-2]	# last part of the dir path
	else:
	    journal = dn.split(os.sep)[-1]	# last part of the dir path

	for fn in os.listdir(dn):

	    numProcessed += 1
	    if args.verbose and numProcessed % 1000 == 0: # progress indicator
		sys.stderr.write("..%d" % numProcessed)

	    id = os.path.splitext(fn)[0]	# filename w/o any extension
	    pn = os.sep.join([dn, fn])		# pathname to open

	    fp = open(pn, 'r')
	    text = fp.read()
	    textLength = len(text)

	    (refKeyTerm, refStart) = refRm.predictRefSection(text)
	    refsLength = textLength - refStart

	    numMiceBefore = len( miceRegex.findall(text, 0, refStart) )
	    numMiceAfter  = len( miceRegex.findall(text, refStart) )

	    if numMiceBefore == 0 and numMiceAfter > 0:
		numRefMiceOnly += 1

	    outputLineParts =     [ "%s" % id]
	    outputLineParts.append( "%d" % textLength )
	    outputLineParts.append( refKeyTerm )
	    outputLineParts.append( "%d" % refsLength )
	    outputLineParts.append( "%4.1f" % (100.0*float(refsLength)/textLength) )
	    outputLineParts.append( "%d" % numMiceBefore )
	    outputLineParts.append( "%d" % numMiceAfter )
	    outputLineParts.append( journal )

	    sys.stdout.write( FD.join(outputLineParts) + '\n')
	    fp.close()

    sys.stderr.write("\n%d articles have 'mice' only in Refs section\n"  \
				    % numRefMiceOnly)

    sys.stderr.write("%d files analyzed. Elapsed time: %8.2f seconds\n\n" \
				    % (numProcessed, time.time() - startTime) )
# ----------------------------------
#
#  MAIN
#
if __name__ == "__main__": analyzeFiles()
