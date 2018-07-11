#!/usr/bin/env python2.7

# NOTE this is a script written early in the reference section analysis.
#  it might not work right now...
#
#  Purpose:
#   output counts of occurrances of different reference section title terms
#	from text files containing extracted text
#
#  Outputs:     write "|" delimited text file summarizing counts to stdout
#
###########################################################################
import sys
import os
import string
import time
import re
import argparse
from ConfigParser import ConfigParser

def getArgs():
    parser = argparse.ArgumentParser( \
                    description='Analyze extracted text Reference Sections')

#    parser.add_argument('-s', '--server', dest='server', action='store',
#        required=False, default='dev',
#        help='db server: adhoc, prod, or dev (default)')
#
#    parser.add_argument('-o', '--output', dest='outputDir', action='store',
#        required=False, default='.',
#        help="output directory. Default: '.'")
#
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
        required=False, help="messages to stderr")

    parser.add_argument('fileNames', nargs=argparse.REMAINDER,
		    help="extracted text files to analyze")

    args =  parser.parse_args()

    return args
###################################

FD = '|'	# output field delimiter

refKeyWords = [ "References",
		"Literature Cited",
		"Acknowledgements",
		"Conflicts of Interest",
		]
outputCols = [ "ID",
		"References",
		"Literature Cited",
		"Acknowledgements",
		"Conflicts of Interest",
		"Doc Length",
		"Journal", 
	    ]

def analyzeFiles ():

    args = getArgs()

    refRegex = []
    for kw in refKeyWords:
	refRegex.append(re.compile(buildRegex(kw)))

    startTime = time.time()

    sys.stdout.write( FD.join(outputCols) + '\n')
    for i,fn in enumerate(args.fileNames):

	if args.verbose and i % 1000 == 0:
	    sys.stderr.write("..%d" % i)

	# assume filename minus extension is an ID (probably pubmed)
	# if there is a dir name before that, assume it is the journal name
	pathParts = fn.split(os.sep)
	id = os.path.splitext(pathParts[-1])[0]
	if len(pathParts) > 1:
	    journal = pathParts[-2]
	else:
	    journal = 'none'

	fp = open(fn, 'r')
	text = fp.read()

	lineParts = ["%s" % id]
	for kw,regex in zip(refKeyWords,refRegex):
	    matches = re.findall(regex,text)
	    num = len(matches)
	    lineParts.append("%d" % num)

	lineParts.append("%d" % len(text))
	lineParts.append("%s" % journal)

	sys.stdout.write( FD.join(lineParts) + '\n')

	fp.close()
    if args.verbose:
	sys.stderr.write("\n%d files analyzed. Elapsed time: %8.2f seconds\n" \
					     % (i, time.time() - startTime) )

# end analyzeFiles() ----------------------------------

def buildRegex(s):
    # for given string, return regex that matches the 1st char and then
    #   all subsequent chars in either case, and must have a word boundary
    #   on both ends.
    reg = r'\b' + s[0]
    for c in s[1:]:
	reg += '[%s%s]' % (c.upper(), c.lower())
    reg += r'\b'
    return reg

#
#  MAIN
#
if __name__ == "__main__": analyzeFiles()
