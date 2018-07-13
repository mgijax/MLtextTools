#!/usr/bin/env python2.7

#
# Purpose: compare the reference predictions from different predictors.
# Inputs:
#     two tab delimited files of predictions
#
# Outputs:
#    write to stdout, a tab delimited comparison report
#    the "comparison" field is meant to be read
#       "the 2nd prediction is same/earlier/later than the 1st prediction"
#
# Note:  this is pretty brittle code - particularly around expectations on
#  the input files and their columns. If things don't work, it is likely
#  to be that the input columns are not what the code expects.
#
###########################################################################
import sys
import os
import string
import time
import re
import argparse
#import refSectionLib
#from ConfigParser import ConfigParser
#import db

#-----------------------------------

def getArgs():
    parser = argparse.ArgumentParser( \
	description='Compare 2 files of reference section predictions')

#    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
#        required=False, help="messages to stderr")

#    parser.add_argument('-e1', dest='f1HasEval', action='store_true',
#        required=False, help="file 1 has ref section evaluations")
#
#    parser.add_argument('-e2', dest='f2HasEval', action='store_true',
#        required=False, help="file 2 has ref section evaluations")

    parser.add_argument('file1', nargs=1,
	help="first file of predictions")
    parser.add_argument('file2', nargs=1,
	help="second file of predictions")

    args =  parser.parse_args()
#    args.hasEvals  = [args.f1HasEval, args.f2HasEval]
    args.inputFiles = args.file1 + args.file2

    return args
###################################

OUTPUT_FD = '\t'	# output field delimiter
OUTPUT_RD = '\n'	# output record delimiter
INPUT_FD = re.compile('[|]|\t')


# ----------------------------------
class RefPrediction (object):
    '''
    Has the attributes of a reference prediction
    '''
    def __init__(self,
	    attrs,		# list of attributes (strings)
	):
	# attrs has fields from manual prediction evaluation
	# These attrs correspond to columns in the spreadsheets we use to
	#   look at the ref section predictions
	self.hasEvaluation = len(attrs) > 8

	if not self.hasEvaluation:
	    (self.ID,
	    self.docLen,     
	    self.splitTerm,  
	    self.nCharsAfter,
	    self.percentAfter,
	    self.nMiceBefore,
	    self.nMiceAfter,
	    self.journal)     = attrs[:8]
	else:
	    (self.ID,
	    self.docLen,     
	    self.splitTerm,  
	    self.evaluation,	# text eval:  "Good", "Too Late", ...
	    self.reason,	# text reason
	    self.nCharsAfter,
	    self.percentAfter,
	    self.nMiceBefore,
	    self.nMiceAfter,
	    self.journal,
	    self.curator)     = attrs[:11]
	self.docLen       = int(self.docLen)
	self.nCharsAfter  = int(self.nCharsAfter)
	self.percentAfter = float(self.percentAfter)
	self.nMiceBefore  = int(self.nMiceBefore)
	self.nMiceAfter   = int(self.nMiceAfter)

    def compare(self, other,	# other RefPrediction to compare
	):
	"""
	Return indication (string)  of how 'self' compares to 'other'.
	E.g., is self earlier/later than other
	"""
	if self.nCharsAfter == other.nCharsAfter:	# add some slop?
	    compTerm = "same"
	elif self.nCharsAfter > other.nCharsAfter:
	    compTerm = "later"
	else:
	    compTerm = "earlier"
	return compTerm

# end class RefPrediction --------------

def main ():

    args = getArgs()
    outFile = sys.stdout

    predictions = [ {}, {} ]	# 2 sets of predictions, indexed by pubmed ID
    hasEvaluations = [False, False]	# [i] == True if pred set i has evals

    # read in the two files of reference section predictions
    for i, (preds, filename) in enumerate( zip(predictions, args.inputFiles) ):
	for line in open(filename).readlines()[1:]:
	    attrs = INPUT_FD.split(line[:-1],12)
	    prediction = RefPrediction(attrs)
	    preds[prediction.ID] = prediction
	    hasEvaluations[i] = prediction.hasEvaluation

    preds0, preds1 = predictions

    # header line for comparison output
    header = OUTPUT_FD.join( getColumnHeaders(hasEvaluations) )
    outFile.write(header + OUTPUT_RD)

    # output the comparisons
    for ID, p0 in preds0.items():	# go through all the preds0 predictions
	p1 = preds1.get(ID, None)	# ID could be in p1 or not
	if p1 != None:
	    columns = getComparisonColumns(hasEvaluations, p0, p1)
	    line = OUTPUT_FD.join( columns )
	    outFile.write(line + OUTPUT_RD)
#	preds1.pop(ID,None)		# remove IDs from p1 as we go
#
#    for ID, p1 in preds1.items():	# go through remaining p1 predictions
#	columns = getComparisonColumns(hasEvaluations, None, p1)
#	line = OUTPUT_FD.join( columns )
#	outFile.write(line + OUTPUT_RD)

# ----------------------------------
def getComparisonColumns( hasEvaluations, p0, p1 ):
    # Return columns for comparison output
    # hasEvaluations[i] == True if pi has an evaluation field
    # p0 or p1 could be None, meaning the ID is not in both sets of predictions
    #   - but not both

    if p0 != None:
	columns = [ str(p0.ID),
		    str(p0.docLen),
		    p0.splitTerm,
		    str(p0.nCharsAfter),
		    str(p0.percentAfter)
		    ]
	if hasEvaluations[0]:
	    columns += [ p0.evaluation, ]
	    columns += [ p0.reason, ]
	journal = p0.journal		# remember journal name
    else:		# no p0
	columns = [ str(p1.ID), str(p1.docLen), '', '', '', ]
	if hasEvaluations[0]: columns += [ '','' ]

    if p0 == None or p1 == None:	# nothing to compare
	columns += [ '' ]
    else:
	columns += [ str(p0.compare(p1)) ]

    if p1 != None:
	columns += [
	    p1.splitTerm,
	    str(p1.nCharsAfter),
	    str(p1.percentAfter), 
	    ]
	if hasEvaluations[1]:
	    columns += [ p1.evaluation, ]
	    columns += [ p1.reason, ]
	journal = p1.journal		# remember journal name
    else:		# no p1
	columns += [ '', '', '', '' ]
	if hasEvaluations[1]: columns += [ '','' ]

    columns += [ journal ]

    return columns

# ----------------------------------
def getColumnHeaders( hasEvaluations):
    # Return columns headings for comparison output
    columns = [
	"ID",
	"Doc Length",
	"P0 Split term",
	"P0 Num chars after",
	"P0 Percent after", 
	]
    if hasEvaluations[0]: columns += [ "P0 Eval", "P0 Reason" ]

    columns += [ "Comparison" ]

    columns += [
	"P1 Split term",
	"P1 Num chars after",
	"P1 Percent after", 
	]
    if hasEvaluations[1]: columns += [ "P1 Eval", "P1 Reason"]

    columns += [ "Journal" ]
    return columns
# ----------------------------------
#
#  MAIN
#
if __name__ == "__main__": main()
