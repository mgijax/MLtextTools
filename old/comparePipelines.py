#!/usr/bin/env python2.7 
# Compare some sklearn Pipelines to each other over multiple
#  train_test_splits().
# Computes Fscore, precision, recall over multiple splits and then
#  computes averages across the splits.
# Also tries "voting" across the different Pipelines to see if that fares
#  better
#
import sys
import time
import argparse
import sklearnHelperLib
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score

#-----------------------------------

def parseCmdLine():
    # Defaults
    NUMSPLITS     = '5'
    TESTSIZE      = '20'
    PIPELINE_DEFS = 'goodPipelines.py'
    BETA          = 4
    LABEL         = 'yes'

    parser = argparse.ArgumentParser(description = \
    'Compare pipelines over mult train_test_splits. Write output to stdout.')

    parser.add_argument('-d', '--data', dest='trainingData', action='store', 
        required=True,
        help='Directory where training data files live.' )

    parser.add_argument('-p','--pipelines',dest='pipelineDefs', action='store',
        required=False, default=PIPELINE_DEFS,
        help= \
	'Python file defining pipelines. Expects "pipelines", a list of Pipeline objects or a single Pipeline obj to run. Default: "%s"'\
						    % PIPELINE_DEFS)
    parser.add_argument('-s','--splits',dest='numSplits', action='store',
        required=False, default=NUMSPLITS,
        help='number of train_test_splits to run. Default: %s' % NUMSPLITS)

    parser.add_argument('-t','--testsize',dest='testSize', action='store',
        required=False, default=TESTSIZE,
        help='percent of samples to use for test set. Default: %s' % TESTSIZE)

    parser.add_argument('-b','--beta',dest='beta', action='store',
        required=False, default=BETA,
        help='integer beta to report in comparisons. Default: %d' % BETA)

    parser.add_argument('--label',dest='label', action='store',
        required=False, default=LABEL,
        help='compute comparison metrics for this label/category. Default: %s' % LABEL)

    parser.add_argument('--vote',dest='vote', action='store_true',
        required=False, help='include a vote of all the pipelines.')

    args = parser.parse_args()
    args.beta = int(args.beta)
    args.numSplits = int(args.numSplits)
    args.testSize = float(args.testSize)/100.0
    return args
#----------------------
  
def main():
    # Could/should refactor this whole thing
    args = parseCmdLine()

    myModule = sklearnHelperLib.importPyFile(args.pipelineDefs)
    pipelines = myModule.pipelines

    if type(pipelines) != type([]): 
	pipelines = [ pipelines ]

    if args.vote: nPipelinesAndVotes = len(pipelines)+1	# include votes
    else: nPipelinesAndVotes = len(pipelines)

    # totals across all the split tries for each pipeline + voted predictions
    #  for computing averages
    pipelineTotals = [ {'fscores':0,
			'precisions': 0,
			'f1': 0,
			'recalls': 0, } for i in range(nPipelinesAndVotes) ]

    # formats for output lines, Pipeline line, votes line, avg line
    pf="Pipeline %d:   F1: %5.3f   F%d: %5.3f   Precision: %4.2f   Recall: %4.2f"
    vf="Votes... %d:   F1: %5.3f   F%d: %5.3f   Precision: %4.2f   Recall: %4.2f"
    af="Average. %d:   F1: %5.3f   F%d: %5.3f   Precision: %4.2f   Recall: %4.2f"

    dataSet = load_files( args.trainingData )
    labelIndex = dataSet['target_names'].index(args.label)

    for sp in range(args.numSplits):
	docs_train, docs_test, y_train, y_test = \
		train_test_split( dataSet.data, dataSet.target,
				test_size=args.testSize, random_state=None)

	predictions = []	# predictions[i]= predictions for ith Pipeline
				#  on this split (for voting)
	print "Sample Split %d" % sp
	for i, pl in enumerate(pipelines):	# for each Pipeline

	    pl.fit(docs_train, y_train)
	    y_pred = pl.predict(docs_test)
	    predictions.append(y_pred)

	    precision, recall, fscore, support = \
			    precision_recall_fscore_support( \
						y_test, y_pred, args.beta,
						pos_label=labelIndex,
						average='binary')
	    f1 = f1_score(y_test, y_pred, pos_label=labelIndex,
							average='binary')
	    pipelineTotals[i]['fscores']    += fscore
	    pipelineTotals[i]['f1']	    += f1
	    pipelineTotals[i]['precisions'] += precision
	    pipelineTotals[i]['recalls']    += recall

	    l = pf % (i, f1, args.beta, fscore, precision, recall)
	    print l

	if args.vote:
	    vote_pred = y_vote( predictions )
	    precision, recall, fscore, support = \
				precision_recall_fscore_support( \
						y_test, vote_pred, args.beta,
						pos_label=labelIndex,
						average='binary')
	    f1 = f1_score(y_test, vote_pred, pos_label=labelIndex,
							average='binary')
	    i = len(pipelines)
	    pipelineTotals[i]['fscores']    += fscore
	    pipelineTotals[i]['f1']	    += f1
	    pipelineTotals[i]['precisions'] += precision
	    pipelineTotals[i]['recalls']    += recall

	    l = vf % (i , f1, args.beta, fscore, precision, recall)
	    print l
    # averages across all the Splits
    print
    for i in range(nPipelinesAndVotes):
	avgFscore    = pipelineTotals[i]['fscores']    / args.numSplits
	avgF1        = pipelineTotals[i]['f1']         / args.numSplits
	avgPrecision = pipelineTotals[i]['precisions'] / args.numSplits
	avgRecall    = pipelineTotals[i]['recalls']    / args.numSplits
	l = af % (i, avgF1, args.beta, avgFscore, avgPrecision, avgRecall)
	print l

    # pipeline info
    print "\nTraining data: %s" % args.trainingData
    print time.strftime("%Y/%m/%d-%H-%M-%S")
    for i,p in  enumerate(pipelines):
	print "\nPipeline %d -------------" % i
	for s in p.steps:
	    print s
#-----------------------

def y_vote( theYs,	# [ [y1's], [y2's], ...] parallel arrays of class assn's
	    ):
    '''
    Assuming each yi is an list of 0 and 1's,
    Return a parallel list that is the "vote" across all the yi's
    Ties default to 0 at this point...
    '''
    # there must be a better way to do this... 
    numOnes = theYs[0]	# numOnes[i] will be the number of 1's across y's[i]

    for Y in theYs[1:]:
	for i, val in enumerate(Y):
	    numOnes[i] += val

    votes = [ 0 for i in range(len(numOnes)) ]
    threshold = len(theYs)/2

    for i, c in enumerate(numOnes):
	if numOnes[i] > threshold: votes[i] = 1

    return votes
#-----------------------
if __name__ == "__main__": main()
