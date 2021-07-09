#!/usr/bin/env python3 
#
# splitSamples.py
# Use this to randomly select a subset of classified samples to use as a test
#   or validation set.
# Takes 1 or more files of classified samples & randomly splits the samples
# into 2 outputs
#   1) the "retained" samples (some specified fraction of the input samples)
#   2) "leftovers" the rest of the samples
#
#   (1) written to a file specified on command line
#   (2) written to a file specified on command line
#   a Summary report is written to stdout
#
# This simply flips a weighted coin for each sample in the input.
#
# Uses a Sample class defined in a sampleDataLib to read/write inputs/outputs.
# Assumes all input files have the same column structure.
#
import sys
import string
import os
import time
import argparse
import random
import utilsLib

DEFAULT_OUTPUT_RETAINED = 'retainedSamples.txt'
DEFAULT_OUTPUT_LEFTOVER = 'leftoverSamples.txt'
DEFAULT_SAMPLEDATALIB  = "sampleDataLib"
DEFAULT_SAMPLE_TYPE = 'BaseSample'
#-----------------------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Randomly split classified sample files into "retained" set' +
                '" & leftovers". Summary stats to stdout.')

    parser.add_argument('inputFiles', nargs='+',
    	help='files of classified samples or "-".')

    parser.add_argument('-f', '--fraction', dest='fraction', action='store',
        required=False, type=float, default=0.2,
        help='fraction of articles to be in the retained set. Float 0..1 .')

    parser.add_argument('--retainedfile', dest='retainedFile', action='store',
        required=False, default=DEFAULT_OUTPUT_RETAINED,
    	help='retained output file. Default: ' + DEFAULT_OUTPUT_RETAINED)

    parser.add_argument('--leftoverfile', dest='leftoverFile', action='store',
        required=False, default=DEFAULT_OUTPUT_LEFTOVER,
    	help='leftover output file. Default: ' + DEFAULT_OUTPUT_LEFTOVER)

    parser.add_argument('--sampledatalib', dest='sampleDataLib',
        default=DEFAULT_SAMPLEDATALIB,
        help="Module to import that defines python sample class. " +
                                        "Default: %s" % DEFAULT_SAMPLEDATALIB)

    parser.add_argument('--sampletype', dest='sampleObjTypeName',
        default=DEFAULT_SAMPLE_TYPE,
        help="Sample class name to use if not specified in sample file. " + 
                                        "Default: %s" % DEFAULT_SAMPLE_TYPE)

    parser.add_argument('--seed', dest='seed', action='store',
        required=False, type=int, default=int(time.time()),
    	help='int seed for random(). Default: a new seed will be generated')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
        default=True, help="include helpful messages to stderr, default")

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
        required=False, help="skip helpful messages to stderr")

    args = parser.parse_args()

    return args
#----------------------

args = parseCmdLine()
sampleDataLib = utilsLib.importPyFile(args.sampleDataLib)

#----------------------
def main():
#----------------------
    startTime = time.time()
    random.seed(args.seed)

    retainedSampleSet = None	# the final sample sets
    leftoverSampleSet = None	#   ...

    # get default sampleObjType
    if not hasattr(sampleDataLib, args.sampleObjTypeName):
        sys.stderr.write("invalid sample class name '%s'\n" \
                                                    % args.sampleObjTypeName)
        exit(5)
    sampleObjType = getattr(sampleDataLib, args.sampleObjTypeName)

    for fn in args.inputFiles:
        verbose("Reading %s\n" % fn)
        if fn == '-': fn = sys.stdin

        inputSampleSet = sampleDataLib.ClassifiedSampleSet(sampleObjType)
        inputSampleSet.read(fn)

        if not retainedSampleSet:	# processing 1st input file
            sampleObjType     = inputSampleSet.getSampleObjType()
            retainedSampleSet = sampleDataLib.ClassifiedSampleSet( \
                                                sampleObjType=sampleObjType)
            leftoverSampleSet = sampleDataLib.ClassifiedSampleSet( \
                                                sampleObjType=sampleObjType)
            verbose("Sample type: %s\n" % sampleObjType.__name__)
        else:
            if sampleObjType != inputSampleSet.getSampleObjType():
                sys.stderr.write( \
                    "Input files have inconsistent sample types: %s & %s\n" % \
                    (sampleObjType.__name__,
                    inputSampleSet.getSampleObjType().__name__) )
                exit(5)

        for sample in inputSampleSet.sampleIterator():
            if random.random() < float(args.fraction):
                retainedSampleSet.addSample(sample)
            else:
                leftoverSampleSet.addSample(sample)

    ### Write output files
    retainedSampleSet.write(args.retainedFile)
    leftoverSampleSet.write(args.leftoverFile)

    verbose('...done. Total time: %8.3f seconds\n' % (time.time()-startTime))

    ### Write summary report
    summary = "\nSummary:  "
    summary += "Retaining random set of samples\n"
    summary += time.ctime() + '\n'
    summary += "Fraction: %5.3f   Seed: %d   Sample type: %s\n" % \
                            (args.fraction, args.seed, sampleObjType.__name__)
    summary += "Input files: %s\n" % str(args.inputFiles)
    summary += '\n'

    summary += "Input Totals:\n"
    totRefs = retainedSampleSet.getNumSamples() + \
                                            leftoverSampleSet.getNumSamples()
    totPos  = retainedSampleSet.getNumPositives() + \
                                            leftoverSampleSet.getNumPositives()
    totNeg  = retainedSampleSet.getNumNegatives() + \
                                            leftoverSampleSet.getNumNegatives()
    summary += formatSummary(totRefs, totPos, totNeg)
    summary += '\n'

    s = retainedSampleSet
    summary += "Retained file '%s': (%5.3f%% of inputs)\n" %  \
                        (args.retainedFile, 100.0 * s.getNumSamples()/totRefs)
    summary += formatSummary(s.getNumSamples(), s.getNumPositives(),
                                                s.getNumNegatives())
    summary += '\n'

    s = leftoverSampleSet
    summary += "Leftover file '%s': (%5.3f%% of inputs)\n" %  \
                        (args.leftoverFile, 100.0 * s.getNumSamples()/totRefs)
    summary += formatSummary(s.getNumSamples(), s.getNumPositives(),
                                                s.getNumNegatives())
    summary += '\n'
    sys.stdout.write(summary)
    return
# end main() ---------------------

def formatSummary(numSamples, numPos, numNeg):
    sum = "%d samples: %d positive (%4.1f%%) %d negative (%4.1f%%)\n" \
                    % (numSamples,
                        numPos, (100.0 * numPos/numSamples),
                        numNeg, (100.0 * numNeg/numSamples), )
    return sum
# ---------------------

def verbose(text):
    if args.verbose:
        sys.stderr.write(text)
        sys.stderr.flush()
# ---------------------

if __name__ == "__main__":
    main()
