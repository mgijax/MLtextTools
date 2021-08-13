#!/usr/bin/env python3 
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
import sklearnHelperLib as skHelper
import utilsLib

DEFAULT_SAMPLEDATALIB  = "sampleDataLib"
DEFAULT_SAMPLE_TYPE  = "BaseSample"
#-----------------------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Apply preprocessor steps to files of samples. Write to stdout')

    parser.add_argument('inputFiles', nargs='+',
    	help='files of samples, "-" for stdin')

    parser.add_argument('-p', '--preprocessor', metavar='PREPROCESSOR',
        dest='preprocessors', action='append', required=False, default=[],
        help='preprocessor, multiples are applied in order. Default is none.' )

    parser.add_argument('--omitrejects', dest='omitRejects',
        action='store_true', 
        help="don't write reject samples, default is write")

    parser.add_argument('--sampledatalib', dest='sampleDataLib',
        default=DEFAULT_SAMPLEDATALIB,
        help="Module to import that defines python sample class. " + 
                                        "Default: %s" % DEFAULT_SAMPLEDATALIB)
                                        
    parser.add_argument('--sampletype', dest='sampleObjTypeName',
        default=DEFAULT_SAMPLE_TYPE,
        help="Sample class name to use if not specified in sample file. " +
                                        "Default: %s" % DEFAULT_SAMPLE_TYPE)

    parser.add_argument('--report', dest='preprocessorReport',
        default=None,
        help="Write a preprocessor report to the specified file. " +
                                    "Default: no report")

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
        default=True, help="include helpful messages to stderr, default")

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
        required=False, help="skip helpful messages to stderr")

    return parser.parse_args()
#----------------------

args = parseCmdLine()
sampleDataLib = utilsLib.importPyFile(args.sampleDataLib)

#----------------------
def main():
#----------------------
    # get default sampleObjType
    if not hasattr(sampleDataLib, args.sampleObjTypeName):
        sys.stderr.write("invalid sample class name '%s'\n" \
                                                    % args.sampleObjTypeName)
        exit(5)
    sampleObjType = getattr(sampleDataLib, args.sampleObjTypeName)

    verbose("Preprocessing steps: %s\n" % ' '.join(args.preprocessors)) 
    totNumSamples = 0
    totNumRejects = 0
    firstFile = True
    startTime = time.time()

    for fn in args.inputFiles:
        verbose("Preprocessing '%s'\n" % fn)
        if fn == '-': fn = sys.stdin

        sampleSet = sampleDataLib.SampleSet(sampleObjType).read(fn)

        if firstFile:
            sampleObjType     = sampleSet.getSampleObjType()
            verbose("Sample type: %s\n" % sampleObjType.__name__)
        else:
            if sampleObjType != sampleSet.getSampleObjType():
                sys.stderr.write( \
                    "Input files have inconsistent sample types: %s & %s\n" % \
                    (sampleObjType.__name__,
                    sampleSet.getSampleObjType().__name__) )
                exit(5)

        rejected = sampleSet.preprocess(args.preprocessors)

        sampleSet.write(sys.stdout, writeHeader=firstFile,
                            writeMeta=firstFile, omitRejects=args.omitRejects)
        firstFile = False
        numSamples = sampleSet.getNumSamples()
        numRejects = len(rejected)
        totNumSamples += numSamples
        totNumRejects += numRejects
        verbose('...done. %d samples, %d marked as reject\n' % \
                                                    (numSamples, numRejects))

    if args.omitRejects: numWritten = totNumSamples - totNumRejects
    else: numWritten = totNumSamples

    if args.preprocessorReport \
        and hasattr(sampleObjType, 'getPreprocessorReport'):
        with open(args.preprocessorReport, 'w') as fp:
            fp.write(sampleObjType.getPreprocessorReport())
        verbose("Wrote preprocessor report to '%s'\n" % args.preprocessorReport)

    verbose("Samples read: %d \t Samples written: %d\n" % \
                                                (totNumSamples, numWritten))
    verbose( "Total time: %8.3f seconds\n\n" % (time.time()-startTime))
# ---------------------
def verbose(text):
    if args.verbose:
        sys.stderr.write(text)
        sys.stderr.flush()
# ---------------------
if __name__ == "__main__": main()
