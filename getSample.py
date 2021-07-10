#!/usr/bin/env python3

# Read a sample file from stdin and extract the sample record
#  for specified pubmed IDs. Write to stdout.
#
import sys
import argparse
import utilsLib

DEFAULT_SAMPLE_TYPE  = "BaseSample"
DEFAULT_SAMPLEDATALIB  = "sampleDataLib"

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='read sample rcds from stdin & write selected rcds to stdout')

    parser.add_argument('sampleIDs', nargs='+',
        help='IDs for samples to select')

    parser.add_argument('--sampledatalib', dest='sampleDataLib',
        default=DEFAULT_SAMPLEDATALIB,
        help="Module to import that defines python sample class. " +
                                        "Default: %s" % DEFAULT_SAMPLEDATALIB)
    parser.add_argument('--sampletype', dest='sampleObjTypeName',
        default=DEFAULT_SAMPLE_TYPE,
        help="Sample class name to use if not specified in sample file. " +
                                            "Default: %s" % DEFAULT_SAMPLE_TYPE)

    parser.add_argument('--justtext', dest='justText', action='store_true',
        help="output just the text of the sample, not the full sample record")

    parser.add_argument('--oneline', dest='oneLine', action='store_true',
        help="smoosh sample records into one line each.")

    parser.add_argument('--header', dest='header', action='store_true',
        help="include a header line in the output.")

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
        required=False, help="include helpful messages to stderr")

    args = parser.parse_args()

    args.sampleFile = sys.stdin		# this could become an arg someday

    return args
#---------------------------

args = parseCmdLine()
sampleDataLib = utilsLib.importPyFile(args.sampleDataLib)

#---------------------------
def main():
    ### ideally, the sampleObjType will be determined from #meta in the
    ###    SampleSet file.
    ### args.sampleObjTypeName will only be used if there is no #meta
    if not hasattr(sampleDataLib, args.sampleObjTypeName):
        sys.stderr.write("invalid sample class name '%s'\n" \
                                                    % args.sampleObjTypeName)
        exit(5)

    sampleObjType = getattr(sampleDataLib, args.sampleObjTypeName)

    sampleSet = sampleDataLib.SampleSet(sampleObjType=sampleObjType)
    sampleSet.read(args.sampleFile)
    verbose("Sample type: '%s'\n" % sampleSet.getSampleObjType().__name__)

    recordEnd = sampleSet.getRecordEnd()
    wroteHeader = False

    for rcdnum, sample in enumerate(sampleSet.sampleIterator()):

        if sample.getID() in args.sampleIDs: 
            verbose("ID '%s' found at record number %d\n" % \
                                                    (sample.getID(), rcdnum))
            if args.justText:
                text = sample.getDocument()
            else:
                if args.header and not wroteHeader:
                    sys.stdout.write(sampleSet.getHeaderLine() + recordEnd)
                    wroteHeader = True
                text = sample.getSampleAsText()

            if args.oneLine:
                text = text.replace('\n', ' ')

            sys.stdout.write(text + recordEnd + '\n')
#---------------------------
def verbose(text):
    if args.verbose:
        sys.stderr.write(text)
        sys.stderr.flush()

if __name__ == "__main__":
    main()
