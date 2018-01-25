#!/usr/bin/env python2.7 
#
# Produce a random subset of lines from a file (typically TSV file)
# Read from stdin, write random subset file and "leftover" file 
#  (the ones not selected at the random subset)
#  of lines.
#
import sys
import string
import random
import argparse
#-----------------------------------

def parseCmdLine():
    # sometime, support different record separators than just '\n'
    parser = argparse.ArgumentParser( \
	    description='Split lines from stdin into a random subset and leftovers. Output lines are in the same order as input.')

    parser.add_argument('-r', '--randomfile', dest='randomFile',
	action='store', type=argparse.FileType('w'), default=sys.stdout,
	help='output file for random lines. Default to stdout')

    parser.add_argument('-l', '--leftoverfile', dest='leftoverFile',
	action='store', type=argparse.FileType('w'), default=None,
	help='output file for leftover lines. Default: no leftovers.')

    parser.add_argument('--noheader', dest='hasHeader',
        action='store_false', required=False, 
        help='input has no header line to keep. Default: preserve header in output files')

    parser.add_argument('-d', '--delim', dest='recordsep', action='store',
	required=False, default='\n',
	help="record delimiter. Default: '\\n'")

    parser.add_argument('-n', '--num', dest='numKeep', action='store',
	required=False, type=int, default=0,
	help='number of random lines to keep')

    parser.add_argument('-f', '--fraction', dest='fractionKeep', action='store',
	required=False, type=float, default=0.0,
    	help='fraction of total lines to keep. Float between 0 and 1.')

    args = parser.parse_args()

    if args.numKeep == 0 and args.fractionKeep == 0.0:
	sys.stderr.write('error: you need to specify -n or -f\n\n')
	parser.print_usage(sys.stderr)
	exit(4)

    return args
#----------------------

# Main prog
def main():
    args = parseCmdLine()

    records = sys.stdin.read().split(args.recordsep)
    del records[-1]			# remove empty record at end of split
    lines = [ x + args.recordsep for x in records] # add back recsep on each

    rfp = args.randomFile
    lfp = args.leftoverFile
    
    if args.hasHeader:
	rfp.write(lines[0])
	if lfp : lfp.write(lines[0])
	del lines[0]

    if args.numKeep != 0:
	num = args.numKeep
    else:
	num = int(len(lines) * args.fractionKeep)

    # list of random line indexes, sorted
    randomLines = sorted(random.sample(range(len(lines)), num))

    numRlines = 0
    numLlines = 0

    for i in range(len(lines)):
	if len(randomLines) != 0 and i == randomLines[0]:
	    rfp.write(lines[i])
	    del randomLines[0]
	    numRlines += 1
	elif lfp:
	    lfp.write(lines[i])
	    numLlines += 1

    return

# ---------------------------
main()
