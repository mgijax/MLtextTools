#!/bin/bash
# Given two files of records/lines,
#   typically 2 training files: "no" samples & "yes" samples,
# Write to stdout a single, combined output (training) file with a random subset
#   of records from one file and all the records from the other (2nd) file.
# Also write two files from the split of the 1st file:
#   the random lines and the other "leftover" lines.
#
# Typically we use this to create a balanced training set when the number
#   of "no" samples and "yes" samples are unbalanced

function Usage() {
    cat - <<ENDTEXT

$0  -n NUMKEEP   -r RANDOMFILE   -l LEFTOVERFILE
		    [--noheader]   FILE2RANDOMIZE  OTHERFILE
    NUMKEEP is number of random lines to select from FILE2RANDOMIZE.
    Write the randomly selected rcds to RANDOMFILE.
    Write the leftover rcds to LEFTOVERFILE.
    Write balanced file to stdout.
    Assume record separator is '\n'.
    If files have headers, include header in stdout, RANDOMFILE, LEFTOVERFILE

ENDTEXT
    exit 5
}
if [ $# -eq 0 ]; then Usage; fi
headerParam=""	# text for the header parameter to getRandomSubset.
		#   default: have a header line
randomRcdFile=""
leftoverRcdFile=""
numKeep=""

while [ $# -gt 0 ]; do
    case "$1" in
    -h|--help) Usage ;;
    -r) randomRcdFile="$2"   ; shift; shift; ;;
    -l) leftoverRcdFile="$2" ; shift; shift; ;;
    -n) numKeep="$2"         ; shift; shift; ;;
    --noheader) headerParam="--noheader"  ; shift; ;;
    -*|--*) echo "invalid option $1"; Usage ;;
    *)
	if [ $# -ne 2 ]; then Usage; fi
	fileToRandomize="$1";
	otherFile="$2";
	shift; shift;  break; ;;
    esac
done
if [ "$randomRcdFile" == "" -o "$leftoverRcdFile" == "" -o "$numKeep" == "" ];
then Usage; fi

getRandomSubset.py $headerParam -n $numKeep -r $randomRcdFile -l $leftoverRcdFile < $fileToRandomize

if [ "$headerParam" == "" ]; then  # have header lines
    head -1 $fileToRandomize > headerline.$$
    tail -n +2 $randomRcdFile >randomRcds.$$
    tail -n +2 $otherFile >otherRcds.$$
    cat headerline.$$ randomRcds.$$ otherRcds.$$
    rm -f headerline.$$ randomRcds.$$ otherRcds.$$
else
    cat $randomRcdFile $otherFile
fi
