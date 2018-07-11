#!/usr/bin/env python2.7

#
#  Purpose:
#	   run sql to get references and their extracted text.
#
#  Outputs:     writes directories named by Journal and writes extracted text
#		files (named by Pubmed ID) into those directories
#
###########################################################################
import sys
import os
import string
import time
import argparse
#from ConfigParser import ConfigParser
import db

#-----------------------------------

def getArgs():
    parser = argparse.ArgumentParser( \
                    description='get extracted text for references')

    parser.add_argument('-s', '--server', dest='server', action='store',
        required=False, default='dev',
        help='db server: adhoc, prod, or dev (default)')

    parser.add_argument('-o', '--output', dest='outputDir', action='store',
        required=False, default='.',
        help="output directory. Default: '.'")

    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
        required=False, help="skip helpful messages to stderr")

    args =  parser.parse_args()

    if args.server == 'adhoc':
	args.host = 'mgi-adhoc.jax.org'
	args.db = 'mgd'
    if args.server == 'prod':
	args.host = 'bhmgidb01'
	args.db = 'prod'
    if args.server == 'dev':
	args.host = 'bhmgidevdb01'
	args.db = 'prod'

    return args
###################################3

SQLSEPARATOR = '||'
QUERY =  \
'''
select a.accid pubmed, r.journal, r.title, bd.extractedtext
from bib_refs r join bib_workflow_data bd on (r._refs_key = bd._refs_key)
     join acc_accession a on
	 (a._object_key = r._refs_key and a._logicaldb_key=29 -- pubmed
	  and a._mgitype_key=1 )
where
r.creation_date > '10/01/2017'
-- r.year in (2014)
and r._referencetype_key=31576687 -- peer reviewed article
and bd.haspdf=1
-- and r.isdiscard = 0
-- and bd._supplemental_key =34026997  -- "supplemental attached"
-- limit 10
'''

def getExtractedText ():

    args = getArgs()

    db.set_sqlServer  ( args.host)
    db.set_sqlDatabase( args.db)
    db.set_sqlUser    ("mgd_public")
    db.set_sqlPassword("mgdpub")

    if args.verbose:
	sys.stderr.write( "Hitting database %s %s as mgd_public\n\n" % \
							(args.host, args.db))

    queries = string.split(QUERY, SQLSEPARATOR)

    startTime = time.time()
    results = db.sql( queries, 'auto')
    endTime = time.time()
    if args.verbose:
	sys.stderr.write( "Total SQL time: %8.3f seconds\n\n" % \
							(endTime-startTime))
	sys.stderr.write( "Writing files..")

    for i,r in enumerate(results[-1]):
	journal = '_'.join( r['journal'].split(' ') )
	dirname =  os.sep.join( [ args.outputDir, journal ] )
	if not os.path.exists(dirname):
	    os.makedirs(dirname)
	filename = os.sep.join( [ dirname, r['pubmed'] ] )

	if args.verbose and i % 1000 == 0:	# write progress indicator
	    sys.stderr.write('%d..' % i)
	with open(filename, 'w') as fp:
	    fp.write( str(r['extractedtext']) )
    if args.verbose:
	sys.stderr.write("\nNumber of files written: %d\n\n" % len(results[-1]))

# end getExtractedText() ----------------------------------

#
#  MAIN
#
if __name__ == "__main__": getExtractedText()
