#!/usr/bin/env python2.7

import string
import re

refKeyTerms = [ "References",
		"Reference",
		"Literature Cited",
		"Acknowledgements",
		"Acknowledgments",
		"Conflicts of Interest",
		"Conflict of Interest",
		]

class RefSectionRemover (object):
    '''
    Class to predict/remove Reference sections from (extracted) text strings
    '''

    def __init__(self,
		 keyTerms=refKeyTerms,	# terms that can begin a ref section
		 maxFraction=0.4	# max fraction of whole doc that the
		 			#  ref section is allowed to be
		):
	self.keyTerms = keyTerms
	self.maxFraction = maxFraction
	self.regexString = self.getRegexString()
	self.refRegex = re.compile(self.regexString)
    # -----------------------

    def removeRefSection(self, text):

	kw, refStart = self.predictRefSection(text)
	return text[:refStart]

    def getBody(self, text): return self.removeRefSection(text)
    # ----------------------------------

    def getRefSection(self, text):

	kw, refStart = self.predictRefSection(text)
	return text[refStart:]
    # ----------------------------------

    def predictRefSection(self, text):
	# predict ref section.
	# Return the key term matched and the position of the predicted
	#  refs section in text.

	textLength = len(text)
	matches = [ x for x in self.refRegex.finditer(text)]

	if len(matches) == 0:
	    lastKeyWord = "No match"
	    refStart = textLength
	else:
	    m = matches[-1]
	    lastKeyWord = m.group(1)	# last matched term
	    refStart = m.start(1)	# position at end of that term
	    refLength = textLength - refStart

	    if float(refLength)/textLength > self.maxFraction:
		lastKeyWord = "Ref Prediction > %4.2f" % self.maxFraction
		refStart = textLength

	return lastKeyWord, refStart
    # ----------------------------------

    def getRegexString(self):
	# for the set of keyTerms, return regex pattern string that matches
	# \n(term1|term2|term3)\n - with each term "spacesAndCaseInsensitive"
	# Assuming the doc still has newline's the '\n's match only if the
	#   keyTerm is on a line by itself - which works well for section
	#   headings.
	# If we apply this to docs where newline's have been removed, we'd
	#   to replace '\n's with '\b's to match work boundaries.
	#   (could add this as a param to constructor at some point)
	# The "spacesAndCaseInsensitive" comes from:
	#   Diff journals use different case conventions, and the text
	#   extraction algorithm sometimes converts case AND sometimes adds
	#   spaces between the individual letters (no idea why)

	regexStr = r'\n('
	insensitive = map(spacesAndCaseInsensitiveRegex, self.keyTerms)
	regexStr += '|'.join(insensitive)
	regexStr += r')\n'
	return regexStr
    # ----------------------------------

#------------------ end Class RefSectionRemover

def spacesAndCaseInsensitiveRegex(s):
    # for given string, return regex pattern string that matches the chars
    #  case insensitively and with optional spaces between the chars.
    reg = []
    for c in s:
	if c.isalpha():
	    reg.append('[%s%s]' % (c.upper(), c.lower()) )
	else:
	    reg.append('[%s]' % c)
    return '[ ]*'.join(reg)
# -----------------------

if __name__ == "__main__":
    rm = RefSectionRemover(maxFraction=0.5, keyTerms=['Refs'])
    print "Regex: %s" % rm.getRegexString()
    shortDoc = "this is the body ........... \nR EFS\n literature section"
    print "doc: %s" % shortDoc
    print "body: %s" % rm.getBody(shortDoc)
    print "refs: %s" % rm.getRefSection(shortDoc)
