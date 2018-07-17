#!/usr/bin/env python2.7

"""
Classes for predicting the location of reference sections in text from
scientific articles
"""

import string
import re

class BaseRefSectionRemover (object):
    '''
    Abstract base class for reference section predictor/removal classes
    '''
    def removeRefSection(self, text):
	"""
	return text with the predicted reference section removed
	"""

	kw, refStart = self.predictRefSection(text)
	return text[:refStart]

    def getBody(self, text): return self.removeRefSection(text)
    # ----------------------------------

    def getRefSection(self, text):
	"""
	Return the predicted reference section (string)
	"""

	kw, refStart = self.predictRefSection(text)
	return text[refStart:]
    # ----------------------------------

    def predictRefSection(self, text):
	"""
	Abstract method.
	Predict ref section.
	Return the text found that we presume starts of the reference
	    section and the position of it in full text.
	"""
	pass
	return "splitKeyWord", len(text)
    # ----------------------------------

    def buildRegexString(self, keyTerms):
	"""
	# for the set of keyTerms, return regex pattern string that matches
	# \n(term1|term2|term3)\n - with each term "spacesAndCaseInsensitive"
	# Assuming the doc still has newline's the '\n's match only if the
	#   keyTerm is on a line by itself - which works well for section
	#   headings.
	# If we apply this to docs where newline's have been removed, we'd
	#   have to replace '\n's with '\b's to match work boundaries.
	#   (could add this as an option at some point)
	# Why "spacesAndCaseInsensitive"?
	#   Diff journals use different case conventions, and the text
	#   extraction algorithm sometimes converts case AND sometimes adds
	#   spaces between the individual letters (no idea why)
	"""
	regexStr = r'\n('
	insensitive = map(spacesAndCaseInsensitiveRegex, keyTerms)
	regexStr += '|'.join(insensitive)
	regexStr += r')\n'
	return regexStr
    # ----------------------------------

    def isReasonableRefStart(self, refStart, totalTextLength):
	"""
	Is the predicted reference start location ok?
	(or is it too early in the document so too much of the doc would be
	 thrown away?)
	Assumes subclass has self.maxFraction
	"""
	refLength = totalTextLength - refStart

	return float(refLength)/totalTextLength <= self.maxFraction
    # ----------------------------------

    def getRegexString(self):
	"""
	Assumes subclass has self.regexString
	"""
	return self.regexString
    # ----------------------------------

#------------------ end Class BaseRefSectionRemover

class RefSectionRemover (BaseRefSectionRemover):
    '''
    Class to predict/remove Reference sections from (extracted) text strings.
    Approach:
    Looks 1st for primary key terms, and if found uses their earliest location.
    If no primary key terms are found, then look for other potential terms, 
    use earliest location.
    New:  move key term "Reference" into the secondary term list
          Add minimum distance for key word from the end of the doc
    '''
    # Dict of key terms and whether they are a primary term
    keyTermDict = {
		"References":            {"isPrimary":True},
		"Literature Cited":      {"isPrimary":True},
		"Reference":             {"isPrimary":False},
		"Acknowledgements":      {"isPrimary":False},
		"Acknowledgments":       {"isPrimary":False},
		"Conflicts of Interest": {"isPrimary":False},
		"Conflict of Interest":  {"isPrimary":False},
		}
    # dict of squashed key terms (no spaces, lower case)
    squashedKeyTerms = \
	dict( [ [x.replace(" ","").lower(), x] for x in keyTermDict.keys() ] )

    def __init__(self,
		 minFraction=0.05,	# min fraction predicted for ref section
		 maxFraction=0.4,	# max fraction of whole doc that the
		 			#  ref section is allowed to be
		):
	self.keyTerms = self.keyTermDict.keys()
	self.minFraction = minFraction
	self.maxFraction = maxFraction
	self.regexString = self.buildRegexString(self.keyTerms)
	self.refRegex = re.compile(self.regexString)
    # -----------------------

    def predictRefSection(self, text):
	"""
	Predict ref section.
	Return the text found that we presume starts of the reference
	    section and the position of it in full text.
	"""
	textLength = len(text)

	# match the reg exp and get list of re.match objects (in order matched)
	reMatches = [ x for x in self.refRegex.finditer(text)]

	if len(reMatches) == 0:
	    return "No match", textLength

	primary, secondary = self.organizeMatches(reMatches)

	primary.reverse()
	secondary.reverse()
	for m in primary:
	    if self.isReasonableRefStart(m['termStart'], textLength):
		return m['matchedText'], m['termStart']

	for m in secondary:
	    if self.isReasonableRefStart(m['termStart'], textLength):
		return m['matchedText'], m['termStart']

	# no match with reasonable location found
	reason = "No match in loc constraints" 
	return reason, textLength
    # ----------------------------------

    def isReasonableRefStart(self, refStart, totalTextLength):
	"""
	Is the predicted reference start location ok?
	(or is it too early in the document so too much of the doc would be
	 thrown away?)
	Assumes subclass has self.maxFraction
	"""
	refLength = totalTextLength - refStart
	percentRefSection = float(refLength)/totalTextLength
	return percentRefSection <= self.maxFraction \
		and percentRefSection >=self.minFraction
    # ----------------------------------

    def organizeMatches(self, reMatches):
	"""
	Return two lists: "match descriptions" to primary terms and to
	    secondary terms.
	Each "match description" is
	    {'termStart': n, 'matchedText': str, 'keyTerm': str}
	"""
	primary = []
	secondary = []

	for reM in reMatches:

	    # assume there is only one group (w/ the matching text) per reMatch
	    matchedText = reM.group(1)
	    keyTerm = self.getMatchedKeyTerm(matchedText)
	    matchDesc = {'matchedText': matchedText,
			'keyTerm':     keyTerm,
			'termStart':   reM.start(1),
			}
	    if self.keyTermDict[keyTerm]['isPrimary']:
		primary.append(matchDesc)
	    else:
		secondary.append(matchDesc)

	return primary, secondary
    # ----------------------------------

    def getMatchedKeyTerm(self, matchedText):
	"""
	Convert matchedText to an actual key term.
	(remove spaces, lower case)
	This effectively "undoes" the weirdness we do to create the regular
	expression for matching the key terms
	"""
	cleansedTerm = matchedText.replace(" ","").lower()
	return self.squashedKeyTerms[cleansedTerm]
    # ----------------------------------

#------------------ end Class RefSectionRemover

class RefSectionRemover_Jul12 (BaseRefSectionRemover):
    '''
    Class to predict/remove Reference sections from (extracted) text strings.
    Approach:
    Looks 1st for primary key terms, and if found uses their earliest location.
    If no primary key terms are found, then look for other potential terms, 
    use earliest location.
    '''
    # Dict of key terms and whether they are a primary term
    keyTermDict = {
		"References":            {"isPrimary":True},
		"Reference":             {"isPrimary":True},
		"Literature Cited":      {"isPrimary":True},
		"Acknowledgements":      {"isPrimary":False},
		"Acknowledgments":       {"isPrimary":False},
		"Conflicts of Interest": {"isPrimary":False},
		"Conflict of Interest":  {"isPrimary":False},
		}
    # dict of squashed key terms (no spaces, lower case)
    squashedKeyTerms = \
	dict( [ [x.replace(" ","").lower(), x] for x in keyTermDict.keys() ] )

    def __init__(self,
		 maxFraction=0.4	# max fraction of whole doc that the
		 			#  ref section is allowed to be
		):
	self.keyTerms = self.keyTermDict.keys()
	self.maxFraction = maxFraction
	self.regexString = self.buildRegexString(self.keyTerms)
	self.refRegex = re.compile(self.regexString)
    # -----------------------

    def predictRefSection(self, text):
	"""
	Predict ref section.
	Return the text found that we presume starts of the reference
	    section and the position of it in full text.
	"""
	textLength = len(text)

	# match the reg exp and get list of re.match objects (in order matched)
	reMatches = [ x for x in self.refRegex.finditer(text)]

	if len(reMatches) == 0:
	    return "No match", textLength

	primary, secondary = self.organizeMatches(reMatches)

	for m in primary:
	    if self.isReasonableRefStart(m['termStart'], textLength):
		return m['matchedText'], m['termStart']

	for m in secondary:
	    if self.isReasonableRefStart(m['termStart'], textLength):
		return m['matchedText'], m['termStart']

	# no match with reasonable location found
	reason = "Ref Prediction > %4.2f" % self.maxFraction
	return reason, textLength
    # ----------------------------------

    def organizeMatches(self, reMatches):
	"""
	Return two lists: "match descriptions" to primary terms and to
	    secondary terms.
	Each "match description" is
	    {'termStart': n, 'matchedText': str, 'keyTerm': str}
	"""
	primary = []
	secondary = []

	for reM in reMatches:

	    # assume there is only one group (w/ the matching text) per reMatch
	    matchedText = reM.group(1)
	    keyTerm = self.getMatchedKeyTerm(matchedText)
	    matchDesc = {'matchedText': matchedText,
			'keyTerm':     keyTerm,
			'termStart':   reM.start(1),
			}
	    if self.keyTermDict[keyTerm]['isPrimary']:
		primary.append(matchDesc)
	    else:
		secondary.append(matchDesc)

	return primary, secondary
    # ----------------------------------

    def getMatchedKeyTerm(self, matchedText):
	"""
	Convert matchedText to an actual key term.
	(remove spaces, lower case)
	This effectively "undoes" the weirdness we do to create the regular
	expression for matching the key terms
	"""
	cleansedTerm = matchedText.replace(" ","").lower()
	return self.squashedKeyTerms[cleansedTerm]
    # ----------------------------------

#------------------ end Class RefSectionRemover_new

class RefSectionRemover_orig (BaseRefSectionRemover):
    '''
    Class to predict/remove Reference sections from (extracted) text strings
    Simple approach, look for last occurrance of key terms that could be
       close to the reference section start.
    '''
    refKeyTerms = [ "References",	# terms that can begin a ref section
		    "Reference",
		    "Literature Cited",
		    "Acknowledgements",
		    "Acknowledgments",
		    "Conflicts of Interest",
		    "Conflict of Interest",
		    ]

    def __init__(self,
		 keyTerms=refKeyTerms,
		 maxFraction=0.4	# max fraction of whole doc that the
		 			#  ref section is allowed to be
		):
	self.keyTerms = keyTerms
	self.maxFraction = maxFraction
	self.regexString = self.buildRegexString(self.keyTerms)
	self.refRegex = re.compile(self.regexString)
    # -----------------------

    def predictRefSection(self, text):
	"""
	Predict ref section.
	Return the text found that we presume starts of the reference
	    section and the position of it in full text.
	"""
	textLength = len(text)
	matches = [ x for x in self.refRegex.finditer(text)]

	if len(matches) == 0:
	    splitKeyWord = "No match"
	    refStart = textLength
	else:
	    m = matches[-1]
	    splitKeyWord = m.group(1)	# last matched term
	    refStart = m.start(1)	# position at end of that term

	    if not self.isReasonableRefStart(refStart, textLength):
		splitKeyWord = "Ref Prediction > %4.2f" % self.maxFraction
		refStart = textLength

	return splitKeyWord, refStart
    # ----------------------------------
	reason = "Ref Prediction > %4.2f" % self.maxFraction

#------------------ end Class RefSectionRemover1

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
    print
    print "Latest approach"
    rm = RefSectionRemover(maxFraction=0.6,)
    print "Regex: %s" % rm.getRegexString()
    shortDoc = "Here is the body ........... \nR eferences\n Here is literature section"
    print "whole doc: '%s'" % shortDoc
    print rm.predictRefSection(shortDoc)
    print "body: '%s'" % rm.getBody(shortDoc)
    print "refs: '%s'" % rm.getRefSection(shortDoc)
