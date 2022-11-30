#!/usr/bin/env python3
"""
# Some helpful MLtextTools utilities
# to run automated tests:  python test_utilsLib.py [-v]
"""
import sys
import os.path
import re
import string
import configparser

#-----------------------------------

def getConfig(fileName,         # config filename to look for
                parentDirs=0,   # number of parent dirs to look for the file in
                                #   (in addition to current dir)
                fileList=[]     # any additional config pathnames to look for
                ):
    """
    Find a config file searching the current directory and optionally n levels
        of parent directories.
    Return a ConfigParser object that has read the files.
    Optional fileList contains the names of config files that take precedence
        over the standard parent directory list.
    (later files in fileList take precedence)
    """
    cp = configparser.ConfigParser()
    cp.optionxform = str # make keys case sensitive

    # generate a path up multiple parent directories to search for config file
    cl = ['/'.join(l)+'/'+fileName for l in [['.']]+[['..']*i \
                                            for i in range(1,parentDirs+1)]]
    cl.reverse()    # Note: later files in the list override earlier files.

    cp.read(cl + fileList)
    return cp
#-----------------------------------

def importPyFile(pyFile):
    ''' Given a python file pathname (w/ or w/o .py), import the file
        as a module
    '''
    pyDir, pyFile = os.path.split(pyFile)
    if pyFile.endswith('.py'): pyFile = pyFile[:-3]

    if pyDir != '': sys.path.insert(0, pyDir)
    myModule =  __import__(pyFile)
    if pyDir != '': del sys.path[0]

    return myModule
#-----------------------------------

nonAsciiRE = re.compile(r'[^\x00-\x7f]')        # match non-ascii chars

def removeNonAscii(text):
    return nonAsciiRE.sub(' ',text)
#-----------------------------------

urls_re = re.compile(r'\b(?:https?://|www[.]|doi)\S*',re.IGNORECASE)

def removeURLsLower(text):
    """ Return text with URLs/DOIs removed and everything in lower case
    """
    return ' '.join(urls_re.split(text)).lower()
#-----------------------------------

token_re = re.compile(r'\b(\w+)\b',re.IGNORECASE)

def tokenPerLine(text):
    """ Return the text with all punctuation removed and each alphanumeric
        token on a line by itself (in token order)
    """
    return '\n'.join([m.group() for m in token_re.finditer(text)]) + '\n'

#-----------------------------------
# Text Transformation utilities
#
# Nomenclature:
# 'regex' = the text of a regular expression (as a string)
# 're'    = a regular expression object from the re module

class MatchRcd (object):
    """ A structure to hold the details of a match to a TextMapping
    """
    def __init__(self, matchType, start, end, matchText, preText, postText,
                        replText):
        self.matchType = matchType      # typically the name of the  mapping
        self.start     = start          # coord of the start of the match
        self.end       = end            # end coord. text[start:end] = match txt
        self.matchText = matchText      # the actual matched text
        self.preText   = preText        # context chars before the match
        self.postText  = postText       # context chars after the match
        self.replText  = replText       # the string that replaced the match

class TextMapping (object):
    """
    IS: A named mapping between a regex and some text that should replace
            any text that matches the regex.
        The replacement text can be a constant string
        OR be a function that takes the matching text (string) as an argument
            and returns the replacement string for that text.
    DOES: Computes replacement strings for a given matching text.
          Keeps track of the strings that were matched & replaced (MatchRcds)
            so you can get a report of what matched.
    """
    def __init__(self, name, regex, replacement, context=0):
        self.name = name
        self.regex = regex
        self.replacement = replacement
        self.numChars = context  # num of chars around the matching text to
                                 #   keep when recording matches to this mapping
        self.resetMatches()

    def resetMatches(self):
        """ Initialize the matchRcds, forgetting any rcds we already have.
        """
        self.matchRcds = []

    def getMatchRcds(self):
        """ Return list of MatchRcds since the last self.resetMatches()
        """
        return self.matchRcds

    def foundMatch(self, text, start, end):
        """ Process the fact that text[start:end] matched this TextMapping.
            Register the match and
            Return the string that should replace text[start:end].
        """
        matchText = text[start:end]

        if type(self.replacement) == type(''): # constant replacement string
            replacement = self.replacement
        else:                                  # function to call
            replacement = self.replacement(matchText)
            
        # Get n chars around the matching text
        preText  = text[max(0, start-self.numChars) : start]
        postText = text[end : end+self.numChars]

        # Record the match
        matchRcd = MatchRcd(self.name, start, end, matchText, preText,
                                                    postText, replacement)
        self.matchRcds.append(matchRcd)

        return replacement

# end class TextMapping -----------------------------------

class TextMappingFromStrings (TextMapping):
    """
    IS: A TextMapping that is built from a set of strings. 
    """
    def __init__(self, name, strings, replacement, context=0):
        regex = self._buildRegex(strings)
        super().__init__(name, regex, replacement, context=context)

    def _buildRegex(self, strings):
        """ Return a regex string that matches the list of strings
        """
        regexes = [ self._str2regex(s) for s in strings ]
        return '|'.join(regexes)

    def _str2regex(self, s):
        """ Return a regex string that matches s:
                Match s exactly surrounded by word boundaries.
            Override this method to customize the regex production
        """
        return escAndWordBoundaries(s)

# end class TextMappingFromStrings  -----------------------------------

class TextMappingFromFile (TextMappingFromStrings):
    """
    IS: A TextMappingFromStrings that is built from a set of strings read in
        from a file.
        'inFile' is either an open filepointer to read from
        or a filename (string)
    """
    def __init__(self, name, inFile, replacement, context=0):

        if type(inFile) == type(''): fp = open(inFile, 'r')
        else: fp = inFile

        # Assume: each line is a string to map, strip initial/final whitespaces
        #         from each line. Ignore blank/whitespace lines & comment lines.
        strings = [ s.strip() for s in fp.readlines()
                                        if s.strip() and not s.startswith('#')]

        if type(inFile) == type(''): fp.close()         # close if we opened it

        super().__init__(name, strings, replacement, context=context)

    def _str2regex(self, s):
        """ Return a regex string that matches s:
                Match s with arbitrary whitespace wherever s has whitespace,
                surrounded by word boundaries.
            Override this method to customize the regex production
        """
        return r'\b' + squeezeAndEscape(s) + r'\b'

# end class TextMappingFromFile  -----------------------------------

#---------------------------------
# Some handy string/regex helper functions
def escAndWordBoundaries(s):
    """ Take a string, and return regex string that matches that string
        exactly, surrounded by word boundaries.
        Note "word boundaries" (\b) is a little more subtle than I thought.
            With the surrounding \b's,
            if s begins or ends with a non-alphanumberic char,
            the resulting regex is not likely to match anything.
            So be careful if your strings begin or end w/ spaces or punct.
    """
    return r'\b' + re.escape(s) + r'\b'

def squeezeAndEscape(s):
    """ Take a string and return a regex string that matches that string
        but with arbitrary whitespaces wherever the original string has
        whitespace.
    """
    return r'\s+'.join( [re.escape(w) for w in s.split()] )

def spacedOutRegex(s):
    # for given str, return regex pattern str that matches the chars
    #  in the str with optional spaces between the chars.
    # Useful because sometimes the PDF text extraction inserts spaces
    #  if it is in all caps or bold or a larger font
    reg = []
    for c in s:
        reg.append('[%s]' % c)
    return '[ ]*'.join(reg)

#---------------------------------

class TextTransformer (object):
    """
    IS: an object that efficiently does a bunch of text transformations based
        on TextMappings.
        The idea is to make one honking regex from all the TextMappings so all
        the TextMappings can be applied in one pass through any text for
        efficiency.
    HAS: list of TextMappings
         Order is important. If two TextMappings match the same text, the
             1st one wins and the second is not matched/applied
    DOES: Apply the TextMappings to strings.
          Get reports back about what TextMappings where applied, where,
              how often, etc.
    EXAMPLE:
    # from utilsLib import TextMapping, TextTransformer
    # mappings = [ # context=5: report 5 chars of context around matches, dflt=0
    #     TextMapping('this2that', r'\bthis\b', 'that', context=5),
    #     TextMapping('upperFIG', r'\bfigure|fig\b', lambda x: x.upper()),
    #     ]              
    # tt = TextTransformer(mappings)
    #
    # for s in [list of strings...]:
    #     transformed_s = tt.transformText(s)
    #     doSomethingWith(transformed_s)
    #
    # print(tt.getReport())       # report of matches aggregated across all
    #
    ##  Or get report of matches for each string:
    # for i,s in enumerate([list of strings...]):
    #     transformed_s = tt.transformText(s)
    #     print("Matches For String %d\n" % i)
    #     print(tt.getReport())       # report of matches for this string
    #     tt.resetMatches()
    """
    def __init__(self, mappings,        # list of TextMappings w/ distinct names
                reFlags=re.IGNORECASE,  # default is to ignore case in matches
                ):
        """ Assumes all the TextMappings have unique names that don't contain
            "<" or ">"
        """
        self.reFlags = reFlags
        self.mappings = mappings                    # the list of mappings
        self.mappingDict = self._buildMappingDict() # {name : mapping obj}
        self.resetMatches()
        self.bigRegex = None
        self.bigRe = None
        self._buildBigRe()

    def _buildMappingDict(self):
        """ Build a dict of names to TextMappings
        """
        d = {}
        for m in self.mappings:
            if m.name in d:
                msg = "Two TextMappings have the same name: '%s'\n" % m.name
                raise ValueError(msg)

            d[m.name] = m
        return d

    def _buildBigRe(self):
        """ combine all the mappings into one big re
        """
        # build a named group for each regex: e.g., '(?P<name>regex)'
        namedRegexes = ['(?P<' + m.name + '>' + m.regex + ')'
                                                        for m in self.mappings] 
        self.bigRegex = '|'.join(namedRegexes)

        self.bigRe = re.compile(self.bigRegex, self.reFlags)

    def getBigRegex(self): return self.bigRegex
    def getBigRe(self):    return self.bigRe

    def transformText(self, text):
        """ Apply the mappings to the text. Return the transformed text
        """
        transformed = ''
        endOfLastMatch = 0      # end position in text of the last match
                                #  processed so far
        for m in self.bigRe.finditer(text):
            name, start, end = findMatchingGroup(m)
            replacement = self.mappingDict[name].foundMatch(text, start, end)
            transformed += text[endOfLastMatch:start] + replacement
            endOfLastMatch = end

        transformed += text[endOfLastMatch:]
        return transformed

    def getMatches(self):
        """ Return list of MatchRcds for matches found so far by this 
            TextTransformer.
        """
        matches = []
        for m in self.mappings:
            matches.extend(m.getMatchRcds())
        return matches

    def getReport(self, title="Text Transformation Report"):
        """ Return a string: nicely formatted matches report
            1st line: title
            2nd line: column headers (tab delimited)
            Tab delimited lines (sorted by matchType, matchText, postText):
            matchType, replText, count, preText, matchText, postText
            (count = number of occurrances of
                preText matchText postText -> preText replText postText)
        """
        output = title + "\n"
        output += '\t'.join(['matchType',        # header line
                                'replText',
                                'numMatches',
                                'preText',
                                'matchText',
                                'postText',
                                ]) + '\n'

        # aggregate matches to counts
        aggMatches = {}
        for m in self.getMatches():
                # order of these fields is intentional so matches sort nicely
            myKey = (m.matchType, m.matchText, m.postText, m.preText,
                                                                    m.replText)
            aggMatches[myKey] = aggMatches.get(myKey, 0) + 1

        # generate output lines w/ counts.
        for myKey in sorted(aggMatches.keys()):
            numMatches = aggMatches[myKey]
            (matchType, matchText, postText, preText, replText) = myKey
            matchText = matchText.replace('\n', '\\n').replace('\t', '\\t')
            postText  = postText.replace('\n', '\\n').replace('\t', '\\t')
            preText   = preText.replace('\n', '\\n').replace('\t', '\\t')
            replText  = replText.replace('\n', '\\n').replace('\t', '\\t')
            line = '\t'.join([matchType,
                                "'%s'" % replText,
                                str(numMatches),
                                "'%s'" % preText,
                                "'%s'" % matchText,
                                "'%s'" % postText,
                                ])
            output += line + '\n'
        return output

    def resetMatches(self):
        """ Clear the matches seen so far by this TextTransformer
        """
        for m in self.mappings:
            m.resetMatches()
# end class TextTransformer -----------------------------------

def findMatchingGroup(m):
    """
    Given an re.Match object, m,
    Find the key (name) of its group that actually matched.
    Return the key & start & end coords of the matching string
    """
    gd = m.groupdict()     # dict of groups in m: {groupname: re.match|None}

    for k in [k for k in gd.keys() if gd[k]]:
        return (k, m.start(k), m.end(k))        # return 1st matching group

    return (None, None, None) # shouldn't happen since some group should match
