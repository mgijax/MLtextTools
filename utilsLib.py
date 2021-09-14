#!/usr/bin/env python3
"""
# Some helpful utilities
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

class TextMapping (object):
    """
    IS: a named mapping between a regex and the text that should replace
        any text that matches the regex.
    DOES: keeps track of the text strings that were replaced
    """
    def __init__(self, name, regex, replacement, context=0):
        self.name = name
        self.regex = regex
        self.replacement = replacement
        self.context = context  # num of chars around the matching string to
                                #   keep when recording matches to this mapping
        self.resetMatches()

    def resetMatches(self):
        self.matches = {}       # {'matching text string': count}

    def wasMatched(self, text, start, end):
        """ Register the fact that text[start:end] was matched/replaced by this
            TextMapping
        """
        # Determine the matching text
        if self.context:        # record n chars around the matching text
            preText  = text[max(0, start-self.context) : start]
            postText = text[end : end+self.context]
            matchingText = "%s|%s|%s" % (preText, text[start:end], postText)
        else:                   # just the matching text
            matchingText = text[start:end]

        # Update the count for this matching text
        self.matches[matchingText] = self.matches.get(matchingText, 0) + 1

    def getMatches(self): return self.matches
# end class TextMapping -----------------------------------

class TextMappingFromStrings (TextMapping):
    """
    IS: A TextMapping that is built from a set of strings. 
        The strings may be read in from a file.
    """
    def __init__(self, name, strings, replacement, context=0):
        regex = self._buildRegex(strings)
        super().__init__(name, regex, replacement, context=context)

    def _buildRegex(self, strings):
        """
        Return a regex string that matches the list of strings
        """
        regexes = []
        for s in strings:
            regex = self._str2regex(s)
            regexes.append(regex)
        
        return '|'.join(regexes)

    def _str2regex(self, s):
        """ convert string to regex string.
            Override this method to customize the regex production
        """
        return escAndWordBoundaries(s)

# end class TextMappingFromStrings  -----------------------------------

class TextMappingFromFile (TextMappingFromStrings):
    """
    IS: A TextMappingFromStrings that is built from a set of strings read in
        from a file.
    """
    def __init__(self, name, inFile, replacement, context=0):

        if type(inFile) == type(''): fp = open(inFile, 'r')
        else: fp = inFile

        # Assume: each line is a string to map, strip initial/final whitespaces
        #         from each line. Ignore blank/whitespace lines.
        strings = [ s.strip() for s in fp.readlines()  
                                        if s.strip() and not s.startswith('#')]

        if type(inFile) == type(''): fp.close()         # close if we opened it

        super().__init__(name, strings, replacement, context=context)

    def _str2regex(self, s):
        """ convert string to regex string.
        """
        return r'\b' + squeezeAndEscape(s) + r'\b'

# end class TextMappingFromFile  -----------------------------------

#---------------------------------
# Some handy string/regex helper functions
def escAndWordBoundaries(s):
    return r'\b' + re.escape(s) + r'\b'

def squeezeAndEscape(s):
    """ squeeze white space, replace with r'\s', and re.escape all other chars
    """
    return r'\s'.join( [re.escape(w) for w in s.split()] )

#---------------------------------

class TextTransformer (object):
    """
    IS: an object that efficiently does a bunch of text transformations based
        on TextMappings.
        The idea is to make one honking regex from all the TextMappings so all
        the TextMappings can be applied in one pass through any text for
        efficiency.
    HAS: bunch of TextMappings
    DOES: apply the TextMappings to one or more strings,
          get reports back about what TextMappings where applied, where,
          how often, etc.
    """
    def __init__(self, mappings,        # list of TextMappings w/ distinct names
                reFlags=re.IGNORECASE,  # default is to ignore case in matches
                ):
        """
        Assumes all the TextMappings have unique names that don't contain
            "<" or ">"
        """
        self.reFlags = reFlags
        self.mappings = mappings                    # the list of mappings
        self.mappingDict = self._buildMappingDict() # dict of mappings
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

    def getMatches(self):
        """ Return a report of all the text matches/transformations found.
            I.e., a list of triples:
                (mapping name, {'matching text':count}, replacement str)
        """
        matches = []
        for k in sorted(self.mappingDict.keys()):
            mapping = self.mappingDict[k]
            if mapping.getMatches():
                matches.append((k, mapping.getMatches(), mapping.replacement))

        return matches

    def getMatchesReport(self, title="Text Transformation Report"):
        """ Return a string: nicely formatted matches report"""
        output = title + "\n"

        for match in self.getMatches():
            name, matches, replacement = match
            for k in sorted(matches.keys()):
                text = k.replace('\n', '\\n').replace('\t', '\\t')
                output += "%s\t'%s'\t%d\t'%s'\n" % \
                                    (name, replacement, matches[k], text,)
        return output

    def transformText(self, text):
        """ Apply the mappings to the text. Return the transformed text
        """
        transformed = ''
        endOfLastMatch = 0      # end position in text of the last match
                                #  processed so far
        for m in self.bigRe.finditer(text):
            key, start, end = findMatchingGroup(m)
            self.mappingDict[key].wasMatched(text, start, end)
            replacement = self.mappingDict[key].replacement
            transformed += text[endOfLastMatch:start] + replacement
            endOfLastMatch = end

        transformed += text[endOfLastMatch:]
        return transformed

    def resetMatches(self):
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
