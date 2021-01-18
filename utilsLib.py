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

