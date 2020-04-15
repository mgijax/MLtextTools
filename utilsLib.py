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

def getConfig(fileList=[]):
    """
    Find config file(s) in parent directories above the current dir
    and return a ConfigParser object that has read the files.
    Optional fileList contains the names of config files that take precedence
    over the standard parent directory list.
    (later files in fileList take precedence)
    """
    cp = configparser.ConfigParser()
    cp.optionxform = str # make keys case sensitive

    # generate a path up multiple parent directories to search for config file
    # (up to 6 levels above)
    cl = ['/'.join(l)+'/config.cfg' for l in [['.']]+[['..']*i \
							for i in range(1,10)]]
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

