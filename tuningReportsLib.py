#!/usr/bin/env python
'''
Routines for reporting information about the performance & tuning of models
and Pipelines.
Generally, each reporting function returns a formatted output string.
'''
import sys
import os.path
import re
import string
#from collections import Mapping, Iterable

#from sklearn.base import TransformerMixin, BaseEstimator
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

SSTART = "### "                 # report output section start delimiter

#-----------------------------------

def getTopFeaturesReport(  \
    orderedFeatures,    # features: [ ('feature name', coef), ...]
    num=20,             # number of features w/ highest & lowest coefs to rpt
    sectionStart=SSTART,
    ):
    '''
    Return report of the features w/ the highest (positive) and lowest
    (negative) coefficients.
    Assumes num < len(orderedFeatures).
    '''
    if not orderedFeatures:             # no coefs
        output =  sectionStart + "Feature weights: not available\n"
        output += "\n"
        return output

    highest = orderedFeatures[:num]
    lowest  = orderedFeatures[len(orderedFeatures)-num:]

    output = sectionStart + "Feature weights: highest %d\n" % len(highest)
    for f,c in highest:
        output += "%+5.4f\t%s\n" % (c,f)
    output += "\n"

    output += sectionStart + "Feature weights: lowest %d\n" % len(lowest)
    for f,c in lowest:
        output += "%+5.4f\t%s\n" % (c,f)
    output += "\n"

    return output
# ---------------------------

if __name__ == "__main__":
    # ad hoc test code
    if False:    # no tests yet
	pass
