#!/usr/bin/env python3
'''
Routines for reporting information about the performance & tuning of models
and Pipelines.
Generally, each reporting function returns a formatted output string.
'''
import sys
import os.path
import time
import re
import string
from sklearn.metrics import fbeta_score, precision_score,\
                        recall_score, classification_report, confusion_matrix

SSTART = "### "                 # report output section start delimiter

#-----------------------------------

def getFormattedMetrics( \
    title,          # string title, typically  "Train" or "Test"
    y_true,         # true category assignments
    y_predicted,    # predicted assignments
    beta,           # for the fbeta_score
    rptClassNames=['yes', 'no'],
                    # class labels for report outputs, in desired order
    rptClassMapping=[1,0],
                    # rptClassMapping[y_val] = corresp name in rptClassNames
    rptNum=1,       # num classes to show in classification_report
                    #   will be the 1st rptNum names in rptClassNames
    yClassNames=['no', 'yes'],
                    # class labels for the actual y_values
    yClassToScore=1,# index of actual class in yClassNames to score
    sstart=SSTART,  # output section start delimiter
    ):
    '''
    Return formated metrics report for a set of predictions.
    y_true and y_predicted are lists of integer category indexes (y_vals).
    Assumes we are using a fbeta score.
    '''
    # concat title string onto all the target class names so
    #  they are easier to differentiate in multiple reports (and you can
    #  grep for them)
    target_names = [ "%s %s" % (title[:5], x) for x in rptClassNames ]

    output = sstart + "Metrics: %s\n" % title
    output += "%s\n" % (classification_report( \
                        y_true, y_predicted,
                        target_names=target_names,
                        labels=rptClassMapping[:rptNum], )
                    )
    output += "%s (%s) F%d: %6.4f    P: %6.4f    R: %6.4f    NPV: %6.4f\n\n" % \
            (
            title[:5],
            yClassNames[yClassToScore],
            beta,
            fbeta_score(y_true, y_predicted, beta=float(beta),
                                                 pos_label=yClassToScore),
            precision_score(y_true, y_predicted, pos_label=yClassToScore),
            recall_score(   y_true, y_predicted, pos_label=yClassToScore),
            # negative predictive value (NPV):
            precision_score(y_true, y_predicted, pos_label=1 - yClassToScore),
            )
    output += "%s\n" % getFormattedCM(y_true, y_predicted,
                                        rptClassNames=rptClassNames,
                                        rptClassMapping=rptClassMapping)
    return output
# ---------------------------

def getFormattedCM( \
    y_true,             # true category assignments 
    y_predicted,        # predicted assignments
    rptClassNames=['yes', 'no'],
                    # class labels for report outputs, in desired order
    rptClassMapping=[1,0],
                    # rptClassMapping[y_val] = corresp name in rptClassNames
    ):
    '''
    Return (minorly) formated confusion matrix
    FIXME: this could be greatly improved
    '''
    output = "%s\n%s\n" % ( \
            str(rptClassNames),
            str(confusion_matrix(y_true, y_predicted, labels=rptClassMapping)))
    return output
# ---------------------------

def getFalsePosNegReport( \
    title,		# string title, typically  "Train" or "Validation"
    falsePositives,	# list of (sample names/IDs) of the falsePositives
    falseNegatives,	# ... 
    num=5,		# number of false pos/negs to display
    sstart=SSTART,	# output section start delimiter
    ):
    '''
    Report on the false positives and false negatives in a validation set
    '''
    output = sstart+"False positives for %s: %d\n" % (title,len(falsePositives))
    for name in falsePositives[:num]:
        output += "%s\n" % name

    output += "\n"
    output += sstart+"False negatives for %s: %d\n" %(title,len(falseNegatives))
    for name in falseNegatives[:num]:
        output += "%s\n" % name

    output += "\n"
    return output
# ---------------------------

def getBestParamsReport( \
    bestParams, # dict parameter_names->(best) parameter values from GridSearch
    pipeline,		# Pipeline from the GridSearch
    sstart=SSTART,	# output section start delimiter
    ):
    """
    Return report (string) of the best parameters selected by a GridSearchCV.
    The idea here is to report enough info that the best Pipeline found in the
    GridSearch and its parameters are documented.
    """
    output = sstart +'Best Pipeline Parameters:\n'
    for pName in sorted(bestParams.keys()):
        output += "%s: %r\n" % ( pName, bestParams[pName] )
    output += "\n"

    for stepName, obj in pipeline.named_steps.items():
        output += "%s:\n%s\n" % (stepName, obj)
        if hasattr(obj, 'get_params'):
            output += "params: " + str(obj.get_params()) + "\n"
        output += "\n"
    output += "\n"

    return output
# ---------------------------

def getGridSearchReport( \
    gridSearch,		# fitted GridSearchCV object to report on
    parameters,		# dict of parameters used in the GridSearchCV
    sstart=SSTART,	# output section start delimiter
    ):
    """
    Return report (string) about the completed (fitted) GridSearchCV
    """
    if not gridSearch: return ''

    output = sstart + 'Grid Search Parameter Options Tried:\n'
    for key in sorted(parameters.keys()):
        output += "%s:%s\n" % (key, str(parameters[key]))
    output += "\n"

    output += sstart + 'Grid Search Scores:\n'
    params           = gridSearch.cv_results_['params']
    mean_test_scores = gridSearch.cv_results_['mean_test_score']
    for i, p in enumerate(params):
        output += str(p) + '\n'
        output += "mean_test_score:  %f\n" % mean_test_scores[i]
    output += "\n"

    output += sstart + 'Grid Search Best Score: %f\n' % gridSearch.best_score_
    output += "\n"
    return output
# ---------------------------

def getTopFeaturesReport(  \
    orderedFeatures,    # features: [ ('feature name', coef), ...]
    num=20,             # number of features w/ highest & lowest coefs to rpt
    sstart=SSTART,	# output section start delimiter
    ):
    '''
    Return report (string) of the features w/ the highest (positive) and lowest
    (negative) coefficients.
    Assumes num < len(orderedFeatures).
    '''
    if not orderedFeatures:             # no coefs
        output =  sstart + "Feature weights: not available\n"
        output += "\n"
        return output

    highest = orderedFeatures[:num]
    lowest  = orderedFeatures[len(orderedFeatures)-num:]

    output = sstart + "Feature weights: highest %d\n" % len(highest)
    for f,c in highest:
        output += "%+5.4f\t%s\n" % (c,f)
    output += "\n"

    output += sstart + "Feature weights: lowest %d\n" % len(lowest)
    for f,c in lowest:
        output += "%+5.4f\t%s\n" % (c,f)
    output += "\n"

    return output
# ---------------------------

def getVectorizerReport(vectorizer,
                        nFeatures=10,
                        sstart=SSTART,	# output section start delimiter
    ):
    '''
    Return report (string) on the fitted vectorizer
    '''
    featureNames = vectorizer.get_feature_names()
    midFeature   = int(len(featureNames)/2)

    output =  sstart + "Vectorizer:   Number of Features: %d\n" \
                                                % len(featureNames)
    output += "First %d features: %s\n\n" % (nFeatures,
                format(featureNames[:nFeatures]) )
    output += "Middle %d features: %s\n\n" % (nFeatures,
                format(featureNames[ midFeature : midFeature+nFeatures]) )
    output += "Last %d features: %s\n\n" % (nFeatures,
                format(featureNames[-nFeatures:]) )
    return output
# ---------------------------

def writeFeatures(  vectorizer, # fitted vectorizer from a pipeline
                    classifier, # fitted classifier from the pipeline
                    outputFile, # filename (string) or file obj (e.g., stdout)
                    values=None,# list of feature values, one for each feature
                                # This is some number per feature based on
                                #   analysis of the features. Typical example:
                                #   number of documents each feature appears in
                                #   (but could be anything)
                                # Values are printed with %d. This could
                                #  become a problem....
    ):
    '''
    Write the full list of feature names to the outputFile.
    if values is a list, should be numeric list parallel to feature names
        we will print these after the feature name.
    if classifier can provide feature coefficients, we will print these too
    All "|" delimited.
    Assumes:  vectorizer has get_feature_names() method
    '''
    delimiter = '|'
    featureNames = vectorizer.get_feature_names()

    hasValues = values != None

    hasCoef = hasattr(classifier, 'coef_')
    if hasCoef: coefficients = classifier.coef_[0].tolist()

    hasImportance = hasattr(classifier, 'feature_importances_')
    if hasImportance: importances = classifier.feature_importances_.tolist()

    if type(outputFile) == type(''): fp = open(outputFile, 'w')
    else: fp = outputFile

    for i,f in enumerate(featureNames):
        fp.write(f)
        if hasValues:     fp.write(delimiter + '%d' % values[i])
        if hasCoef:       fp.write(delimiter + '%+5.4f' % coefficients[i])
        if hasImportance: fp.write(delimiter + '%+5.4f' % importances[i])
        fp.write('\n')
    return
# ----------------------------

def getFormattedTime():
    return time.strftime("%Y/%m/%d-%H-%M-%S")
# ---------------------------
if __name__ == "__main__":
    # ad hoc test code
    if False:    # no tests yet
        pass
