'''
Sklearn helper and other basic text stuff.

In particular classes/functions to do stemming.

Convention: trying to use camelCase for all the names here, but
    sklearn typically_uses_names with underscores.

SEE NOTES ON STEMMING AT THE BOTTOM OF THIS FILE...
'''
import sys
import os.path
import re
import string
from collections.abc import Mapping, Iterable

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk.stem.snowball as nltk

#-----------------------------------

def isOneCombination(paramGrid,	# GridSearchCV params: dict or list of dicts
    ):
    '''
    Return True if there is only one combination of parameters
    Return False if there are multiple combinations
    '''

    if isinstance(paramGrid, Mapping):
        for l in paramGrid.values():
            if len(l) > 1: return False

    elif isinstance(paramGrid, Iterable): # list of dictionaries
        return False

    else:
        raise TypeError('Parameter grid is not a dict or list of dicts')

    return True
#-----------------------------------

def convertParamGrid(paramDict,	# GridSearchCV params: dict
    ):
    '''
    "Flatten" a GridSearchCV parameter dictionary, e.g., 
    { key0 : [l0, l1, ...], key1: [m0, m1, ...] }
    --> {key0: l0, key1: m0}
    '''
    result = {}
    for key,val in paramDict.items():
        result[key] = val[0]
    return result
#-----------------------------------

def getFalsePosNeg( y_true,		# [true vals], typically 0/1 or yes/no 
                    y_predicted,	# [pred vals], typically 0/1 or yes/no 
                    sampleNames,	# [ sample names]
                    positiveClass=1,    # value in y_ considered "pos"
    ):
    '''
    Return lists of (sample names of) false positives and false negatives
        comparing true values to predicted ones
    Assumes y_true, y_predicted, sampleNames are parallel lists
    '''
    falsePositives = []
    falseNegatives = []

    for trueY, predY, name in zip(y_true, y_predicted, sampleNames):

        predType = predictionType(trueY, predY, positiveClass=positiveClass)
        if   predType == 'FP':
            falsePositives.append(name)
        elif predType == 'FN':
            falseNegatives.append(name)

    return falsePositives, falseNegatives
# ---------------------------

def predictionType( y_true,		# true value, typically 0/1 or yes/no 
                    y_predicted,	# pred value, typically 0/1 or yes/no 
                    positiveClass=1,	# value in y_ considered "pos"
    ):
    '''
    Return 'FP', 'FN', 'TP', or 'TN' depending on y_true and y_predicted
    '''
    if y_true == y_predicted:
        if y_predicted == positiveClass: retVal = "TP"
        else: retVal = 'TN'
    else:
        if y_predicted == positiveClass: retVal = 'FP'
        else: retVal = 'FN'
    return retVal
# ---------------------------

def getOrderedFeatures( vectorizer,     # fitted vectorizer from a pipeline
                        classifier      # trained classifier from a pipeline
    ):
    '''
    Return list of pairs, [ (feature, coef), (feature, coef), ... ]
        ordered from highest coef to lowest.
    Assumes:  vectorizer has get_feature_names() method
    '''
    if hasattr(classifier, 'feature_importances_'):
        coefficients = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        coefficients = classifier.coef_[0].tolist()
    else:
        return []

    featureNames = vectorizer.get_feature_names()

    pairList = zip(featureNames, coefficients)

    selCoef = lambda x: x[1]    # select the coefficient in the pair
    return sorted(pairList, key=selCoef, reverse=True)
# ---------------------------

def getConfidenceValues(classifier,
                        samples,		# list of samples to predict
                        positiveClass=1,	# value in y_ considered "pos"
                        ):
    """
    Return a list of "confidence" values for the prediction of each sample.
    If classifier does not have any mechanism to return confidences, return None
    Assumes predictions are binary (for two classes, positive and negative)
    """
    if classifier and hasattr(classifier, "decision_function"):
        confidences = classifier.decision_function(samples).tolist()
    elif classifier and hasattr(classifier, "predict_proba"):
        confidences = getProbaConfidences(classifier, samples,
                                                positiveClass=positiveClass)
    else:
        confidences = None
    return confidences
# ---------------------------

def getProbaConfidences(classifier,
                        samples,		# list of samples to predict
                        positiveClass=1		# value in y_ considered "pos"
                        ):
    """
    Compute "confidence" values for predictions when classifier has
        predict_proba() method
    Mimic how decision_function() works in classifiers w/ that method:
        Return "confidence" for the predicted class, but make this:
            pos if the predicted class is the positiveClass (index)
            neg if the predicted class is not the positiveClass
    See
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier.decision_function
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba
    https://datascience.stackexchange.com/questions/22762/understanding-predict-proba-from-multioutputclassifier

    Assumes predictions are binary (for two classes, positive and negative)
    Assumes classifier hasattr(classifier, "predict_proba"):

    JIM: This works for RandomForestClassifier. 
    Need to check this is consistent for other classifiers w/ predict_proba.
    """
    confidences = []
    negativeClass = 1 - positiveClass

    values = classifier.predict_proba(samples)

    for i in range(len(values)):            # for each doc
        posProb = values[i][positiveClass]
        negProb = values[i][negativeClass]
        if posProb > negProb:
            confidences.append(posProb)
        else:
            confidences.append(-negProb)
    return confidences
# ---------------------------

class FeatureDocCounter(BaseEstimator, TransformerMixin):
    """
    An sklearn Estimator that gives you access to the number of documents each
        feature occurs in.
    Put this "Estimator" into your pipeline after the vectorizer,
    then Access this list of counts via getValues()

    Example Usage:
    # (1) define the featureEvaluator step in your pipeline

        import sklearnHelperLib as skHelper
        pipeline = Pipeline( [
        ('vectorizer',       CountVectorizer(),),
        ('featureEvaluator', skHelper.FeatureDocCounter()),
        ('classifier',       SGDClassifier() ),
        ] )
    # (2) Run your GridSearchCV

        gs = GridSearchCV(pipeline, ...)
        gs.fit( docs, y)

    # (3) get your featureEvaluator back from the best_estimator

        featureEvaluator = gs.best_estimator.named_steps['featureEvaluator']
        featureDocCounts = featureEvaluator.getValues()

        # now featureDocCounts is a list parallel to the vectorizer's
        #   feature names
        vectorizer = gs.best_estimator.named_steps['vectorizer']
        featureNames = vectorizer.get_feature_names()

        for feat, count in zip(featureNames, featureDocCounts):
            ...
    """
    def transform(self, X):
        '''don't actually transform X, just gather the counts.'''

        # I wish I understood scipy.sparse matrices and/or numpy.matrix,
        #   but these magic words seem to work (after trial and error)
        self.docCounts = X.sum(axis=0,dtype=int).tolist()[0]
        return X

    def fit(self, X, y=None, **fit_params):
        '''nothing to actually fit'''
        return self

    def getValues(self):
        """ Return list of counts, one integer for each feature,
            in feature order.
            Each count is the number of docs the feature occurs in.
        """
        return self.docCounts
# end class FeatureDocCounter ----------

# ---------------------------
# Stemming....
# Probably best to preprocess the whole data set once
#  and stem it (and remove URLs) if stemming makes a big enough difference.
#
# Stemming in Vectorizer subclasses:
# See: https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
# This is subtle:
# Vectorizers have build_preprocessor() method that returns a preprocessor()
#   function.
# The preprocessor() function is called for each document (string) to do any
#   preprocessing, returning string.
# What we do here:    Subclass each of the common Vectorizers
#  and override the build_preprocessor() method to return a stemming
#    preprocessor function.
# ---------------------------
stemmer = nltk.EnglishStemmer()
token_re = re.compile("\\b([a-z_]\w+)\\b",re.IGNORECASE) # match words

class StemmedCountVectorizer(CountVectorizer):
    def build_preprocessor(self):# override super's build_preprocessor method
        '''
        Return preprocessor function that stems.
        '''
        # get the super class's preprocessor function for this object.
        preprocessor = super(type(self), self).build_preprocessor()

        # Tokenize and stem the string returned by the super's preprocessor
        #   method.
        # This should stem all words in  {bi|tri|...}grams and preserve any
        #  functionality implemented in the preprocessor.
        # (at the cost of an extra tokenizing step)
        def my_preprocessor( doc):
            output = ''
            for m in token_re.finditer( preprocessor(doc) ):
                output += " " + stemmer.stem(m.group())
            return output

        return my_preprocessor
# ---------------------------

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_preprocessor(self):# override super's build_preprocessor method
        '''
        Return preprocessor function that stems.
        '''
        # get the super class's preprocessor function for this object.
        preprocessor = super(type(self), self).build_preprocessor()

        # Tokenize and stem the string returned by the super's preprocessor
        #   method.
        # This should stem all words in  {bi|tri|...}grams and preserve any
        #  functionality implemented in the preprocessor.
        # (at the cost of an extra tokenizing step)
        def my_preprocessor( doc):
            output = ''
            for m in token_re.finditer( preprocessor(doc) ):
                output += " " + stemmer.stem(m.group())
            return output

        return my_preprocessor

# ---------------------------

# Different stemming approach: Stemming in a custom preprocessor.
# This might be faster than the above classes since we will be stemming
#  at the same time as the rest of the preprocessor.
# Also you CAN try vectorizer_preprocessor{_stem} as options in GridSearch.
#  BUT this doesn't generalize to arbritary preprocessors.

urls_re = re.compile("\\bhttps?://\\S*",re.IGNORECASE) # match URLs

def vectorizer_preprocessor_stem(input):
    '''
    Cleanse documents (strings) before they are passed to a vectorizer
       tokenizer.
    Currently: lower case everyting, remove URLs, and stem
    To use:
    vectorizer = CountVectorizer(preprocessor=vectorizer_preprocessor_stem)
    '''
    output = ''
    
    for s in urls_re.split(input):	# split (and remove) URLs
        s.lower()
        for m in token_re.finditer(s):
            output += " " + stemmer.stem(m.group())
    return output
# ---------------------------

def vectorizer_preprocessor(input):
    '''
    Cleanse documents (strings) before they are passed to a vectorizer
       tokenizer.
    Currently: lower case everything, remove URLs 
    To use: vectorizer = CountVectorizer(preprocessor=vectorizer_preprocessor)
    '''
    output = ''

    for s in urls_re.split(input):
        output += ' ' + s.lower() 
    return output
# ---------------------------

stemmingNotes= \
'''
### Stemming is weird
* First, note that stemming typically doesn't improve models very much, but
    if you want to continue.... here is what I've learned
* Stemming is not built into sklearn. you must combine w/ nltk yourself.
    * import nltk.stem.snowball as nltk,  nltk.EnglishStemmer()
        (same for lemmatization - haven't tried this directly)
* OPTIONS:
    * Preprocess the data files beforehand (outside of Vectorizer)
        * leads to fastest tuning runs as you don't have to keep stemming
            on each run.
        * cannot tune by comparing stemming to non-stemming via GridSearchCV
    * Create StemmingVectorizer subclass
        * see https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
        * In subclass approach, you cannot easily test stemming vs. non-stemming
            as you tune (without swapping vectorizer classes around)
        * Things to know:
            * the preprocessor() takes a doc (string),returns a modified doc.
                * the default preprocessor() does .lower() if lowercase=True
            * the tokenizer() takes a doc, returns list of tokens. Includes:
                * reg-ex tokenizing, n-gramming, stopword processing,
                    min_df, max_df processing
            * Not sure where unicode, accent handling happens
            * the analyzer() function runs both preprocessor() and then
                tokenizer(). If you want to do something in between of after
                these, customize this.
        * override build_analyzer() (and ultimately the analyzer() function) by
            taking output of the super.build_analyzer() and stemming each token
            that comes out.
            * only stems the last word in each n-gram
        * OR override build_preprocessor() (and ultimately preprocessor()) by
            taking output of the super.build_preprocessor(), tokenizing words,
            and stemming each word, putting words back into a string to be
            passed to the tokenizer.
            * you don't HAVE to combine stemming w/ other preprocessing
            * redundant tokenizing, 2 passes over the raw document (if 
                the default preprocessor() does anything)
        * OR override build_preprocessor() by providing your own preprocessor()
            function.
            * you must combine stemming w/ any other preprocessing you're doing 
            * remember to .lower() if desired
            * likely faster than making 2 passes
        * OR override build_tokenizer() (and ultimately tokenizer())
            * you'd have to figure out where to stick stemming into the process
            * seems painful as the tokenizer does a lot (I haven't tried this) 
    * Pass your own preprocessor() function at Vectorizer instantiation.
            * equivalent to the last option above
            * BUT you can include/exclude this preprocessor() option to test
                stemming vs. non-stemming as you tune
            * you must combine stemming w/ any other preprocessing you're doing 
            * remember to .lower() if desired
            * likely faster than making 2 passes
    * Pass your own analyzer() function at Vectorizer instantiation.
            * same issues as above
    * Pass your own tokenizer() function at Vectorizer instantiation.
            * seems painful as the tokenizer does a lot (I haven't tried this)
    * STEMMING AND STOPWORDS. Note the stopword list is NOT
        stemmed, e.g.,  "become", "becomes", "becoming" are stopwords.
        But in the input doc, these get stemmed to "becom" which is not removed
        during stopword processing. Sigh. So you might want to enhance the
        stopword list by stemming its elements. Fortunately words removed by 
        min_df and max_df have already been stemmed (if you are stemming).
'''
if __name__ == "__main__":
    # ad hoc test code
    if True:    # predictionType() testing
        print("should be FN: " + predictionType(1, 0, positiveClass=1))	# FN
        print("should be FP: " + predictionType(1, 0, positiveClass=0))	# FP
        print('')
        print("should be FP: " + predictionType(0, 1, positiveClass=1))	# FP
        print("should be FN: " + predictionType(0, 1, positiveClass=0))	# FN
        print('')
        print("should be TN: " + predictionType(0, 0, positiveClass=1))	# TN
        print("should be TP: " + predictionType(0, 0, positiveClass=0))	# TP
        print('')
        print("should be TP: " + predictionType(1, 1, positiveClass=1))	# TP
        print("should be TN: " + predictionType(1, 1, positiveClass=0))	# TN

    if True:	# getFalsePosNeg() testing
        fpos, fneg = getFalsePosNeg([0,1,0,1],
                                    [0,0,1,1],
                                    ['s1', 's2 FN', 's3 FP', 's4'],
                                    positiveClass=1,
                                    )
        print('false Positives: ', fpos)
        print('false Negatives: ', fneg)
        fpos, fneg = getFalsePosNeg([0,1,0,1],
                                    [0,0,1,1],
                                    ['s1', 's2 FP', 's3 FN', 's4'],
                                    positiveClass=0,
                                    )
        print('false Positives: ', fpos)
        print('false Negatives: ', fneg)
    if True:  # param grid tests
        simpleGrid = {'key0' : ['a0'], 'key1': ['b0'] }
        multiGrid = {'key0' : ['a0'], 'key1': ['b1','b2'], 'key2': ['c2','c3']}
        dictGrid = [ simpleGrid, multiGrid ]
        print("should be True: %s" % isOneCombination(simpleGrid)) 
        print("should be False: %s" % isOneCombination(multiGrid)) 
        print("should be False: %s" % isOneCombination(dictGrid)) 
        print(str(convertParamGrid(simpleGrid)))
        print(str(convertParamGrid(multiGrid)))
