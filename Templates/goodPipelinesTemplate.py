
# Definitions of good Pipelines to compare.
# Define a variable "pipelines" that is the list of Pipelines.

import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
#-----------------------

global pipelines
pipelines = [ \
    Pipeline( [		# pipeline 0
        ('vectorizer', TfidfVectorizer( analyzer='word',
                        strip_accents=None,     # done in preprocessing
                        decode_error='strict',  # handled in preproc
                        lowercase=False,        # done in preprocessing
                        stop_words='english',
                        ngram_range=(1,2),
                        min_df=0.1,  max_df=.7,
                        )),
        ('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
        #('scaler'    , MaxAbsScaler(copy=True)),
        ('classifier', SGDClassifier(verbose=0,
                        loss='modified_huber',
                        penalty='l2',
                        alpha=10,
                        learning_rate='optimal',
                        class_weight='balanced',
                        eta0 = .01,
                        )),
        ] ),
    Pipeline( [		# pipeline 1
        ('vectorizer', TfidfVectorizer( analyzer='word',
                        strip_accents=None,     # done in preprocessing
                        decode_error='strict',  # handled in preproc
                        lowercase=False,        # done in preprocessing
                        stop_words='english',
                        ngram_range=(1,2),
                        min_df=0.1,  max_df=.7,
                        )),
        ('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
        #('scaler'    , MaxAbsScaler(copy=True)),
        ('classifier', SGDClassifier(verbose=0,
                        loss='log',
                        penalty='l2',
                        alpha=5,
                        learning_rate='optimal',
                        class_weight='balanced',
                        eta0 = .01,
                        )),
        ] ),
] # end Pipelines[]
#-----------------------
