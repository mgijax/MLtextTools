import sys
import textTuningLib as tl
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import SGDClassifier
#-----------------------
args = tl.args
randomSeeds = tl.getRandomSeeds( { 	# None means generate a random seed
		'randForSplit'      : args.randForSplit,
		'randForClassifier' : args.randForClassifier,
		} )
pipeline = Pipeline( [
('vectorizer', CountVectorizer(
#('vectorizer', TfidfVectorizer(
		strip_accents=None,	# if done in preprocessing
		decode_error='strict',	# if handled in preproc
		lowercase=False,	# if done in preprocessing
		stop_words='english',
		binary=True,
		#token_pattern=r'\b([a-z_]\w+)\b', Use default for now
		),),
#('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
#('scaler'    , MaxAbsScaler(copy=True)),
('classifier', SGDClassifier(verbose=0, class_weight='balanced',
		random_state=randomSeeds['randForClassifier']) ),
] )
parameters={'vectorizer__ngram_range':[(1,2)],
	'vectorizer__min_df':[10],
	'vectorizer__max_df':[.75],
	'classifier__alpha':[1],
	'classifier__learning_rate':['invscaling'],
	'classifier__eta0':[ .01],
	'classifier__loss':[ 'hinge' ],
	'classifier__penalty':['l2'],
	}
p = tl.TextPipelineTuningHelper( pipeline, parameters,
		    randomSeeds=randomSeeds,         ).fit()
print p.getReports()
