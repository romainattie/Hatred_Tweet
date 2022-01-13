from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin


class FirstColumnSelector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.iloc[:, 0]


def get_pipeline(method='CountVectorizer', model_name='LogisticRegression'):

   # Transformer choice
    if method == 'CountVectorizer':
        vectorizer = CountVectorizer()
    elif method == 'TfidfVectorizer':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError('No good method')


    # Model choice
    if model_name == 'LogisticRegression':
        model = LogisticRegression()
    else:
        raise ValueError('No good model_name choice')

    pipe = Pipeline([
        ('resample', RandomOverSampler()),
        ('adaptator', FirstColumnSelector()),
        ('transformers', vectorizer),
        ('Model', model)
        ])

    return pipe




# tester avec bigram pour la tokenization
