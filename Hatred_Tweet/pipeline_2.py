# Pipeline
from imblearn.pipeline import Pipeline
# Balancing (resampling)
from imblearn.over_sampling import RandomOverSampler
# Adaptator
from sklearn.base import BaseEstimator, TransformerMixin
# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Numerical scaler
from sklearn.preprocessing import RobustScaler
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB


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
    elif model_name == 'BernoulliNB':
        model = BernoulliNB()
    else:
        raise ValueError('No good model_name choice')

    pipe = Pipeline([('resample', RandomOverSampler()),
                     ('adaptator', FirstColumnSelector()),
                     ('transformers', vectorizer),
                     ('numerical_scaler', RobustScaler(with_centering=False)),
                     ('Model', model)])

    return pipe


# tester avec bigram(ngram_range=(2,2)) pour la tokenization(transformer)
