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

from sklearn.compose import ColumnTransformer
from imblearn.pipeline import make_pipeline
from sklearn.utils import resample


class FirstColumnSelector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.iloc[:, 0]




def get_pipeline(method='CountVectorizer', model_name='LogisticRegression',
                 with_num_features=True):

    # Transformer choice
    if method == 'CountVectorizer':
        vectorizer = CountVectorizer()
    elif method == 'TfidfVectorizer':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError('No good method')

    balancing = RandomOverSampler()

    text_transformer = Pipeline([
        ('adaptator', FirstColumnSelector()),
        ('transformer', vectorizer)
    ])


    num_transformer = Pipeline([('num_transformer', RobustScaler())])

    if with_num_features:
        num_list = [('text_trans', text_transformer, ['text']),
                    ('num_trans', num_transformer, [
                        'favorite_count', 'retweet_count', 'followers_count',
                        'friends_count', 'statuses_count'
                    ])]
    else:
        num_list=[
                    ('text_trans', text_transformer, ['text'])]

    preprocess = ColumnTransformer(num_list)

    # Model choice
    if model_name == 'LogisticRegression':
        model = LogisticRegression(max_iter=5000, verbose=1)
    elif model_name == 'BernoulliNB':
        model = BernoulliNB()
    else:
        raise ValueError('No good model_name choice')

    pipe = make_pipeline(balancing, preprocess, model)

    return pipe
