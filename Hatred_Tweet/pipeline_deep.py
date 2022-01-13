# Pipeline
from imblearn.pipeline import Pipeline
# Balancing (resampling)
from imblearn.over_sampling import RandomOverSampler, SMOTE
# Adaptator
from sklearn.base import BaseEstimator, TransformerMixin

# Numerical scaler
from sklearn.preprocessing import RobustScaler
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

from sklearn.compose import ColumnTransformer
from imblearn.pipeline import make_pipeline
from sklearn.utils import resample
from xgboost import XGBRegressor


class FirstColumnSelector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.iloc[:, 0]


def get_pipeline(model_name='LogisticRegression',
                 with_num_features=True,
                 balancing='SMOTE'):

    if balancing == 'RandomOverSampler':
        balancing = RandomOverSampler()
    elif balancing == 'SMOTE':
        balancing = SMOTE()
    else:
        return ValueError('No good balancing method')

    adaptator = FirstColumnSelector()

    num_transformer = Pipeline([('num_transformer', RobustScaler())])

    num_list = [('num_trans', num_transformer, [
        'favorite_count', 'retweet_count', 'followers_count', 'friends_count',
        'statuses_count'
    ])]

    preprocess = ColumnTransformer(num_list)

    # Model choice
    if model_name == 'LogisticRegression':
        model = LogisticRegression(max_iter=5000, verbose=1)
    elif model_name == 'BernoulliNB':
        model = BernoulliNB()
    elif model_name == 'XGBRegressor':
        model = XGBRegressor(max_depth=10, n_estimators=300, learning_rate=0.1)
    else:
        raise ValueError('No good model_name choice')

    if with_num_features:
        pipe = make_pipeline(balancing, preprocess, model)
    else:
        pipe = make_pipeline(balancing, model)

    return pipe
