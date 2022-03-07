from inappropriate_tweets_detection.pipeline_deep import get_pipeline
from inappropriate_tweets_detection.preproc import data_preproc
import joblib
import pandas as pd

pipe = joblib.load('../saved_pipelines/logreg_pipeline_text.joblib')


def harassment_predict(text):

    prep_text = data_preproc(text)

    result = pipe.predict_proba(pd.DataFrame([[prep_text]], columns=['text']))[0,1]

    return result
