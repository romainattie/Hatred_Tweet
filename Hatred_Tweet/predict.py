from Hatred_Tweet.pipeline_deep import get_pipeline
from Hatred_Tweet.preproc import data_preproc
import joblib
import pandas as pd

pipe = joblib.load('../saved_pipelines/logreg_pipeline_text.joblib')


def harassment_predict(text):

    prep_text = data_preproc(text)

    result = pipe.predict(pd.DataFrame([[prep_text]], columns=['text']))

    if result == 0:
        return 'No Hatred'
    else:
        return 'Hatred'
