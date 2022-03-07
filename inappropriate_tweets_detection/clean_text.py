'''
Differences between preproc clean_data() & clean_data_balancing():
- Balancing added in function clean_data()
'''

import pandas as pd
# Balancing
from imblearn.over_sampling import RandomOverSampler


def get_data():
    df = pd.read_csv('raw_data/MeTooHate.csv')
    return df


# On ne garde que le message et la catégorie (0 = non harcèlement, 1 = harcèlement)
def clean_text(df):
    df = df[['text', 'category']]  # Columns selection
    df = df.drop_duplicates()  # No duplicates
    df = df.dropna()  # Drop missing values de text
    # No need to manage outliers for now
    X = df.drop('category', axis=1)
    y = df['category']
    return X, y


def clean_text_2(df):
    df = df[[
        'text', 'favorite_count', 'retweet_count', 'followers_count',
        'friends_count', 'statuses_count', 'category'
    ]]  # Columns selection
    df = df.drop_duplicates(subset='text')  # No duplicates
    df = df.dropna()  # Drop missing values de text
    # No need to manage outliers for now
    X = df.drop('category', axis=1)
    y = df['category']
    return X, y


# On ne garde que le message et la catégorie (0 = non harcèlement, 1 = harcèlement)
def clean_text_balancing(df):
    df = df[['text', 'category']]  # Columns selection
    df = df.drop_duplicates()  # No duplicates
    df = df.dropna()  # Drop missing values de text
    # No need to manage outliers for now
    ros = RandomOverSampler()
    X = df.drop('category', axis=1)
    y = df['category']
    X_ros, y_ros = ros.fit_resample(X, y)
    return X_ros, y_ros
