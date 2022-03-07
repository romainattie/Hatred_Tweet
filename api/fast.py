from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from inappropriate_tweets_detection.predict import harassment_predict

@app.get("/")
def test():
    return {'test': 'Bonjour bro'}

@app.get("/predict")
def predict(text):
    return {
        'Result': harassment_predict(text)
    }

@app.get("/predict_deep")
def predict_deep(text):
    return {'Result': 'In development'}
