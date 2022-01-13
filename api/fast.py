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

from Hatred_Tweet.predict import harassment_predict



@app.get("/")
def test():
    return {'test': 'Bonjour bro'}

@app.get("/predict")
def predict(text):
    return {
        'Result': harassment_predict(text)
    }
