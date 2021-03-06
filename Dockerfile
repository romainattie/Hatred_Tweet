FROM python:3.8.12-buster

COPY requirements_docker.txt /requirements_docker.txt
RUN pip install --upgrade pip
RUN pip install -r requirements_docker.txt

COPY saved_pipelines /saved_pipelines
COPY inappropriate_tweets_detection /inappropriate_tweets_detection
COPY api /api
# COPY /home/romain/code/romainattie/gcp/wagon-bootcamp-330815-e4e29e97a65e.json /credentials.json

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
