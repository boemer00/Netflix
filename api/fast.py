
# $DELETE_BEGIN
from datetime import datetime
import pytz

import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2


@app.get("/")
def index():
    return dict(movie_quote="Look mom, I'm roadkill. Ha Ha! -- The Mask (1994)")


@app.get("/predict")
def predict(year,                   # 2000
            runtime,                # 90 min
            rated,                  # R, PG-13
            country):               # USA


    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
        Year=[int(year)],
        Runtime=[runtime],
        Rated=[rated],
        Country=[country]))

    # ⚠️ TODO: get model from GCP

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(prediction=pred)
# $DELETE_END

if __name__ == "__main__":
    print(predict(2000, '90', 'R', 'USA'))