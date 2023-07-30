

import logging
import pandas as pd
import os
from ml.tests.test_conf import data, cat_features, features, model, dataset_split, encoder_lb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_data_shape(data):
    """
    Test shape of the data
    """
    # Check the df shape
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing dataset: The file doesn't appear to have rows and columns")
        raise err

def test_data_features(data, features):
    """
    Test features of the data
    """
    try:

        assert set(data.columns) == set(features)
    
    except AssertionError as err:
        logging.error(
        "Testing dataset: Features are missing in the data columns")
        raise err


def test_model(model, dataset_split):
    """
    Check if model is able to make predictions
    """
    try:
        X_train, y_train, X_test, y_test = dataset_split
        preds = model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    except Exception as err:
        logging.error(
        "Testing model: Saved model is not able to make new predictions")
        raise err

from fastapi.testclient import TestClient
#from fastapi import HTTPException
import json
import logging
from main import app

client = TestClient(app)

def test_get():
    """
    Test welcome message for get at root
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "The API is working!"



def test_post_50k(data, encoder_lb):
    encoder, _ = encoder_lb

    """
    # take a sample from the data with a high salary
    sample=data[data['salary']=='<=50K'].iloc[500].to_dict()
    # remove target
    _ = sample.pop('salary')
    """
   
    sample = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 287372,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Husband",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
        }
    #data = json.dumps(sample)
    #print(data)

    r = client.post("/inference/", json=sample )
    

     # test response and output
    assert r.status_code == 200
    assert r.json()["prediction"]== '>50K'

def test_post_0k(data, encoder_lb):
    encoder, _ = encoder_lb

    """
    # take a sample from the data with a high salary
    sample=data[data['salary']=='>50K'].iloc[500].to_dict()
    # remove target
    _ = sample.pop('salary')

    data = json.dumps(sample)
    """
    sample =  {  'age':24,
                'workclass':"Private", 
                'fnlgt':505119,
                'education':"Some-college",
                'education_num':14,
                'marital_status':"Married-civ-spouse",
                'occupation':"Sales",
                'relationship':"Husband",
                'race':"Black",
                'sex':"Male",
                'capital_gain':0,
                'capital_loss':0,
                'hours_per_week':0,
                'native_country':"Cuba"
            }

    r = client.post("/inference/", json=sample )

     # test response and output
    assert r.status_code == 200
    assert r.json()["prediction"][0] == '<=50K'