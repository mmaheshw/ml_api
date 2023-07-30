from fastapi.testclient import TestClient
from fastapi import HTTPException
import json
import logging
from main import app
import pickle
from ml_api.conftest import data, cat_features, features, model, dataset_split, encoder_lb, encoder

client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200, "response not successful"
    assert response.json() == "The API is working!"

def test_post_inference_one():
    # Load the encoder directly
    encoder_path = './model/encoder.pkl'
    encoder = pickle.load(open(encoder_path, 'rb'))

    # Load the label binarizer directly
    lb_path = './model/labeler.pkl'
    lb = pickle.load(open(lb_path, 'rb'))

    input_data = {'age': 76,
                 'workclass': 'Private',
                 'fnlgt': 124191,
                 'education': 'Masters',
                 'education_num': 14,
                 'marital_status': 'Married-civ-spouse',
                 'occupation': 'Exec-managerial',
                 'relationship': 'Husband',
                 'race': 'White',
                 'sex': 'Male',
                 'capital_gain': 0,
                 'capital_loss': 0,
                 'hours_per_week': 40,
                 'native_country': 'United-States'}
    expected_response = input_data.copy()
    expected_response['prediction'] = '>50K'  # Add this line
    response_post = client.post("/inference/", json=input_data)
    assert response_post.status_code == 200, "response not successful with {}".format(response_post.json())
    assert response_post.json() == expected_response

def test_post_inference_two():
    encoder_path = './model/encoder.pkl'
    encoder = pickle.load(open(encoder_path, 'rb'))

    # Load the label binarizer directly
    lb_path = './model/labeler.pkl'
    lb = pickle.load(open(lb_path, 'rb'))
    input_data = {'age': 22,
                 'workclass': 'Private',
                 'fnlgt': 201490,
                 'education': 'HS-grad',
                 'education_num': 9,
                 'marital_status': 'Never-married',
                 'occupation': 'Adm-clerical',
                 'relationship': 'Own-child',
                 'race': 'White',
                 'sex': 'Male',
                 'capital_gain': 0,
                 'capital_loss': 0,
                 'hours_per_week': 20,
                 'native_country': 'United-States'}
    expected_response = input_data.copy()
    expected_response['prediction'] = '<=50K'  # Add this line
    response_post = client.post("/inference/", json=input_data)
    assert response_post.status_code == 200, "response not successful with {}".format(response_post.json())
    assert response_post.json() == expected_response