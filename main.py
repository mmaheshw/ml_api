"""
Script for FastAPI instance and model inference
author: Manjai Maheshwari
Date: July 2023
"""


# Put the code for your API here.
#!/home/manjari/miniconda3/envs/myenv/bin/python
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from ml.model import load_model, inference
from ml.data import process_data
import os
import pandas as pd
from fastapi import Depends
import pickle


MODEL_PATH = "./model/"
class InputData(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
                        "example": {
                                    'age':50,
                                    'workclass':"Private", 
                                    'fnlgt':234721,
                                    'education':"Doctorate",
                                    'education_num':16,
                                    'marital_status':"Separated",
                                    'occupation':"Exec-managerial",
                                    'relationship':"Not-in-family",
                                    'race':"Black",
                                    'sex':"Female",
                                    'capital_gain':0,
                                    'capital_loss':0,
                                    'hours_per_week':50,
                                    'native_country':"United-States"
                                    }
                        }


# Instantiate the app.
app = FastAPI()

# Load model artifacts on startup of the application to reduce latency.
# Use the "startup" event to load the model and set the lifespan context manager.
@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model_path = os.path.join(MODEL_PATH, 'model.pkl')
    encoder_path = os.path.join(MODEL_PATH, 'encoder.pkl')
    labeler_path = os.path.join(MODEL_PATH, 'labeler.pkl')
    model, encoder, lb = load_model(model_path, encoder_path, labeler_path)
    print(f"Encoder in startup_event: {encoder}")

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return "The API is working!"

def load_encoder():
    encoder_path = os.path.join(MODEL_PATH, 'encoder.pkl')
    encoder = pickle.load(open(encoder_path, 'rb'))
    return encoder

def load_lb():
    lb_path = os.path.join(MODEL_PATH, 'labeler.pkl')
    lb = pickle.load(open(lb_path, 'rb'))
    return lb

def load_model():
    model_path = os.path.join(MODEL_PATH, 'model.pkl')
    model = pickle.load(open(model_path, 'rb'))
    return model

# This allows sending of data (our InferenceSample) via POST to the API.
@app.post("/inference/")
async def inference(inference: InputData, model = Depends(load_model), encoder = Depends(load_encoder), lb = Depends(load_lb) ):
    print(f"Encoder in inference: {encoder}")
    print(inference)
    data = {  'age': inference.age,
                'workclass': inference.workclass, 
                'fnlgt': inference.fnlgt,
                'education': inference.education,
                'education_num': inference.education_num,
                'marital_status': inference.marital_status,
                'occupation': inference.occupation,
                'relationship': inference.relationship,
                'race': inference.race,
                'sex': inference.sex,
                'capital_gain': inference.capital_gain,
                'capital_loss': inference.capital_loss,
                'hours_per_week': inference.hours_per_week,
                'native_country': inference.native_country,
                }

    # prepare the sample for inference as a dataframe
    sample = pd.DataFrame(data, index=[0])
    print(sample)

    # apply transformation to sample data
    cat_features = [
                    "workclass",
                    "education",
                    "marital_status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native_country",
                    ]

    # model_path  = os.path.join(MODEL_PATH,'model.pkl')
    # encoder_path = os.path.join(MODEL_PATH,'encoder.pkl')
    # labeler_path = os.path.join(MODEL_PATH,'labeler.pkl')
    # model, encoder, lb = load_model(model_path, encoder_path, labeler_path)
 
    sample,_,_,_ = process_data(
                                sample, 
                                categorical_features=cat_features, 
                                training=False, 
                                encoder=encoder, 
                                lb=lb
                                )

                         
    #prediction = inference(model,sample)
    prediction = model.predict(sample)

    # convert prediction to label and add to data output
    print(prediction)
    if prediction[0]>0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K'
    
    data['prediction'] = prediction

    return data