"""
Script to train machine learning model.
author: Manjari Maheshwari
Date: July 2023
"""

# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import *
import logging, os

import warnings
warnings.filterwarnings("ignore")

# Initialize logging
logging.basicConfig(filename='info_logs.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')


def go():
    """Script for the project. It trains a ml model by applying the functions defined in "./model.py"
    """
    logging.info(12*'*'+'START OF MODEL TRAINING' + 12*'*')
    
    # Add code to load in the data.
    data_file = "./data/census.csv"
    data = pd.read_csv(data_file)
    logging.info('DATA LOADED')

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

    # sometimes X_test.shape is not what it should, so must be verified
    X_test_shape = 0
    while X_test_shape!=108:

        # Optional enhancement, use K-fold cross validation instead of a train-test split.
        train, test = train_test_split(data, test_size=0.20,random_state=107, stratify=data['salary'])
        logging.info(f"Data shape after split. Train: {train.shape} \t Test:{test.shape}")

        # Proces the TRAIN data with the process_data function.
        X_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True)

        # Proces the TEST data with the process_data function.
        X_test, y_test, encoder, lb = process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )

        logging.info(f"X_test data shape after transformation: {X_test.shape}")
        X_test_shape = X_test.shape[1]
        

    # Train and save a model.
    MODEL_PATH = './model'

    # if the model exists, load the model
    if os.path.isfile(os.path.join(MODEL_PATH,'model.pkl')):
        logging.info(f"A model already exists...")
        model_path  = os.path.join(MODEL_PATH,'model.pkl')
        encoder_path = os.path.join(MODEL_PATH,'encoder.pkl')
        labeler_path = os.path.join(MODEL_PATH,'labeler.pkl')
        model, encoder, lb = load_model(model_path, encoder_path, labeler_path)
        logging.info(f"model, encoder and labeler loaded")

    else:
        logging.info(f"A model does not exist... finding the best model...")
        model = train_model(X_train, y_train, grid_search=False)
        save_model(model, MODEL_PATH, encoder, lb)
        logging.info(f"Best model saved")
  
    # evaluate predictions
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logging.info("Model`s metrics:")
    logging.info(f"Precision: {precision}, Recall: {recall}, fbeta: {fbeta}")

    # compute models metrics on the slices of the data for categorical variables and save them in slice_output.txt
    SLICE_OUTPUT_PATH = "./slice_output.txt"

    with open(SLICE_OUTPUT_PATH, 'w') as file:
        logging.info("Computing slices...")
        for feature in cat_features:
            df_result = slice_data(test, feature, y_test, preds)
            df_string = df_result.to_string(header=False, index=False)
            file.write(df_string)
    
    logging.info(12*'*'+'FINISH OF MODEL TRAINING' + 12*'*')

if __name__ == '__main__':
    go()