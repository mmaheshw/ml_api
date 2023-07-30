import pytest
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os

from ml.data import process_data

@pytest.fixture(scope='session')
def encoder():
    encoder_path = './model/encoder.pkl'
    encoder = pickle.load(open(encoder_path, 'rb'))
    return encoder

@pytest.fixture(scope='session')
def data():
    print(os.getcwd())

    data_path = "./data/census.csv"
    df = pd.read_csv(data_path)
    print(df)

    return df

@pytest.fixture(scope="session")
def cat_features():
    """
    Fixture - will return the categorical features as argument
    """
    cat_features = [    "workclass",
                        "education",
                        "marital_status",
                        "occupation",
                        "relationship",
                        "race",
                        "sex",
                        "native_country"]
    return cat_features

@pytest.fixture(scope="session")
def features():
    """
    Fixture - will return the categorical features as argument
    """
    features = ['age', 'workclass', 'fnlgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
       'salary']
    return features

@pytest.fixture(scope='session')
def model():

    model_path = './model/model.pkl'
    model = pickle.load(open(model_path, 'rb'))

    return model


@pytest.fixture(scope="session")
def dataset_split(data, cat_features):
    """
    Fixture - returns cleaned train dataset to be used for model testing
    """
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=cat_features,
                                            label="salary",
                                            training=True
                                        )
    X_test, y_test, encoder, lb = process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="session")
def encoder_lb(data, cat_features):
    """
    Fixture - returns encoder and labeler
    """
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=cat_features,
                                            label="salary",
                                            training=True
                                        )
    X_test, y_test, encoder, lb = process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
    return encoder, lb

