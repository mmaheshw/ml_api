from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle, os
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, grid_search = False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    ## TAKES TOO LONG
    if grid_search:
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear','rbf']} 

        grid = GridSearchCV(SVC(random_state=107, verbose=True),
                    param_grid, 
                    scoring='accuracy',
                    cv=10,
                    refit=True,
                    n_jobs=-1, 
                    verbose=3)

        grid.fit(X_train, y_train)
        print(f"Model's best score: {grid.best_score_}")
        print(f"Model's best params: {grid.best_params_}")

        best_model = grid.best_estimator_
        best_model.fit(X_train, y_train)

        return best_model
    else:

        #model = SVC(C = 0.1, gamma = 0.01, kernel ='rbf', random_state=1, verbose=True) 
        model = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        model.fit(X_train, y_train)
        return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path, encoder=None, labeler=None):
    """
    Save the given model object to a specified file path using pickle.

    Args:
        model (object): The machine learning model object to be saved.
        path (str): The file path to save the model to.
        encoder (object, optional): The encoder object used to transform input data, if any.
        labeler (object, optional): The label encoder object used to transform target data, if any.
    """  

    pickle.dump(model, open(os.path.join(path,'model.pkl'), 'wb'))
    if encoder:
        pickle.dump(encoder, open(os.path.join(path,'encoder.pkl'), 'wb'))

    if labeler:
        pickle.dump(labeler, open(os.path.join(path,'labeler.pkl'), 'wb'))

def load_model(model_path, encoder_path, labeler_path):
    """
    Load a previously saved machine learning model object, encoder object, and labeler object.

    Args:
        model_path (str): The file path to load the saved model from.
        encoder_path (str): The file path to load the saved encoder object from.
        labeler_path (str): The file path to load the saved labeler object from.

    Returns:
        tuple: A tuple containing the loaded model object, encoder object, and labeler object.
    """
    model = pickle.load(open(model_path, 'rb'))
    
    encoder = pickle.load(open(encoder_path, 'rb'))

    lb = pickle.load(open(labeler_path, 'rb'))

    return model, encoder, lb


def slice_data(data, fixed_var, target_data, preds):
    """
    Computes precision, recall, and F-beta score for a given target variable and model predictions, sliced by a 
    specified fixed variable.

    Args:
        data (pandas.DataFrame): The input dataset containing the fixed and target variables.
        fixed_var (str): The name of the fixed variable to slice the data by.
        target_data (numpy.array): The true target values.
        preds (numpy.array): The predicted target values.

    Returns:
        pandas.DataFrame: A dataframe containing the fixed variable, class, precision, recall, and F-beta score for
        each slice of the data.
    """
     
    df_temp = pd.DataFrame(columns=['fixed_var','class','precision', 'recall', 'fbeta'])
 
    print(fixed_var)
    #print(data[fixed_var])
    #print(target_data[fixed_var])
   
    for cls in data[fixed_var].unique():

        slice_mask = data[fixed_var]==cls

        target_data_slice = target_data[slice_mask]
        preds_slice = preds[slice_mask]
        
        precision, recall, fbeta = compute_model_metrics(target_data_slice, preds_slice)
        print(precision, recall, fbeta)
        
        new_row = pd.Series({'fixed_var': fixed_var, 'class': cls, 'precision':precision,'recall':recall,'fbeta':fbeta})

        df_temp = pd.concat([df_temp, new_row.to_frame().T], ignore_index=True)

    return df_temp