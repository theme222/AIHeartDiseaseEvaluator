import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

from TechnicalToolsV3 import log, time_convert
from time import time

data_filename = "heart.csv"


def data_config(data, random_state):
    # Note to self X (Input) is features y is target (Output)
    X = data.drop(columns=["target"])
    y = data["target"]

    # Split data 80 : 20 training testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    '''
    test_size = 0.2 (Ratio of test data)
    random_state = 42 (Seed of randomizer)
    stratify = y (Ensures ratio of target stays consistent between training and testing)
    '''

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(
        X_test)  # Using transform uses the mean and the deviation calculated from scaler.fit_transform()

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    return LogisticRegression().fit(X_train, y_train)


def export_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def model_analysis(filename, data):

    with open(filename, 'rb') as file:
        model = pickle.load(file)

    visualization_df = pd.DataFrame(model.coef_.T, index=data.drop(columns=['target']).columns, columns=["Coefficient"])
    log(var=visualization_df, mode='result', include_var_formatting=False)


def workflow(random_state):
    log(f"Importing data from {data_filename}", mode="info")
    data = pd.read_csv(data_filename)
    log("Data headers", var=data.head(), include_var_formatting=False)

    log("Using random state :", var=random_state)
    X_train, X_test, y_train, y_test = data_config(data, random_state)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    log("Total accuracy", mode='result', var=accuracy)

    log("Classification report :", var=classification_report(y_test, y_pred), mode="result",
        include_var_formatting=False)
    log("Confusion matrix :", var=confusion_matrix(y_test, y_pred), mode="result", include_var_formatting=False)

    export_model(model, "Models/LogisticalRegression.pkl")
    model_analysis("Models/LogisticalRegression.pkl", data)


if __name__ == '__main__':
    start_time = time()
    workflow(int(start_time))
    log("Total time taken :", var=time_convert(time() - start_time), mode="success")
