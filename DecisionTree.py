import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle

from TechnicalToolsV3 import log, time_convert
from time import time

from LogisticalRegression import data_config

data_filename = "heart.csv"


def train_model(X_train, y_train, random_state):
    return DecisionTreeClassifier(random_state=random_state).fit(X_train, y_train)


def export_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def model_analysis(filename, data):
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=data.drop(columns=["target"]).columns,
              class_names=["Normal", "Heart Disease"])
    plt.show()


def workflow(random_state):
    log(f"Importing data from {data_filename}", mode="info")
    data = pd.read_csv(data_filename)
    log("Data headers", var=data.head(), include_var_formatting=False)

    log("Using random state :", var=random_state)
    X_train, X_test, y_train, y_test = data_config(data, random_state)
    model = train_model(X_train, y_train, random_state)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred, )
    log("Total accuracy", mode='result', var=accuracy)

    log("Classification report :", var=classification_report(y_test, y_pred), mode="result",
        include_var_formatting=False)
    log("Confusion matrix :", var=confusion_matrix(y_test, y_pred), mode="result", include_var_formatting=False)

    export_model(model, "Models/DecisionTree.pkl")
    model_analysis("Models/DecisionTree.pkl", data)


if __name__ == '__main__':
    start_time = time()
    workflow(int(start_time))
    log("Total time taken :", var=time_convert(time() - start_time), mode="success")
