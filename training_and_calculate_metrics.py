import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    #trenowanie modelu
    model.fit(X_train, y_train)

    #predykcje
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #obliczanie metryk 
    train_metrics = calculate_metrics(y_train, y_train_pred, "train")
    test_metrics = calculate_metrics(y_test, y_test_pred, "test")

    return train_metrics, test_metrics

def calculate_metrics(y_true, y_pred, split):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\nMetryki dla RidgeClassifier ({split}): \n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    #macierz pomyłek
    if split =="test":
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
        plt.title(f"Macierz pomyłek - RidgeClassifier ({split})")
        plt.show()

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1':f1}
