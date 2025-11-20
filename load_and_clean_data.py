import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from dictionaries_and_extracting_features import strong_negative_words, strong_positive_words, weak_negative_words, weak_positive_words, weak_irrelevant_words, weak_neutral_words, strong_irrelevant_words, strong_neutral_words
def load_and_clean_data():
    #Poprawa nagłówków kolumn, zmieniło kolumny na 0,1,2,3
    training_dataset = pd.read_csv("twitter_training.csv", header=None)
    validation_dataset = pd.read_csv("twitter_validation.csv", header=None)

    #Nadajemy nazwy kolumnom
    training_dataset.columns = ["id", "topic", "sentiment", "text"]
    validation_dataset.columns = ["id", "topic", "sentiment", "text"]

    #Usuwamy duplikaty i NaN
    clean_training = training_dataset.drop_duplicates(subset=["text"], keep="last")
    clean_training = clean_training.dropna(subset=["sentiment", "text", "topic"])
    clean_training = clean_training[clean_training["text"] != ""]
    clean_training = clean_training.reset_index(drop=True)

    # preprocess tekstu
    clean_training["text"] = clean_training["text"].apply(preprocess)
    clean_training = clean_training[clean_training["text"] != ""]
    clean_training = clean_training.reset_index(drop=True)

    return clean_training
