import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.split()
    text = " ".join(text)
    text = text.strip()
    return text


#Wyświetlanie wierszy datasetu treningowego
print(training_dataset.head())

#Ile jest próbek do trenowania?
#print(training_dataset.shape)

#Ile jest próbek do walidacji?
#print(validation_dataset.shape)

#wyświetlamy kolumny 
print(training_dataset.columns)

#Sprawdzamy ile jest duplikatów
print("Before: Duplicated: ", training_dataset.duplicated(subset="text").sum())

#Usuwamy duplikaty
clean_training = training_dataset.drop_duplicates(subset="text", keep="last")

#Sprawdzamy ile jest duplikatów
print("After: Duplicated: ", clean_training.duplicated(subset="text").sum())

print("Is NaN: ", clean_training["text"].isna().sum())
print("Is empty: ", (clean_training["text"]=="").sum())

clean_training = clean_training.dropna(subset="text")

clean_training = clean_training[clean_training["text"]!= ""]

clean_training["text"] = clean_training["text"].apply(preprocess)