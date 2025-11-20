import pandas as pd
from preprocess import preprocess

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
