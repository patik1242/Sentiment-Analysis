import pandas as pd
from preprocess import preprocess

def load_and_clean_data():
    # Wczytanie datasetów bez nagłówków (oryginalny format Kaggle'a)
    training_dataset = pd.read_csv("twitter_training.csv", header=None)

    # Nadanie nazw kolumnom – ułatwia dalsze przetwarzanie i selekcję
    training_dataset.columns = ["id", "topic", "sentiment", "text"]

    # Usunięcie duplikatów na podstawie samego tekstu – typowy krok w projektach NLP,
    # bo powtarzające się tweety zaburzają rozkład danych
    clean_training = training_dataset.drop_duplicates(subset=["text"], keep="last")

    # Usunięcie wierszy z brakującymi wartościami w kluczowych kolumnach
    clean_training = clean_training.dropna(subset=["sentiment", "text", "topic"])

    # Wykluczenie pustych tekstów (po preprocessingu niektóre tweety mogą być puste)
    clean_training = clean_training[clean_training["text"] != ""]
    clean_training = clean_training.reset_index(drop=True)

    # Wstępny preprocessing tekstu (tokenizacja, czyszczenie znaków, normalizacja)
    clean_training["text"] = clean_training["text"].apply(preprocess)

    # Usunięcie tweetów, które stały się puste po czyszczeniu – brak treści = brak sygnału
    clean_training = clean_training[clean_training["text"] != ""]
    clean_training = clean_training.reset_index(drop=True)

    # Zwrócenie czystego datasetu gotowego do feature engineering
    return clean_training
