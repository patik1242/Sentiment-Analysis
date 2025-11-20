import pandas as pd

from preprocess import preprocess
from dictionaries_and_extracting_features import extract_features


def add_features(clean_training):
    # Ekstrakcja cech z tekstu (custom feature engineering)
    features_series = clean_training["text"].apply(extract_features)
    features_df = pd.DataFrame(features_series.tolist())

    # Dołączenie wygenerowanych cech do oryginalnego dataframe
    clean_training = pd.concat([clean_training, features_df], axis=1)

    # Usuwanie ewentualnych wierszy z brakującymi wartościami cech
    # (bez tego model może rzucać błędy przy skalowaniu lub trenowaniu)
    clean_training = clean_training.dropna(subset=[
        "weak_pos", "strong_pos", "weak_neg", "strong_neg",
        "is_positive_dominant", "is_negative_dominant",
        "word_count", "polarity", "has_link", "starts_with_RT",
        "digit_count", "has_hashtag"
    ])

    # Zsumowane cechy pomocnicze: liczba pozytywnych i negatywnych słów
    clean_training["sum_pos"] = clean_training["weak_pos"] + clean_training["strong_pos"]
    clean_training["sum_neg"] = clean_training["weak_neg"] + clean_training["strong_neg"]

    # Krótka diagnostyka datasetu — informacja o tym,
    # ile tweetów nie zawiera żadnych słów pozytywnych / negatywnych
    print("Tweety BEZ żadnych pozytywnych słów:", (clean_training["sum_pos"] == 0).mean())
    print("Tweety BEZ żadnych negatywnych słów:", (clean_training["sum_neg"] == 0).mean())

    # Preprocessing kolumny 'topic' tak samo jak tekstu,
    # żeby TF-IDF działał na spójnie oczyszczonym tekście
    clean_training["topic"] = clean_training["topic"].apply(preprocess)

    # Łączenie tekstu z topic w jedną kolumnę wejściową dla TF-IDF
    clean_training["full_text"] = (
        clean_training["text"].fillna("") + " " +
        clean_training["topic"].fillna("")
    )

    return clean_training
