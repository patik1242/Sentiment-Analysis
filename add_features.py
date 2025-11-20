import pandas as pd
import numpy as np


from dictionaries_and_extracting_features import extract_features

def add_features(clean_training):
    features_series = clean_training["text"].apply(extract_features)
    features_df = pd.DataFrame(features_series.tolist())

    clean_training = pd.concat([clean_training, features_df], axis=1)
    clean_training = clean_training.dropna(subset=[
        "weak_pos", "strong_pos", "weak_neg", "strong_neg",
        "is_positive_dominant", "is_negative_dominant",
        "word_count", "polarity", "has_link", "starts_with_RT",
        "digit_count", "has_hashtag"
    ])

    clean_training["sum_pos"] = clean_training["weak_pos"] + clean_training["strong_pos"]
    clean_training["sum_neg"] = clean_training["weak_neg"] + clean_training["strong_neg"]

    print("Tweety BEZ żadnych pozytywnych słów:", (clean_training["sum_pos"] == 0).mean())
    print("Tweety BEZ żadnych negatywnych słów:", (clean_training["sum_neg"] == 0).mean())

    clean_training["full_text"] = clean_training["text"].fillna("") + " " + clean_training["topic"].fillna("")

    return clean_training
