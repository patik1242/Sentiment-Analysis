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

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9#@\s]", "", text)
    text = text.split()
    text = " ".join(text)
    text = text.strip()
    return text

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


def extract_features(text):
    weak_pos=strong_pos=weak_neg=strong_neg = 0 
    positive_ratio= negative_ratio= 0
    is_positive_dominant= is_negative_dominant = 0
    has_link=starts_with_RT= digit_count = 0
    has_hashtag = 0 

    words = text.split()
    text_len = len(words) if len(words)>0 else 1
    for w in words:
        token = w.lstrip("#")
        if token in weak_positive_words:
            weak_pos+=1
        elif token in strong_positive_words:
            strong_pos+=1
        elif token in weak_negative_words:
            weak_neg+=1
        elif token in strong_negative_words:
            strong_neg+=1 
        elif w.startswith("#"):
            has_hashtag =1

    positive_ratio = (weak_pos+ strong_pos)/(text_len+1)
    negative_ratio = (weak_neg+strong_neg)/(text_len+1)


    if positive_ratio- negative_ratio>0.1:
        is_positive_dominant= 1

    if negative_ratio- positive_ratio>0.1:
        is_negative_dominant= 1
    
    if "http" in text.lower():
        has_link = 1
    
    clean = text.strip().upper()
    if clean.startswith("RT "):
        starts_with_RT = 1
    
    for char in text:
        if char.isdigit():
            digit_count+=1
    

    polarity = positive_ratio-negative_ratio
    return {
        "weak_pos": weak_pos, 
        "strong_pos": strong_pos,
        "weak_neg": weak_neg, 
        "strong_neg": strong_neg,
        "is_positive_dominant": is_positive_dominant,
        "is_negative_dominant": is_negative_dominant,
        "word_count": text_len,
        "polarity": polarity,
        "has_link": has_link, 
        "starts_with_RT": starts_with_RT,
        "digit_count": digit_count,
        "has_hashtag": has_hashtag
    }

#Poprawa nagłówków kolumn, zmieniło kolumny na 0,1,2,3
training_dataset = pd.read_csv("twitter_training.csv", header=None)
validation_dataset = pd.read_csv("twitter_validation.csv", header=None)

#Nadajemy nazwy kolumnom
training_dataset.columns = ["id", "topic", "sentiment", "text"]
validation_dataset.columns = ["id", "topic", "sentiment", "text"]


#Usuwamy duplikaty
clean_training = training_dataset.drop_duplicates(subset=["text"], keep="last")
clean_training = clean_training.dropna(subset=["sentiment", "text", "topic"])
clean_training = clean_training[clean_training["text"]!= ""]
clean_training = clean_training.reset_index(drop=True)


#Sprawdzamy ile jest duplikatów
print("After: Duplicated: ", clean_training.duplicated(subset=["text"]).sum())
print("Is NaN: ", clean_training["text"].isna().sum())
print("Is empty: ", (clean_training["text"]=="").sum())

clean_training["text"] = clean_training["text"].apply(preprocess)

clean_training = clean_training[clean_training["text"]!= ""]
clean_training = clean_training.reset_index(drop=True)
print("After preprocess - Is empty: ", (clean_training["text"]=="").sum())


features_series= clean_training["text"].apply(extract_features)
features_df = pd.DataFrame(features_series.tolist())

clean_training = pd.concat([clean_training, features_df], axis=1)
clean_training = clean_training.dropna(subset=["weak_pos", "strong_pos", "weak_neg", "strong_neg", 
                                               "is_positive_dominant", "is_negative_dominant", 
                                               "word_count", "polarity", "has_link", "starts_with_RT", "digit_count"
                                               ,"has_hashtag"])

print("isna",clean_training["sentiment"].isna().sum())
print("unique",clean_training["sentiment"].unique())
print("Final shape:", clean_training.shape)


clean_training["sum_pos"] = clean_training["weak_pos"] + clean_training["strong_pos"]
clean_training["sum_neg"] = clean_training["weak_neg"] + clean_training["strong_neg"]


print("Tweety BEZ żadnych pozytywnych słów:", (clean_training["sum_pos"]==0).mean())
print("Tweety BEZ żadnych negatywnych słów:", (clean_training["sum_neg"]==0).mean())



#print("\n=== DIAGNOSTYKA ===")
#print("Rozkład klas:")
#print(y.value_counts(normalize=True))
#print("\nStatystyki cech:")
#print(X.describe())
#print("\nIle tweetów ma WSZYSTKIE cechy = 0?")
#print((X.sum(axis=1) == 0).sum())



clean_training["full_text"] = clean_training["text"].fillna("") + " " +  clean_training["topic"].fillna("")

texts=clean_training["full_text"]

X_custom = clean_training[["weak_pos", "strong_pos", "weak_neg", "strong_neg", 
                    "is_positive_dominant", "is_negative_dominant", 
                    "word_count", "polarity", "has_link", "starts_with_RT", 
                    "digit_count", "has_hashtag"]]

y = clean_training["sentiment"]

X_text_train, X_text_test, y_train, y_test = train_test_split(texts, y,test_size=0.2, random_state=42, stratify = y)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', RidgeClassifier(class_weight="balanced"))
])

param_grid = {
    'tfidf__max_features': [5000, 8000, 10000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'model__alpha': [0.3, 0.5, 0.7, 1.0, 1.2]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_text_train, y_train)

print("Najlepsze parametry:", grid.best_params_)
print("Najlepszy wynik F1 (CV):", grid.best_score_)

#wybera wiersze po indeksach
#loc - wybiera dokładnie te same tweety, ale z moich cech numerycznych
X_custom_train = X_custom.loc[X_text_train.index]
X_custom_test = X_custom.loc[X_text_test.index]

model_tfidf = TfidfVectorizer(max_features=grid.best_params_['tfidf__max_features'], 
                              ngram_range=grid.best_params_['tfidf__ngram_range'])

model_tfidf.fit(X_text_train)

#ostajemy (liczba_tweetow_train x liczba_cech_TFIDF)
#zależy nam, aby RidgeClassifier miało wektorowe wejście treningowe i wejście do predyckji
X_train_tfidf = model_tfidf.transform(X_text_train)
X_test_tfidf=model_tfidf.transform(X_text_test)

X_train_final= hstack([X_train_tfidf, X_custom_train])
X_test_final = hstack([X_test_tfidf, X_custom_test])


model = RidgeClassifier(class_weight="balanced", 
                        alpha=grid.best_params_['model__alpha'])
results = []

print("====WYNIKI DLA TWITTER SENTIMENTAL====")
train_metrics, test_metrics = train_and_evaluate_model(model, X_train_final, X_test_final, y_train, y_test)
results = {"train": train_metrics, "test": test_metrics}

train_acc = results['train']['accuracy']
test_acc = results['test']['accuracy']

print(f"\nRidgeClassifier: ")
print(f" Accuracy train: {train_acc:.4f}, test: {test_acc:.4f}")
if train_acc - test_acc >0.05:
    print("\n Wniosek: Model wykazuje oznaki przetrenowania (overfitting).")
elif abs(train_acc-test_acc) <0.02:
    print("\n Wniosek: Model dobrze generalizuje, brak oznak przetrenowania.")
else:
    print("\n Wniosek: Model działa poprawnie, z lekką różnicą w generalizacji.")

summary=[{"Model":"RidgeClassifier", "Train acc": train_acc, "Test acc": test_acc, "Test F1": results['test']['f1']}]
df = pd.DataFrame(summary)
print("\nPodsumowanie wyników:\n")
print(df.to_string(index=False))


#wizualizacja wyników
plt.figure(figsize=(12,6))
metrics = ['accuracy', 'precision', 'recall', 'f1']

df_plot = pd.DataFrame([{
    'Model': "RidgeClassifierC",
    'Accuracy': results['test']['accuracy'],
    'Precision': results['test']['precision'],
    'Recall': results['test']['recall'],
    'F1': results['test']['f1']  
}])

#wizualizacja
df_plot.set_index('Model').plot(kind='bar', figsize=(12,6))
plt.title("Porównanie metryk testowych — Twitter Sentiment")
plt.ylabel("Wartosc metryki")
plt.ylim(0,1) #oś y w przedziale od 0 do 1
plt.xticks(rotation=45) #napis pod kątem
plt.tight_layout() 
plt.show() #Pokazuje wykres

