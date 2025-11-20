import pandas as pd
import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from training_and_calculate_metrics import train_and_evaluate_model

def train_with_grid_and_custom_features(clean_training):
    texts = clean_training["full_text"]

    X_custom = clean_training[[
        "weak_pos", "strong_pos", "weak_neg", "strong_neg",
        "is_positive_dominant", "is_negative_dominant",
        "word_count", "polarity", "has_link", "starts_with_RT",
        "digit_count", "has_hashtag"
    ]]

    y = clean_training["sentiment"]

    X_text_train, X_text_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

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

    X_custom_train = X_custom.loc[X_text_train.index]
    X_custom_test = X_custom.loc[X_text_test.index]

    model_tfidf = TfidfVectorizer(
        max_features=grid.best_params_['tfidf__max_features'],
        ngram_range=grid.best_params_['tfidf__ngram_range']
    )

    model_tfidf.fit(X_text_train)

    X_train_tfidf = model_tfidf.transform(X_text_train)
    X_test_tfidf = model_tfidf.transform(X_text_test)

    X_train_final = hstack([X_train_tfidf, X_custom_train])
    X_test_final = hstack([X_test_tfidf, X_custom_test])

    model = RidgeClassifier(
        class_weight="balanced",
        alpha=grid.best_params_['model__alpha']
    )

    print("====WYNIKI DLA TWITTER SENTIMENTAL====")
    train_metrics, test_metrics = train_and_evaluate_model(
        model, X_train_final, X_test_final, y_train, y_test
    )

    train_acc = train_metrics['accuracy']
    test_acc = test_metrics['accuracy']

    print(f"\nRidgeClassifier: ")
    print(f" Accuracy train: {train_acc:.4f}, test: {test_acc:.4f}")
    if train_acc - test_acc > 0.05:
        print("\n Wniosek: Model wykazuje oznaki przetrenowania (overfitting).")
    elif abs(train_acc - test_acc) < 0.02:
        print("\n Wniosek: Model dobrze generalizuje, brak oznak przetrenowania.")
    else:
        print("\n Wniosek: Model działa poprawnie, z lekką różnicą w generalizacji.")

    summary = [{
        "Model": "RidgeClassifier",
        "Train acc": train_acc,
        "Test acc": test_acc,
        "Test F1": test_metrics['f1']
    }]
    df_summary = pd.DataFrame(summary)
    print("\nPodsumowanie wyników:\n")
    print(df_summary.to_string(index=False))

    plt.figure(figsize=(12,6))
    df_plot = pd.DataFrame([{
        'Model': "RidgeClassifierC",
        'Accuracy': test_metrics['accuracy'],
        'Precision': test_metrics['precision'],
        'Recall': test_metrics['recall'],
        'F1': test_metrics['f1']
    }])

    df_plot.set_index('Model').plot(kind='bar', figsize=(12,6))
    plt.title("Porównanie metryk testowych — Twitter Sentiment")
    plt.ylabel("Wartosc metryki")
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
