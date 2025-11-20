from add_features import add_features
from load_and_clean_data import load_and_clean_data
from train_with_grid_and_custom_features import train_with_grid_and_custom_features

def main():
    # 1. Wczytanie i wstępne czyszczenie danych
    #    (duplikaty, brakujące wartości, preprocessing tekstu)
    clean_training = load_and_clean_data()

    # 2. Feature engineering — generowanie naszych własnych cech
    #    (polaryzacja, słowa pozytywne/negatywne, linki, RT, cyfry, hashtagi itp.)
    clean_training = add_features(clean_training)

    # 3. Trening modelu + GridSearch + łączenie TF-IDF z custom features
    #    (prawdziwy pipeline: optymalizacja → fit → ocena → metryki → wykresy)
    train_with_grid_and_custom_features(clean_training)


# Standardowy punkt wejścia aplikacji — umożliwia uruchamianie jako skrypt
if __name__ == "__main__":
    main()
