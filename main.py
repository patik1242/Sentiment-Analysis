from add_features import add_features
from load_and_clean_data import load_and_clean_data
from train_with_grid_and_custom_features import train_with_grid_and_custom_features

def main():
    clean_training = load_and_clean_data()

    clean_training = add_features(clean_training)

    train_with_grid_and_custom_features(clean_training)

if __name__ == "__main__":
    main()