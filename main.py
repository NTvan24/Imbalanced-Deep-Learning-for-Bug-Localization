
import extract_features
import train


def main():
    # Extract features from the dataset
    extract_features.main()

    # Train the model
    train.main()
if __name__ == "__main__":
    main()