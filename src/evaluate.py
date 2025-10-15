from tensorflow import keras
from src import config
from src.data_loader import load_and_prepare_data


def evaluate():
    X_train, X_valid, y_train, y_valid = load_and_prepare_data()

    print(f"Loading model from {config.MODEL_PATH}...")
    try:
        model = keras.models.load_model(config.MODEL_PATH)
    except IOError:
        print("Error: Model file not found. Please run train.py first.")
        return

    # 3. Evaluate the model
    print("Evaluating model performance on the validation set...")
    results = model.evaluate(X_valid, y_valid, verbose=0)

    print("\n--- Evaluation Results ---")
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")


if __name__ == '__main__':
    evaluate()