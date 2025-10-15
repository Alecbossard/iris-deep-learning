import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from src import config
from src.data_loader import load_and_prepare_data
from src.model import build_model


def train():
    print("Loading and preparing data...")
    X_train, X_valid, y_train, y_valid = load_and_prepare_data()

    # 2. Build model
    print("Building model...")
    model = build_model()
    model.summary()

    # 3. Set up callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        restore_best_weights=True,
    )

    # 4. Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=[early_stopping],
        verbose=1,  # Show progress bar
    )

    # 5. Save the trained model
    print(f"Training complete. Saving model to {config.MODEL_PATH}")
    model.save(config.MODEL_PATH)

    # 6. Plot and show results
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot(title="Loss")
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")
    plt.show()

    print("\n--- Training Summary ---")
    print(f"Best Validation Loss: {history_df['val_loss'].min():0.4f}")
    print(f"Best Validation Accuracy: {history_df['val_accuracy'].max():0.4f}")


if __name__ == '__main__':
    train()