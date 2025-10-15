from tensorflow import keras
from tensorflow.keras import layers
from src import config


def build_model():
    model = keras.Sequential([
        keras.Input(shape=config.INPUT_SHAPE),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(3, activation='softmax'),
    ])


    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model