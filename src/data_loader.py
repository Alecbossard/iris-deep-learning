import pandas as pd
from sklearn.model_selection import train_test_split
from src import config


def load_and_prepare_data():
    data = pd.read_csv(config.DATA_PATH, header=None)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    X = data.drop('species', axis=1)
    y = pd.get_dummies(data['species'])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X.astype('float32'),
        y.astype('float32'),
        random_state=config.RANDOM_STATE,
        stratify=y,
        train_size=config.TRAIN_SIZE
    )
    return X_train, X_valid, y_train, y_valid