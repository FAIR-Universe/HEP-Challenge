import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def __init__(self, train_data):
        self.model = Sequential()

        n_dim = train_data.shape[1]

        self.model.add(Dense(10, input_dim=n_dim, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(X_train, y_train, sample_weight=weights_train, epochs=2, verbose=2)

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()
