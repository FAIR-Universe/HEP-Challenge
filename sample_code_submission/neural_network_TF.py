import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle


class NeuralNetwork:
    """
    This class implements a neural network classifier.

    Attributes:
        model (Sequential): The neural network model.
        scaler (StandardScaler): The scaler used for feature scaling.

    Methods:
        __init__(self, train_data): Initializes the NeuralNetwork object.
        fit(self, train_data, y_train, weights_train=None): Fits the neural network model to the training data.
        predict(self, test_data): Predicts the output labels for the test data.
        save(self, model_name): Saves the trained model and scaler to disk.
        load(self, model_path): Loads a trained model and scaler from disk.

    """

    def __init__(self, train_data):
        self.model = Sequential()

        n_dim = train_data.shape[1]

        self.model.add(Dense(100, input_dim=n_dim, activation="relu"))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(100, activation="relu"))        
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None):
        """
        Fits the neural network model to the training data.

        Args:
            train_data (pandas.DataFrame): The input training data.
            y_train (numpy.ndarray): The target training labels.
            weights_train (numpy.ndarray, optional): The sample weights for training data.

        Returns:
            None

        """
        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(X_train, y_train, sample_weight=weights_train, epochs=2, verbose=2)

    def predict(self, test_data):
        """
        Predicts the output labels for the test data.

        Args:
            test_data (pandas.DataFrame): The input test data.

        Returns:
            numpy.ndarray: The predicted output labels.

        """
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()
    
    def save(self, model_name):
        """
        Saves the trained model and scaler to disk.

        Args:
            model_name (str): The name of the model file to be saved.

        Returns:
            None

        """
        model_path = model_name + ".keras"
        self.model.save(model_path)
        
        scaler_path = model_name + ".pkl"
        pickle.dump(self.scaler, open(scaler_path, "wb"))
        
    def load(self, model_path):
        """
        Loads a trained model and scaler from disk.

        Args:
            model_path (str): The path to the saved model file.

        Returns:
            Sequential: The loaded model.

        """
        self.model = load_model(model_path)
        self.scaler = pickle.load(open(model_path.replace(".keras", ".pkl"), "rb"))
        
        return self.model
