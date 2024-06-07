import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.optim as optim
import pickle


class NeuralNetwork(nn.Module):
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
        super(NeuralNetwork, self).__init__()

        n_dim = train_data.shape[1]

        self.fc1 = nn.Linear(n_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.scaler = StandardScaler()

    def forward(self, x):
        x = self.scaler.transform(x)
        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x.flatten().ravel()

    def fit(self, train_data, y_train, w_train=None):
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

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        w_train = torch.tensor(w_train, dtype=torch.float32)

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        def weighted_loss(y, y_hat, w):
            return (criterion(y, y_hat) * w).mean()

        epochs = 1
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_train).ravel()
            loss = weighted_loss(outputs, y_train, w_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def predict(self, test_data):
        """
        Predicts the output labels for the test data.

        Args:
            test_data (pandas.DataFrame): The input test data.

        Returns:
            numpy.ndarray: The predicted output labels.

        """        
        test_data = self.scaler.transform(test_data)
        test_data = torch.from_numpy(test_data).float()
        outputs = self.forward(test_data)
        return outputs.detach().numpy()

    def save(self, model_name):
        """
        Saves the trained model and scaler to disk.

        Args:
            model_name (str): The name of the model file to be saved.

        Returns:
            None

        """        
        model_path = model_name + ".pt"
        torch.save(self.state_dict(), model_path)
        
        scaler_path = model_name + ".pkl"
        pickle.dump(self.scaler, open(scaler_path, "wb"))

    def load(self, model_path):
        """
        Loads a trained model and scaler from disk.

        Args:
            model_path (str): The path to the saved model file.

        Returns:
            NeuralNetwork: The loaded model.

        """
        self.load_state_dict(torch.load(model_path))
        self.scaler = pickle.load(open(model_path.replace(".pt", ".pkl"), "rb"))

        return self
