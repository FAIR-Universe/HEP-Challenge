import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
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
        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        w_train = torch.tensor(w_train, dtype=torch.float32)
        

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        def weighted_loss(y, y_hat, w):
            return (criterion(y, y_hat)*w).mean()
        
        epochs = 1
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_train).ravel()
            loss = weighted_loss(outputs, y_train,w_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        test_data = torch.from_numpy(test_data).float()
        outputs = self.forward(test_data)
        return outputs.detach().numpy()
