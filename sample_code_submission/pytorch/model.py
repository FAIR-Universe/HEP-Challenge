import os
from sys import path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

submissions_dir = os.path.dirname(os.path.abspath(__file__))
path.append(submissions_dir)

from bootstrap import *

EPSILON = np.finfo(float).eps
CUDA = 'cuda'
CPU = 'cpu'


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TrainDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict['data']
        self.labels = data_dict['labels']
        self.weights = data_dict['weights']
        self.settings = data_dict['settings']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = torch.tensor(self.data.iloc[index].values, dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        weight = torch.tensor(self.weights[index], dtype=torch.float32)

        return data_sample, label, weight


class TestDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict['data']
        self.weights = data_dict['weights']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = torch.tensor(self.data.iloc[index].values, dtype=torch.float32)
        weight = torch.tensor(self.weights[index], dtype=torch.float32)

        return data_sample, weight


# ------------------------------
# Baseline Model
# ------------------------------
class Model():
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init : initialize a classifier
    2) fit : can be used to train a classifier
    3) predict: predict mu_hats,  delta_mu_hat and q1,q2

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods
    """

    def __init__(
            self,
            train_set=None,
            systematics=None
    ):
        """
        Model class constructor

        Params:
            train_set:
                labelled train set

            systematics:
                systematics class

        Returns:
            None
        """

        # Set class variables from parameters
        self.train_set = train_set
        self.systematics = systematics

        input_size = len(self.train_set['data'].columns)
        hidden_size = int(input_size/2)
        output_size = 2
        learning_rate = 0.001
        device = torch.device(CUDA if torch.cuda.is_available() else CPU)

        self.device = device
        self.model = SimpleModel(input_size, hidden_size, output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = 10

        print(f"[!] Device: {self.device}")

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """
        if self.device == torch.device(CPU):
            print("[-] GPU not found. Not proceeding with CPU!")
            return

        train_dataset = TrainDataset(self.train_set)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        for epoch in range(self.num_epochs):
            for inputs, labels, weights in train_dataloader:
                inputs, labels, weights = inputs.to(self.device), labels.to(self.device), weights.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            print(f'[*] --- Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}')

        Y_hat_train = self._predict(train_dataset)
        Y_train = np.array(train_dataset.labels).astype(int)
        weights_train = train_dataset.weights

        # compute gamma_roi
        weights_train_signal = weights_train[Y_train == 1]
        weights_train_background = weights_train[Y_train == 0]

        Y_hat_train_signal = Y_hat_train[Y_train == 1]
        Y_hat_train_background = Y_hat_train[Y_train == 0]

        self.gamma_roi = (weights_train_signal[Y_hat_train_signal == 1]).sum()

        # compute beta_roi
        self.beta_roi = (weights_train_background[Y_hat_train_background == 1]).sum()
        if self.gamma_roi == 0:
            self.gamma_roi = EPSILON

    def predict(self, test_set):
        """
        Params:
            test_set

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """
        if self.device == torch.device(CPU):
            print("[-] GPU not found. Not proceeding with CPU!")
            return None

        test_dataset = TestDataset(test_set)
        Y_hat_test = self._predict(test_dataset)
        weights_test = test_dataset.weights

        # get n_roi
        n_roi = self.N_calc_2(weights_test[Y_hat_test == 1])

        mu_hat = (n_roi - self.beta_roi)/self.gamma_roi

        sigma_mu_hat = np.std(mu_hat)
        delta_mu_hat = 2*sigma_mu_hat

        mu_p16 = np.percentile(mu_hat, 16)
        mu_p84 = np.percentile(mu_hat, 84)

        return {
            "mu_hat": mu_hat.mean(),
            "delta_mu_hat": delta_mu_hat,
            "p16": mu_p16,
            "p84": mu_p84
        }

    def _predict(self, dataset):

        test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        predictions = []

        with torch.no_grad():
            for inputs, *_ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def N_calc_2(self, weights, n=100):
        total_weights = []
        for i in range(n):
            bootstrap_weights = bootstrap(weights=weights, seed=42+i)
            total_weights.append(np.array(bootstrap_weights).sum())
        n_calc_array = np.array(total_weights)
        return n_calc_array
