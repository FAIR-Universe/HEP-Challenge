# ------------------------------
# Dummy Sample Submission
# ------------------------------

XGBOOST = True
TENSORFLOW = False
TORCH = False

from statistical_analysis import calculate_saved_info, compute_mu

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
current_dir = os.path.dirname(__file__)

class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class consists of the following functions:
    1) __init__:
        Initializes the Model class.
        Args:
            * get_train_set (callable, optional): A function that returns a dictionary with data, labels, weights, detailed_labels, and settings.
            * systematics (object, optional): A function that can be used to get a dataset with systematics added.
        Returns:
            None

    2) fit:
        Trains the model.
        Params:
            None
        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.
        Returns:
            None

    3) predict:
        Predicts the values for the test set.
        Parameters:
            test_set (dict): A dictionary containing the test data and weights.
        Returns:
            dict: A dictionary with the following keys:
            * 'mu_hat': The predicted value of mu.
            * 'delta_mu_hat': The uncertainty in the predicted value of mu.
            * 'p16': The lower bound of the 16th percentile of mu.
            * 'p84': The upper bound of the 84th percentile of mu.

    4) balance_set:
        Balances the training set by equalizing the number of background and signal events.
        Params:
            None
        Returns:
            dict: A dictionary with the balanced training set.
    """

    def __init__(self, get_train_set=None, systematics=None):
        """
        Initializes the Model class.

        Args:
            * get_train_set (callable, optional): A function that returns a dictionary with data, labels, weights,detailed_labels and settings.
            * systematics (object, optional): A function that can be used to get a dataset with systematics added.

        Returns:
            None
        """
        self.get_train_set = get_train_set
        self.systematics = systematics
        self.re_train = True
        
        if XGBOOST:
            from boosted_decision_tree import BoostedDecisionTree
            module_file = current_dir + "/model_XGB.json"
            self.model = BoostedDecisionTree()

        elif TENSORFLOW:
            from neural_network_TF import NeuralNetwork
            module_file = current_dir + "/model_tf.keras"
            self.model = NeuralNetwork()
            
        elif TORCH:
            from neural_network_torch import NeuralNetwork
            module_file = current_dir +  "/model_torch.pt"
            self.model = NeuralNetwork()

        self.name = self.model.name
        
        try:
            self.model.load(module_file)
            self.re_train = False
            
        except:
            print("Model not found, retraining the model")
            


    def fit(self):
        """
        Trains the model.

        Params:
            None

        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.

        Returns:
            None
        """

        if self.re_train:

            train_set = self.get_train_set() # train_set is a dictionary with data, labels, and weights
            train_set = self.systematics(train_set)
            
            training_set, holdout_set = train_test_split(
                train_set, test_size=0.5, random_state=42, reweight=True
            )

            balanced_set = balance_set(training_set)
            
            self.model.fit(balanced_set["data"], balanced_set["labels"], balanced_set["weights"])
            self.model.save()
            
                
        saved_info_file = current_dir + "/saved_info_" + self.name + ".pkl"
        if os.path.exists(saved_info_file):
            with open(saved_info_file, "rb") as f:
                self.saved_info = pickle.load(f)
        else:
            self.saved_info = calculate_saved_info(self.model, holdout_set)
            with open(saved_info_file, "wb") as f:
                pickle.dump(self.saved_info, f)

        if self.re_train:

            train_score = self.model.predict(train_set["data"])
            train_results = compute_mu(
                train_score, train_set["weights"], self.saved_info
            )
            
            holdout_score = self.model.predict(holdout_set["data"])
            holdout_results = compute_mu(
                holdout_score, holdout_set["weights"], self.saved_info
            )

            print("Train Results: ")
            for key in train_results.keys():
                print("\t", key, " : ", train_results[key])
                
            print("Holdout Results: ")
            for key in holdout_results.keys():
                print("\t", key, " : ", holdout_results[key])


    def predict(self, test_set):
        """
        Predicts the values for the test set.

        Parameters:
            test_set (dict): A dictionary containing the test data, and weights.

        Returns:
            dict: A dictionary with the following keys:
            * 'mu_hat': The predicted value of mu.
            * 'delta_mu_hat': The uncertainty in the predicted value of mu.
            * 'p16': The lower bound of the 16th percentile of mu.
            * 'p84': The upper bound of the 84th percentile of mu.
        """

        test_data = test_set["data"]
        test_weights = test_set["weights"]

        predictions = self.model.predict(test_data)

        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)

        print("Test Results: ", result_mu_cal)

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result

def balance_set(train_set):
    """
    Balances the training set by equalizing the number of background and signal events.

    Args:
        train_set (dict): A dictionary containing the training data, labels, and weights.

    Returns:
        dict: A dictionary with the balanced training set.
    """
    balanced_set = train_set.copy()

    weights_train = train_set["weights"].copy()
    train_labels = train_set["labels"].copy()
    class_weights_train = (
        weights_train[train_labels == 0].sum(),
        weights_train[train_labels == 1].sum(),
    )

    for i in range(len(class_weights_train)):  # loop on B then S target
        # training dataset: equalize number of background and signal
        weights_train[train_labels == i] *= (
            max(class_weights_train) / class_weights_train[i]
        )
        # test dataset : increase test weight to compensate for sampling

    balanced_set["weights"] = weights_train

    return balanced_set


def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):
    """
    Splits the data into training and testing sets.

    Args:
        * data_set (dict): A dictionary containing the data, labels, weights, detailed_labels, and settings
        * test_size (float, optional): The size of the testing set. Defaults to 0.2.
        * random_state (int, optional): The random state. Defaults to 42.
        * reweight (bool, optional): Whether to reweight the data. Defaults to False.

    Returns:
        tuple: A tuple containing the training and testing
    """
    data = data_set["data"].copy()
    train_set = {}
    test_set = {}
    full_size = len(data)

    print(f"Full size of the data is {full_size}")

    for key in data_set.keys():
        if (key != "data") and (key != "settings"):
            data[key] = np.array(data_set[key])

    train_data, test_data = sk_train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    for key in data_set.keys():
        if (key != "data") and (key != "settings"):
            train_set[key] = np.array(train_data.pop(key))
            test_set[key] = np.array(test_data.pop(key))

    train_set["data"] = train_data
    test_set["data"] = test_data

    if reweight is True:
        signal_weight = np.sum(data_set["weights"][data_set["labels"] == 1])
        background_weight = np.sum(data_set["weights"][data_set["labels"] == 0])
        signal_weight_train = np.sum(train_set["weights"][train_set["labels"] == 1])
        background_weight_train = np.sum(train_set["weights"][train_set["labels"] == 0])
        signal_weight_test = np.sum(test_set["weights"][test_set["labels"] == 1])
        background_weight_test = np.sum(test_set["weights"][test_set["labels"] == 0])

        train_set["weights"][train_set["labels"] == 1] = train_set["weights"][
            train_set["labels"] == 1
        ] * (signal_weight / signal_weight_train)
        test_set["weights"][test_set["labels"] == 1] = test_set["weights"][
            test_set["labels"] == 1
        ] * (signal_weight / signal_weight_test)

        train_set["weights"][train_set["labels"] == 0] = train_set["weights"][
            train_set["labels"] == 0
        ] * (background_weight / background_weight_train)
        test_set["weights"][test_set["labels"] == 0] = test_set["weights"][
            test_set["labels"] == 0
        ] * (background_weight / background_weight_test)

    return train_set, test_set
