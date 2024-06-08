# ------------------------------
# Dummy Sample Submission
# ------------------------------

XGBOOST = False
TENSORFLOW = True
TORCH = False

from statistical_analysis import calculate_saved_info, compute_mu

import os
import pickle


class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class consists of the following functions:
    1) __init__:
        Initializes the Model class.
        Args:
            get_train_set (callable, optional): A function that returns a dictionary with data, labels, weights, detailed_labels, and settings.
            systematics (object, optional): A function that can be used to get a dataset with systematics added.
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
            - 'mu_hat': The predicted value of mu.
            - 'delta_mu_hat': The uncertainty in the predicted value of mu.
            - 'p16': The lower bound of the 16th percentile of mu.
            - 'p84': The upper bound of the 84th percentile of mu.

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
            get_train_set (callable, optional): A function that returns a dictionary with data, labels, weights,detailed_labels and settings.
            systematics (object, optional): A function that can be used to get a dataset with systematics added.

        Returns:
            None
        """
        self.train_set = (
            get_train_set  # train_set is a dictionary with data, labels, and weights
        )
        self.systematics = systematics

        del self.train_set["settings"]

        print("Full data: ", self.train_set["data"].shape)
        print("Full Labels: ", self.train_set["labels"].shape)
        print("Full Weights: ", self.train_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.train_set["weights"][self.train_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.train_set["weights"][self.train_set["labels"] == 0].sum(),
        )
        print(" \n ")

        self.re_train = True

        if XGBOOST:
            from boosted_decision_tree import BoostedDecisionTree

            self.model = BoostedDecisionTree()
            module_file = "model_XGB.json"
            if os.path.exists(module_file):
                self.model.load(module_file)
                self.re_train = False  # if model is already trained, no need to retrain

            self.name = "model_XGB"

            print("Model is BDT")
        elif TENSORFLOW:
            from neural_network_TF import NeuralNetwork

            module_file = "./model_tf.keras"
            self.model = NeuralNetwork(train_data=self.train_set["data"])
            if os.path.exists(module_file):
                self.model.load(module_file)
                self.re_train = False  # if model is already trained, no need to retrain

            self.name = "model_tf"
            print("Model is TF NN")

        elif TORCH:
            from neural_network_torch import NeuralNetwork

            module_file = "./model_torch.pt"
            self.model = NeuralNetwork(train_data=self.train_set["data"])
            if os.path.exists(module_file):
                self.model.load(module_file)
                self.re_train = False  # if model is already trained, no need to retrain

            self.name = "model_torch"
            print("Model is Torch NN")

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

            balanced_set = self.balance_set()
            self.model.fit(
                balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
            )

            self.model.save(self.name)

        saved_info_file = "./saved_info_" + self.name + ".pkl"
        if os.path.exists(saved_info_file):
            with open(saved_info_file, "rb") as f:
                self.saved_info = pickle.load(f)
        else:
            self.saved_info = calculate_saved_info(self.model, self.train_set)
            with open(saved_info_file, "wb") as f:
                pickle.dump(self.saved_info, f)

        train_score = self.model.predict(self.train_set["data"])
        train_results = compute_mu(
            train_score, self.train_set["weights"], self.saved_info
        )

        print("Train Results: ")
        for key in train_results.keys():
            print("\t", key, " : ", train_results[key])

    def balance_set(self):
        balanced_set = self.train_set.copy()

        weights_train = self.train_set["weights"].copy()
        train_labels = self.train_set["labels"].copy()
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

    def predict(self, test_set):
        """
        Predicts the values for the test set.

        Parameters:
            test_set (dict): A dictionary containing the test data, and weights.

        Returns:
            dict: A dictionary with the following keys:
            - 'mu_hat': The predicted value of mu.
            - 'delta_mu_hat': The uncertainty in the predicted value of mu.
            - 'p16': The lower bound of the 16th percentile of mu.
            - 'p84': The upper bound of the 84th percentile of mu.
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
