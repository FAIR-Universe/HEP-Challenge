# ------------------------------
# Dummy Sample Submission
# ------------------------------

XGBOOST = True
TENSORFLOW = False
TORCH = False

from statistical_analysis import  StatisticalAnalysis
import numpy as np
import os

current_file = os.path.dirname(os.path.abspath(__file__))

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
            get_train_set  # train_set is a dictionary with data, labels and weights
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

        self.training_set, valid_set = train_test_split(
            data_set=self.train_set, test_size=0.5, random_state=42, reweight=True
        )

        del self.train_set

        print("Training Data: ", self.training_set["data"].shape)
        print("Training Labels: ", self.training_set["labels"].shape)
        print("Training Weights: ", self.training_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 0].sum(),
        )
        print()
        print("Valid Data: ", valid_set["data"].shape)
        print("Valid Labels: ", valid_set["labels"].shape)
        print("Valid Weights: ", valid_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            valid_set["weights"][valid_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            valid_set["weights"][valid_set["labels"] == 0].sum(),
        )
        print(" \n ")


        print("Training Data: ", self.training_set["data"].shape)

        self.re_train = True
        
        if XGBOOST:
            from boosted_decision_tree import BoostedDecisionTree

            self.model = BoostedDecisionTree()
            module_file = current_file + "/model_XGB.json"
            if os.path.exists(module_file):
                self.model.load(module_file)
                self.re_train = False  # if model is already trained, no need to retrain

            self.name = "model_XGB"

            print("Model is BDT")
        elif TENSORFLOW:
            from neural_network_TF import NeuralNetwork

            module_file = current_file + "/model_tf.keras"
            self.model = NeuralNetwork(train_data=self.training_set["data"])
            if os.path.exists(module_file):
                self.model.load(module_file)
                self.re_train = False  # if model is already trained, no need to retrain

            self.name = "model_tf"
            print("Model is TF NN")

        elif TORCH:
            from neural_network_torch import NeuralNetwork

            module_file = current_file + "/model_torch.pt"
            self.model = NeuralNetwork(train_data=self.train_set["data"])
            if os.path.exists(module_file):
                self.model.load(module_file)
                self.re_train = False  # if model is already trained, no need to retrain

            self.name = "model_torch"
            print("Model is Torch NN")
        self.stat_analysis = StatisticalAnalysis(self.model, valid_set)


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
            self.model.save( current_file + "/" + self.name)


        saved_info_file = current_file + "/saved_info_" + self.name + ".pkl"
        if os.path.exists(saved_info_file):
            self.stat_analysis.load(saved_info_file) 
        else:   
            self.stat_analysis.calculate_saved_info()
            self.stat_analysis.save(saved_info_file)

        train_score = self.model.predict(self.training_set["data"])
        train_results = self.stat_analysis.compute_mu(
            train_score, self.training_set["weights"],plot=True)
        
        print("Train Results: ")
        for key in train_results.keys():
            print("\t", key, " : ", train_results[key])

    def balance_set(self):
        balanced_set = self.training_set.copy()

        weights_train = self.training_set["weights"].copy()
        train_labels = self.training_set["labels"].copy()
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

        result = self.stat_analysis.compute_mu(predictions, test_weights)

        print("Test Results: ", result)


        return result

def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):
    data = data_set["data"].copy()
    train_set = {}
    test_set = {}
    full_size = len(data)
    
    print(f"Full size of the data is {full_size}")
    
    np.random.seed(random_state)
    if isinstance(test_size, float):
        test_number = int(test_size * full_size)
        random_index = np.random.randint(0, full_size, test_number)
    elif isinstance(test_size, int):
        random_index = np.random.randint(0, full_size, test_size)
    else:
        raise ValueError("test_size should be either float or int")

    full_range = data.index
    remaining_index = full_range[np.isin(full_range, random_index, invert=True)]
    remaining_index = np.array(remaining_index)
    
    print(f"Train size is {len(remaining_index)}")
    print(f"Test size is {len(random_index)}")
    
    for key in data_set.keys():
        if (key != "data") and (key != "settings"):
            array = np.array(data_set[key])
            test_set[key] = array[random_index]
            train_set[key] = array[remaining_index]

    test_set["data"] = data.iloc[random_index]
    train_set["data"] = data.iloc[remaining_index]

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

