# ------------------------------
# Dummy Sample Submission
# ------------------------------

from statistical_analysis import StatisticalAnalysis
from sklearn.model_selection import train_test_split as sk_train_test_split
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

        self.systematics = systematics

        self.get_train_set = get_train_set

        self.re_train = True

        from boosted_decision_tree import BoostedDecisionTree

        self.model = BoostedDecisionTree()
        module_file = current_file + "/model_XGB.json"
        if os.path.exists(module_file):
            self.model.load(module_file)
            self.re_train = False  # if model is already trained, no need to retrain

        self.name = "model_XGB"

        print("Model is BDT")

        self.stat_analysis = StatisticalAnalysis(self.model, stat_only=False, bins=5)

    def fit(self, stat_only: bool = None, syst_settings: dict[str, bool] = None):
        """
        Trains the model.

        Params:
            stat_only (bool, optional): Force to compute stats only results [the highest priority]. Defaults to None.
            syst_settings (dict, optional): Dictionary containing the systematic settings of whether to fix systematics in fitting. For example, {'jes': True}. Defaults to None.

        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.

        Returns:
            None
        """
        
        saved_info_file = current_file + "/saved_info_" + self.name + ".pkl"

        
        if self.re_train:
            train_set = self.get_train_set()

            """
            The systematics code does the following
            1. Apply systematics 
            2. Apply post-systematics cuts to the data
            3. Compute Dervied features
            
            NOTE:
            Without this transformation, the data will not be representative of the pseudo-experiments
            """
            train_set = self.systematics(train_set)

            print("Full data: ", train_set["data"].shape)
            print("Full Labels: ", train_set["labels"].shape)
            print("Full Weights: ", train_set["weights"].shape)
            print(
                "sum_signal_weights: ",
                train_set["weights"][train_set["labels"] == 1].sum(),
            )
            print(
                "sum_bkg_weights: ",
                train_set["weights"][train_set["labels"] == 0].sum(),
            )
            print(" \n ")

            # First, split the data into two parts: 1/2 and 1/2
            training_set, temp_set = train_test_split(
                train_set, test_size=0.5, random_state=42, reweight=True
            )

            # Now split the temp_set into validation and holdout sets (statistical template set) with equal size
            temp_set["data"] = temp_set["data"].reset_index(drop=True)
            valid_set, holdout_set = train_test_split(
                temp_set, test_size=0.5, random_state=42, reweight=True
            )

            del train_set

            def print_set_info(name, dataset):
                print(f"{name} Set:")
                print(f"{'-' * len(name)} ")
                print(f"  Data Shape:          {dataset['data'].shape}")
                print(f"  Labels Shape:        {dataset['labels'].shape}")
                print(f"  Weights Shape:       {dataset['weights'].shape}")
                print(
                    f"  Sum Signal Weights:  {dataset['weights'][dataset['labels'] == 1].sum():.2f}"
                )
                print(
                    f"  Sum Background Weights: {dataset['weights'][dataset['labels'] == 0].sum():.2f}"
                )
                print("\n")

            print_set_info("Training", training_set)
            print_set_info("Validation", valid_set)
            print_set_info("Holdout (For Statistical Template)", holdout_set)

            balanced_set = balance_set(training_set)

            self.model.fit(
                balanced_set["data"],
                balanced_set["labels"],
                balanced_set["weights"],
                valid_set=[
                    valid_set["data"],
                    valid_set["labels"],
                    valid_set["weights"],
                ],
            )

            self.model.save(current_file + "/" + self.name)
            
            self.stat_analysis.calculate_saved_info(holdout_set)
            self.stat_analysis.save(saved_info_file)

        else:
            assert os.path.exists(
                saved_info_file
            ), f"Saved info file {saved_info_file} does not exist."
            self.stat_analysis.load(saved_info_file)

        def predict_and_analyze(
            dataset_name, data_set, fig_name, stat_only, syst_settings
        ):
            score = self.model.predict(data_set["data"])
            results = self.stat_analysis.compute_mu(
                score,
                data_set["weights"],
                plot=fig_name,
                stat_only=stat_only,
                syst_fixed_setting=syst_settings,
            )

            print(f"{dataset_name} Results:")
            print(f"{'-' * len(dataset_name)} Results:")
            for key, value in results.items():
                print(f"\t{key} : {value}")
            print("\n")

        if self.re_train:
            # Predict and analyze for each set
            datasets = [
                ("Training", training_set, "train_mu"),
                ("Validation", valid_set, "valid_mu"),
                ("Holdout", holdout_set, "holdout_mu"),
            ]

            for name, dataset, plot_name in datasets:
                predict_and_analyze(
                    name,
                    dataset,
                    plot_name,
                    stat_only=stat_only,
                    syst_settings=syst_settings,
                )


    def predict(
        self, test_set, stat_only: bool = None, syst_settings: dict[str, float] = None
    ):
        """
        Predicts the values for the test set.

        Params:
            stat_only (bool, optional): Force to compute stats only results [the highest priority]. Defaults to None.
            syst_settings (dict, optional): Dictionary containing the systematic settings of whether to fix systematics in fitting. For example, {'jes': True}. Defaults to None.

        Returns:
            dict: A dictionary with the following keys:
            - 'mu_hat': The predicted value of mu.
            - 'delta_mu_hat': The uncertainty in the predicted value of mu.
            - 'p16': The lower bound of the 16th percentile of mu.
            - 'p84': The upper bound of the 84th percentile of mu.
        """

        test_data = test_set["data"]
        test_weights = test_set["weights"]

        print("[*] -> test weights sum = ", test_weights.sum())

        predictions = self.model.predict(test_data)

        result = self.stat_analysis.compute_mu(
            predictions,
            test_weights,
            stat_only=stat_only,
            syst_fixed_setting=syst_settings,
        )

        print("Test Results: ", result)

        return result


def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):
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

def balance_set(training_set):
    balanced_set = training_set.copy()

    weights_train = training_set["weights"].copy()
    train_labels = training_set["labels"].copy()
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