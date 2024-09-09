# ------------------------------
# Dummy Sample Submission
# ------------------------------

from statistical_analysis import StatisticalAnalysis
from sklearn.model_selection import train_test_split as sk_train_test_split
import numpy as np
import os

current_file = os.path.dirname(os.path.abspath(__file__))

SYST_NAME = os.getenv("MODEL_SYST_NAME", "stat_only")

syst_fixed_setting = {
        'tes': 1.0,
        'bkg_scale': 1.0,
        'jes': 1.0,
        'soft_met': 0.0,
        'ttbar_scale': 1.0,
        'diboson_scale': 1.0,
}


if SYST_NAME == "stat_only":
    STAT_ONLY = True
else:
    STAT_ONLY = False
    
SYST_FIXED = {k: v for k, v in syst_fixed_setting.items() if k != SYST_NAME}

class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    Atributes:
        * get_train_set (callable): A function that returns a dictionary with data, labels, weights, detailed_labels and settings.
        * systematics (object): A function that can be used to get a dataset with systematics added.
        * model (object): The model object.
        * name (str): The name of the model.
        * stat_analysis (object): The statistical analysis object.
        
    Methods:
        * fit(): Trains the model.
        * predict(test_set): Predicts the values for the test set.

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

        self.systematics = systematics

        self.get_train_set = get_train_set

        self.re_train = True
        self.re_compute = True

        from boosted_decision_tree import BoostedDecisionTree
        self.name = "model_XGB"
        self.model = BoostedDecisionTree()
        module_file = current_file + f"/{self.name}.json"
        if os.path.exists(module_file):
            self.model.load(module_file)
            self.re_train = False  # if model is already trained, no need to retrain

        print("Model is ", self.name)

        self.stat_analysis = StatisticalAnalysis(self.model, stat_only=STAT_ONLY, bins=20, systematics=self.systematics, fixed_syst=SYST_FIXED)

        saved_info_file_dir = current_file + "/saved_info_" + self.name
        if os.path.exists(saved_info_file_dir):
            self.re_compute = not self.stat_analysis.load(saved_info_file_dir)
        else:
            os.makedirs(saved_info_file_dir, exist_ok=True)

        


        print("STAT_ONLY: ", STAT_ONLY)
        print("SYST_FIXED: ", SYST_FIXED)
        print("SYST_NAME: ", SYST_NAME)

    def fit(self):
        """
        Trains the model.

        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.

        Returns:
            None
        """
        
        saved_info_file_dir = current_file + "/saved_info_" + self.name

        if self.re_train or self.re_compute:
            train_set = self.get_train_set()

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

            # train : validation : template = 3 : 1 : 6
            temp_set, holdout_set = train_test_split(
                train_set, test_size=0.6, random_state=42, reweight=True
            )
            temp_set["data"] = temp_set["data"].reset_index(drop=True)
            training_set, valid_set = train_test_split(
                temp_set, test_size=0.2, random_state=42, reweight=True
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

            """
            The systematics code does the following
            1. Apply systematics 
            2. Apply post-systematics cuts to the data
            3. Compute Dervied features

            NOTE:
            Without this transformation, the data will not be representative of the pseudo-experiments
            """
            training_set = self.systematics(training_set)  # to get derived and post cuts
            valid_set = self.systematics(valid_set)  # to get derived and post cuts

            print_set_info("Training", training_set)
            print_set_info("Validation", valid_set)
            print_set_info("Holdout (For Statistical Template)", holdout_set)

            if self.re_train:
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

            self.stat_analysis.calculate_saved_info(holdout_set, saved_info_file_dir)

        self.stat_analysis.alpha_function()


        def predict_and_analyze(
            dataset_name, data_set, fig_name
        ):
            score = self.model.predict(data_set["data"])
            results = self.stat_analysis.compute_mu(
                score,
                data_set["weights"],
                plot=fig_name,
            )

            print(f"{dataset_name} Results:")
            print(f"{'-' * len(dataset_name)} Results:")
            for key, value in results.items():
                print(f"\t{key} : {value}")
            print("\n")

        if self.re_train or self.re_compute:
            # Predict and analyze for each set
            datasets = [
                ("Training", training_set, "train_mu"),
                ("Validation", valid_set, "valid_mu"),
                # ("Holdout", holdout_set, "holdout_mu"),
            ]

            for name, dataset, plot_name in datasets:
                predict_and_analyze(
                    name,
                    dataset,
                    plot_name,
                )


    def predict(self, test_set):
        """
        Predicts the values for the test set.

        Args:
            * test_set (dict): A dictionary containing the data and weights

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
        )

        print("Test Results: ", result)

        return result


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

def balance_set(training_set):
    """
    Balances the training set by equalizing the number of background and signal events.

    Args:
        training_set (dict): A dictionary containing the data, labels, weights, detailed_labels, and settings.

    Returns:
        dict: A dictionary containing the balanced training set.
    """
    
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