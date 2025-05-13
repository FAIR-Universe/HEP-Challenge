# ------------------------------
# Dummy Sample Submission
# ------------------------------
import pandas as pd

from statistical_analysis import StatisticalAnalysis
from boosted_decision_tree import BoostedDecisionTree

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


def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):
    data = data_set.copy()
    full_size = len(data)
    print(f"Full size of the data is {full_size}")

    train_data, test_data = sk_train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    if reweight:
        # Compute original total weights
        signal_weight = np.sum(data_set.loc[data_set["labels"] == 1, "weights"])
        background_weight = np.sum(data_set.loc[data_set["labels"] == 0, "weights"])

        # Compute train/test weights
        signal_weight_train = np.sum(train_data.loc[train_data["labels"] == 1, "weights"])
        background_weight_train = np.sum(train_data.loc[train_data["labels"] == 0, "weights"])
        signal_weight_test = np.sum(test_data.loc[test_data["labels"] == 1, "weights"])
        background_weight_test = np.sum(test_data.loc[test_data["labels"] == 0, "weights"])

        # Apply reweighting using .loc to avoid chained assignment
        train_data.loc[train_data["labels"] == 1, "weights"] *= (
                signal_weight / signal_weight_train
        )
        train_data.loc[train_data["labels"] == 0, "weights"] *= (
                background_weight / background_weight_train
        )
        test_data.loc[test_data["labels"] == 1, "weights"] *= (
                signal_weight / signal_weight_test
        )
        test_data.loc[test_data["labels"] == 0, "weights"] *= (
                background_weight / background_weight_test
        )

    return train_data, test_data


def prepare_balanced_datasets(
        training_df: pd.DataFrame,
        valid_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training and validation DataFrames by popping labels and weights,
    and balancing training weights by class.

    Args:
        training_df (pd.DataFrame): Training dataset with "labels" and "weights" columns.
        valid_df (pd.DataFrame): Validation dataset with "labels" and "weights" columns.

    Returns:
        Tuple of:
            - training_df: DataFrame without "labels" and "weights"
            - train_weights: Series of adjusted training weights
            - train_labels: Series of training labels
            - valid_df: DataFrame without "labels" and "weights"
            - valid_weights: Series of validation weights (unaltered)
            - valid_labels: Series of validation labels
    """
    training_df = training_df.copy()
    valid_df = valid_df.copy()

    # Pop weights and labels
    train_weights = training_df.pop("weights")
    train_labels = training_df.pop("labels")

    valid_weights = valid_df.pop("weights")
    valid_labels = valid_df.pop("labels")

    # Compute class weights
    class_weights_train = (
        train_weights[train_labels == 0].sum(),
        train_weights[train_labels == 1].sum(),
    )

    # Reweight training weights to balance classes
    for i in range(len(class_weights_train)):
        train_weights[train_labels == i] *= (
                max(class_weights_train) / class_weights_train[i]
        )

    return (
        training_df, train_weights, train_labels,
        valid_df, valid_weights, valid_labels
    )


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

        self.name = "model_XGB"
        self.model = BoostedDecisionTree()
        module_file = current_file + f"/{self.name}.json"
        if os.path.exists(module_file):
            self.model.load(module_file)
            self.re_train = False  # if model is already trained, no need to retrain

        print("Model is ", self.name)

        self.stat_analysis = StatisticalAnalysis(
            self.model, stat_only=STAT_ONLY, bins=20, systematics=self.systematics,
            fixed_syst=SYST_FIXED
        )

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

        training_df, valid_df = None, None
        if self.re_train or self.re_compute:
            train_set = self.get_train_set()
            train_set.pop("detailed_labels")

            # train : validation : template = 3 : 1 : 6
            temp_set, holdout_set = train_test_split(
                train_set, test_size=0.6, random_state=42, reweight=True
            )
            temp_set = temp_set.reset_index(drop=True)
            training_set, valid_set = train_test_split(
                temp_set, test_size=0.2, random_state=42, reweight=True
            )

            del train_set

            def print_set_info(name, dataset):
                print(f"{name} Set:")
                print(f"{'-' * len(name)} ")
                print(f"  Data Shape:          {dataset.shape}")
                print(
                    f"  Sum Signal Weights:  {dataset['weights'][dataset['labels'] == 1].sum():.2f}"
                )
                print(
                    f"  Sum Background Weights: {dataset['weights'][dataset['labels'] == 0].sum():.2f}"
                )

            """
            The systematics code does the following
            1. Apply systematics 
            2. Apply post-systematics cuts to the data
            3. Compute Dervied features

            NOTE:
            Without this transformation, the data will not be representative of the pseudo-experiments
            """
            training_df = self.systematics(training_set)  # to get derived and post cuts
            valid_df = self.systematics(valid_set)  # to get derived and post cuts

            print_set_info("Training", training_df)
            print_set_info("Validation", valid_df)
            print_set_info("Holdout (For Statistical Template)", holdout_set)

            if self.re_train:
                (
                    train_X, train_weights, train_labels,
                    valid_X, valid_weights, valid_labels
                ) = prepare_balanced_datasets(training_df, valid_df)

                print("Training", train_X.shape, f"Signal Weights: {train_weights[train_labels == 1].sum()}, Background Weights: {train_weights[train_labels == 0].sum()}")
                print("Validation", valid_X.shape, f"Signal Weights: {valid_weights[valid_labels == 1].sum()}, Background Weights: {valid_weights[valid_labels == 0].sum()}")

                self.model.fit(
                    train_X,
                    train_labels,
                    weights=train_weights,
                    valid_set=[
                        valid_X,
                        valid_labels,
                        valid_weights,
                    ],
                )

                self.model.save(current_file + "/" + self.name)

            self.stat_analysis.calculate_saved_info(holdout_set, saved_info_file_dir)

        self.stat_analysis.alpha_function()

        def predict_and_analyze(
                dataset_name, data_set, fig_name
        ):
            weights = data_set.pop("weights")
            labels = data_set.pop("labels")

            print(f"Sum Signal Weights: {weights[labels == 1].sum()}")
            print(f"Sum Background Weights: {weights[labels == 0].sum()}")

            score = self.model.predict(data_set)
            results = self.stat_analysis.compute_mu(
                score,
                weights,
                plot=fig_name,
                file_path=saved_info_file_dir,
            )

            print(f"{dataset_name} Results:")
            print(f"{'-' * len(dataset_name)} Results:")
            for key, value in results.items():
                print(f"\t{key} : {value}")
            print("\n")

        if self.re_train or self.re_compute:
            # Predict and analyze for each set
            datasets = [
                ("Training", training_df, "train_mu"),
                ("Validation", valid_df, "valid_mu"),
                ("Holdout", self.systematics(holdout_set) , "holdout_mu"),
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
