# ------------------------------
# Dummy Sample Submission
# ------------------------------
import pandas as pd
import matplotlib.pyplot as plt

XGBOOST = True
TENSORFLOW = False
TORCH = False

from statistical_analysis import StatisticalAnalysis
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

        # First, split the data into two parts: 1/2 and 1/2
        train_set, temp_set = train_test_split(self.train_set, test_size=0.5, random_state=42)

        # Now split the temp_set into validation and holdout sets (statistical template set) with equal size
        temp_set['data'] = temp_set['data'].reset_index(drop=True)
        valid_set, holdout_set = train_test_split(temp_set, test_size=0.5, random_state=42)

        self.training_set = train_set
        self.valid_set = valid_set
        self.holdout_set = holdout_set
        self.weights_summary = {}

        del self.train_set

        def print_set_info(name, dataset):

            summary = {
                "sum_of_weights_s": dataset["weights"][dataset["labels"] == 1].sum(),
                "sum_of_weights_b": dataset["weights"][dataset["labels"] == 0].sum(),
                "sum_of_weights": dataset["weights"].sum(),
            }

            print(f"{name} Set:")
            print(f"{'-' * len(name)} ")
            print(f"  Data Shape:          {dataset['data'].shape}")
            print(f"  Labels Shape:        {dataset['labels'].shape}")
            print(f"  Weights Shape:       {dataset['weights'].shape}")
            print(f"  Sum Signal Weights:  {summary['sum_of_weights_s']:.2f}")
            print(f"  Sum Background Weights: {summary['sum_of_weights_b']:.2f}")
            print("\n")

            return summary

        self.weights_summary['Training'] = print_set_info("Training", self.training_set)
        self.weights_summary['Validation'] = print_set_info("Validation", self.valid_set)
        self.weights_summary['Holdout'] = print_set_info("Holdout (For Statistical Template)", self.holdout_set)

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

        self.stat_analysis = StatisticalAnalysis(self.model, self.holdout_set, stat_only=False)

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

        if self.re_train:

            balanced_set = self.balance_set()

            if XGBOOST:
                self.model.fit(
                    balanced_set["data"], balanced_set["labels"], balanced_set["weights"],
                    valid_set=[self.valid_set["data"], self.valid_set["labels"], self.valid_set["weights"]],
                )
            else:
                self.model.fit(
                    balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
                )
            self.model.save(current_file + "/" + self.name)

        saved_info_file = current_file + "/saved_info_" + self.name + ".pkl"
        if os.path.exists(saved_info_file):
            self.stat_analysis.load(saved_info_file)
        else:
            self.stat_analysis.calculate_saved_info()
            self.stat_analysis.save(saved_info_file)

        def predict_and_analyze(dataset_name, data_set, fig_name, stat_only, syst_settings, scale):
            score = self.model.predict(data_set["data"])
            results = self.stat_analysis.compute_mu(
                score,
                data_set["weights"],
                plot=fig_name,
                stat_only=stat_only,
                syst_fixed_setting=syst_settings,
                return_distribution=True,
                global_scale=scale,
            )

            print(f"{dataset_name} Results:")
            print(f"{'-' * len(dataset_name)} Results:")
            for key, value in results.items():
                if key == "distribution":
                    print(value.to_string())
                else:
                    print(f"\t{key} : {value}")
            print("\n")

            return results

        # Predict and analyze for each set
        datasets = [
            ("Training", self.training_set, "train_mu"),
            ("Validation", self.valid_set, "valid_mu"),
            ("Holdout", self.holdout_set, "holdout_mu"),
        ]

        results = {}
        for name, dataset, plot_name in datasets:
            scale = self.weights_summary[name]["sum_of_weights"] / self.weights_summary["Holdout"]["sum_of_weights"]
            results[name] = predict_and_analyze(name, dataset, plot_name, stat_only=stat_only, syst_settings=syst_settings, scale=scale)

        plot_train_valid_holdout(
            results['Training'], results['Validation'], results['Holdout'],
            save_name=f"plots/train_valid_holdout.png"
        )

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

    def predict(self, test_set, stat_only: bool = None, syst_settings: dict[str, float] = None):
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

        predictions = self.model.predict(test_data)

        result = self.stat_analysis.compute_mu(
            predictions,
            test_weights,
            stat_only=stat_only,
            syst_fixed_setting=syst_settings
        )

        print("Test Results: ", result)

        return result


def plot_train_valid_holdout(train_data, valid_data, holdout_data, save_name: str = None):
    # Creating DataFrames
    df1 = pd.DataFrame(train_data['distribution'])
    df2 = pd.DataFrame(valid_data['distribution'])
    df3 = pd.DataFrame(holdout_data['distribution'])

    mu1 = train_data['mu_hat']
    mu2 = valid_data['mu_hat']
    mu3 = holdout_data['mu_hat']

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 15), sharex='col')

    # Plot obsData, template_s, template_b for each dataframe in the same plot
    ax1.plot(df1.index, df1['obsData'], label='Train: obsData', marker='o', color='red')
    ax1.plot(df1.index, df1['template_s'], label='Train: template_S', marker='x', color='red')
    ax1.plot(df1.index, df1['template_b'], label='Train: template_B', marker='^', color='red')

    ax1.plot(df2.index, df2['obsData'], label='Valid: obsData', marker='o', linestyle='--', color='blue')
    ax1.plot(df2.index, df2['template_s'], label='Valid: template_S', marker='x', linestyle='--', color='blue')
    ax1.plot(df2.index, df2['template_b'], label='Valid: template_B', marker='^', linestyle='--', color='blue')

    ax1.plot(df3.index, df3['obsData'], label='Holdout: obsData', marker='o', linestyle='-.', color='green')
    ax1.plot(df3.index, df3['template_s'], label='Holdout: template_S', marker='x', linestyle='-.', color='green')
    ax1.plot(df3.index, df3['template_b'], label='Holdout: template_B', marker='^', linestyle='-.', color='green')

    ax1.set_ylabel('MVA score')
    ax1.set_yscale('log')
    ax1.legend()

    # Plot ratios
    ratio_df1 = df1['obsData'] / (df1['template_s'] + df1['template_b'])
    ratio_df2 = df2['obsData'] / (df2['template_s'] + df2['template_b'])
    ratio_df3 = df3['obsData'] / (df3['template_s'] + df3['template_b'])

    ax2.plot(df1.index, ratio_df1, label=f'Train: $\\mu={mu1:.2f}$', marker='o', color='red')
    ax2.plot(df2.index, ratio_df2, label=f'Valid: $\\mu={mu2:.2f}$', marker='x', linestyle='--', color='blue')
    ax2.plot(df3.index, ratio_df3, label=f'Holdout: $\\mu={mu3:.2f}$', marker='^', linestyle='-.', color='green')

    ax2.axhline(y=1.0, color='grey', linestyle='--', alpha = 0.25)
    ax2.set_ylabel('obsData / (S + B)')
    ax2.legend()

    obs_ratio_df1 = df1['obsData'] / df3['obsData']
    obs_ratio_df2 = df2['obsData'] / df3['obsData']
    s_ratio_df1 = df1['template_s'] / df3['template_s']
    s_ratio_df2 = df2['template_s'] / df3['template_s']
    b_ratio_df1 = df1['template_b'] / df3['template_b']
    b_ratio_df2 = df2['template_b'] / df3['template_b']

    ax3.plot(df1.index, obs_ratio_df1, label='Train: Data', marker='o', color='red')
    ax3.plot(df1.index, obs_ratio_df2, label='Valid: Data', marker='o', color='blue')
    ax3.plot(df1.index, s_ratio_df1, label='Train: S', marker='x', linestyle='--', color='red')
    ax3.plot(df1.index, s_ratio_df2, label='Valid: S', marker='x', linestyle='--', color='blue')
    ax3.plot(df1.index, b_ratio_df1, label='Train: B', marker='^', linestyle='-.', color='red')
    ax3.plot(df1.index, b_ratio_df2, label='Valid: B', marker='^', linestyle='-.', color='blue')

    ax3.axhline(y=1.0, color='grey', linestyle='--', alpha = 0.25)
    ax3.set_xlabel('Bins')
    ax3.set_ylabel('(Train, Valid) / Holdout')
    ax3.legend()

    # Get current y-limits
    current_min_y1, current_max_y1 = ax1.get_ylim()
    current_min_y2, current_max_y2 = ax2.get_ylim()

    # Set new y-limits to 1.5 times the current maximum y-value
    ax1.set_ylim(current_min_y1, current_max_y1 * 10)
    ax2.set_ylim(current_min_y2, current_max_y2 * 1.15)

    plt.tight_layout()
    plt.show()
    if save_name is not None:
        plt.savefig(save_name)


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
