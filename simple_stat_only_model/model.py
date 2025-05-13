

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def calculate_saved_info(model, train_df,labels,weights):


    score = model.predict(train_df)

    print("score shape before threshold", score.shape)

    score = score.flatten() > 0.5
    score = score.astype(int)

    print("score shape after threshold", score.shape)

    gamma = np.sum(weights * score * labels) + 0.1

    beta = np.sum(weights * score * (1 - labels)) - 0.1

    saved_info = {"beta": beta, "gamma": gamma}

    print("saved_info", saved_info)

    return saved_info



def compute_mu(score, weight, saved_info):

    score = score.flatten() > 0.5
    score = score.astype(int)

    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    del_mu_stat = (
        np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    )
    del_mu_sys = abs(0.1 * mu)
    del_mu_tot = (1 / 2) * np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


from sklearn.model_selection import train_test_split as sk_train_test_split

def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):

    data = data_set.copy()

    full_size = len(data)

    print(f"Full size of the data is {full_size}")


    train_data, test_data = sk_train_test_split(
        data, test_size=test_size, random_state=random_state
    )


    if reweight is True:
        signal_weight = np.sum(data_set["weights"][data_set["labels"] == 1])
        background_weight = np.sum(data_set["weights"][data_set["labels"] == 0])
        signal_weight_train = np.sum(train_data["weights"][train_data["labels"] == 1])
        background_weight_train = np.sum(train_data["weights"][train_data["labels"] == 0])
        signal_weight_test = np.sum(test_data["weights"][test_data["labels"] == 1])
        background_weight_test = np.sum(test_data["weights"][test_data["labels"] == 0])

        train_data["weights"][train_data["labels"] == 1] = train_data["weights"][
            train_data["labels"] == 1
        ] * (signal_weight / signal_weight_train)
        test_data["weights"][test_data["labels"] == 1] = test_data["weights"][
            test_data["labels"] == 1
        ] * (signal_weight / signal_weight_test)

        train_data["weights"][train_data["labels"] == 0] = train_data["weights"][
            train_data["labels"] == 0
        ] * (background_weight / background_weight_train)
        test_data["weights"][test_data["labels"] == 0] = test_data["weights"][
            test_data["labels"] == 0
        ] * (background_weight / background_weight_test)

    return train_data, test_data


class Model:

    def __init__(self, get_train_set=None, systematics=None):

        self.get_train_set = get_train_set
        self.systematics = systematics
        self.re_train = True
                
        self.model = XGBClassifier(eval_metric=["error", "logloss", "rmse"],)
        self.name = "model_XGB"
        self.scaler = StandardScaler()
        

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


        train_set = self.get_train_set(train_size=50_000) # train_set is a dictionary with data, labels, and weights
        
        train_set.pop("detailed_labels")

        training_df, holdout_df = train_test_split(
            train_set, test_size=0.5, random_state=42, reweight=True
        )
        
        del train_set
        
        training_df = self.systematics(training_df)

        weights_train = training_df.pop("weights")
        train_labels = training_df.pop("labels")

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

        print(training_df.columns)
        print(training_df.shape)
                
        self.scaler.fit_transform(training_df)

        X_train_data = self.scaler.transform(training_df)
        self.model.fit(X_train_data, train_labels, weights_train)

        holdout_df = self.systematics(holdout_df)
        holdout_weights = holdout_df.pop("weights")
        holdout_labels = holdout_df.pop("labels")

        self.saved_info = calculate_saved_info(self.model, holdout_df, holdout_labels, holdout_weights)

        holdout_score = self.model.predict(holdout_df)
        holdout_results = compute_mu(
            holdout_score, holdout_weights, self.saved_info
        )
            
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

        test_data = self.scaler.transform(test_data)
        predictions = self.model.predict_proba(test_data)[:, 1]
    
        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)

        print("Test Results: ", result_mu_cal)

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result

