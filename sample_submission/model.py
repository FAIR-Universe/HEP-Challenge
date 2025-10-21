#!/usr/bin/env python3


from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from LHC_statistics import compute_mu, calculate_saved_info
import numpy as np

class Model:

    def __init__(self, get_train_set=None, systematics=None):

        self.get_train_set = get_train_set
        self.systematics = systematics
                
        self.model = XGBClassifier(
            n_estimators=300, eval_metric="logloss"
        )
        self.name = "model_XGB"
        self.scaler = StandardScaler()

        self.N_events_total = 15_000_000

        

    def fit(self):
        """
        Trains the model.

        Params:
            None

        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model.The saved information is used
            to compute the train results.

        Returns:
            None
        """

        from visualization import stacked_histogram, Dataset_visualise, roc_curve_wrapper
        
        try :
            data_df = self.get_train_set(train_size=self.N_events_total)
        except Exception as e :
            print(e)
            data_df = self.get_train_set(train_size=self.N_events_total)
        
        #Divide the dataset
        
        training_df, temp_df = train_test_split(
            data_df, test_size= (2 / 3) , random_state=42, reweight=True
        )
        
        holdout_df, test_df = train_test_split(
            temp_df, test_size=(1/2), random_state=42, reweight=True
        )
        
        print("Training set size: ", training_df.shape)
        print("Holdout set size: ", holdout_df.shape)
        print("Test set size: ", test_df.shape)
        
        train_vis = Dataset_visualise(
            data_set=training_df,
            name="Training Set",
            columns=[
                "PRI_met",
                "PRI_had_pt",
                "DER_mass_vis"],
        )
        
        train_vis.examine_dataset()

                
        training_df = self.systematics(training_df)
        
        training_set = {
            "labels": training_df.pop("labels"),
            "weights": training_df.pop("weights"),
            "detailed_labels": training_df.pop("detailed_labels"),
            "data": training_df
        }
        
        
        balanced_set = balance_set(training_set)
                                
        self.scaler.fit(balanced_set["data"])
        
        X_train_data = self.scaler.transform(balanced_set["data"])
        self.model.fit(X_train_data,balanced_set["labels"], balanced_set["weights"])
        
        
        del balanced_set
        
        holdout_df = self.systematics(holdout_df)

        
        holdout_set = {
            "labels": holdout_df.pop("labels"),
            "weights": holdout_df.pop("weights"),
            "detailed_labels": holdout_df.pop("detailed_labels"),
            "data": holdout_df
        }
                
        
        X_holdout = self.scaler.transform(holdout_set["data"])

        holdout_score = self.model.predict_proba(X_holdout)[:, 1]

        self.saved_info = calculate_saved_info(holdout_score, holdout_set)


        holdout_results = compute_mu(
            holdout_score, holdout_set["weights"], self.saved_info
        )
        
        roc_curve_wrapper(holdout_score, holdout_set["labels"],holdout_set["weights"], plot_label="Holdout ROC curve")

        
        stacked_histogram(detailed_label=holdout_set["detailed_labels"],
            field=holdout_score,
            weights=holdout_set["weights"],
            target=holdout_set["labels"],
            y_scale="linear",
            plot_label="Holdout Score No weights",
            weighted=False
        )
        
        stacked_histogram(detailed_label=holdout_set["detailed_labels"],
            field=holdout_score,
            weights=holdout_set["weights"],
            target=holdout_set["labels"],
            y_scale="log",
            plot_label="Holdout Score",
        )

            
        print("Holdout Results: ")
        for key in holdout_results.keys():
            print("\t", key, " : ", holdout_results[key])

        test_vis = Dataset_visualise(
            data_set=test_df,
            name="Training Set",
            columns=[
                "PRI_met",
                "PRI_had_pt",
                "DER_mass_vis"],
        )
        
        test_vis.examine_dataset()    
        bootstraped_test_df = test_df.copy()

        test_df = self.systematics(test_df)

        test_set ={
            "labels": test_df.pop("labels"),
            "weights": test_df.pop("weights"),
            "detailed_labels": test_df.pop("detailed_labels"),
            "data": test_df
        }

        random_state = np.random.RandomState(42)
        mu_test =random_state.uniform(0.1, 3)
                        
        test_set["weights"][test_set["labels"] == 1] *= mu_test
                
        X_test =  self.scaler.transform(test_set["data"])
        
        test_score = self.model.predict_proba(X_test)[:, 1]
        test_results = compute_mu(
            test_score, test_set["weights"], self.saved_info
        )

        mu_test_hat = test_results["mu_hat"]
        
        print("Test Results: ")
        for key in test_results.keys():
            print("\t", key, " : ", test_results[key])
        
        stacked_histogram(detailed_label=holdout_set["detailed_labels"],
            field=holdout_score,
            weights=holdout_set["weights"],
            target=holdout_set["labels"],
            mu_hat=mu_test_hat,
            pseudo_field=test_score,
            pseudo_weight=test_set["weights"],
            y_scale="log",
            plot_label="BDT Score",
            seperate_signal=True,
            path_to_figures="./"
        )
                  

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



from sklearn.model_selection import train_test_split as sk_train_test_split
import pandas as pd

def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):

    train_set, test_set = sk_train_test_split(
        data_set, test_size=test_size, random_state=random_state
    )


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
