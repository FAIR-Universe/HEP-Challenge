import os
from sys import path
import numpy as np
import pandas as pd
from math import sqrt, log
from lightgbm import LGBMRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

# ------------------------------
# Absolute path to submission dir
# ------------------------------
submissions_dir = os.path.dirname(os.path.abspath(__file__))
path.append(submissions_dir)

from bootstrap import *
from systematics import postprocess

# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps


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

            test_sets:
                unlabelled test sets

            systematics:
                systematics class

        Returns:
            None
        """

        # Set class variables from parameters
        self.train_set = train_set
        self.systematics = systematics

        # Intialize class variables
        self.validation_sets = None
        self.theta_candidates = np.arange(0.5, 0.99, 0.01)
        self.best_theta = 0.9
        self.scaler = StandardScaler()
        self.scaler_tes = StandardScaler()

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """

        self._generate_validation_sets()
        self._init_model()
        self._train()
        self._choose_theta()
        self.mu_hat_calc()
        self._validate()
        self._compute_validation_result()

    def predict(self, test_set):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        test_df = test_set['data']
        test_df = self.scaler.transform(test_df)
        Y_hat_test = self._predict(test_df, self.best_theta)

        weights_train = self.train_set["weights"].copy()
        weights_test = test_set["weights"].copy()

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

    def _init_model(self):
        print("[*] - Intialize Baseline Model (XGBoostRegressor)")

        lgbm = LGBMRegressor(learning_rate=0.05, n_estimators=100,n_jobs=1, random_state=42)
        self.model = MultiOutputRegressor(lgbm, n_jobs=1)

    def _generate_validation_sets(self):
        print("[*] - Generating Validation sets")

        # Calculate the sum of weights for signal and background in the original dataset
        signal_weights = self.train_set["weights"][self.train_set["labels"] == 1].sum()
        background_weights = self.train_set["weights"][self.train_set["labels"] == 0].sum()

        # Split the data into training and validation sets while preserving the proportion of samples with respect to the target variable
        train_df, valid_df, train_labels, valid_labels, train_weights, valid_weights = train_test_split(
            self.train_set["data"],
            self.train_set["labels"],
            self.train_set["weights"],
            test_size=0.05,
            stratify=self.train_set["labels"]
        )

        train_df, mu_calc_set_df, train_labels, mu_calc_set_labels, train_weights, mu_calc_set_weights = train_test_split(
            train_df,
            train_labels,
            train_weights,
            test_size=0.5,
            shuffle=True,
            stratify=train_labels
        )

        # Calculate the sum of weights for signal and background in the training and validation sets
        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()
        valid_signal_weights = valid_weights[valid_labels == 1].sum()
        valid_background_weights = valid_weights[valid_labels == 0].sum()
        mu_calc_set_signal_weights = mu_calc_set_weights[mu_calc_set_labels == 1].sum()
        mu_calc_set_background_weights = mu_calc_set_weights[mu_calc_set_labels == 0].sum()

        # Balance the sum of weights for signal and background in the training and validation sets
        train_weights[train_labels == 1] *= signal_weights / train_signal_weights
        train_weights[train_labels == 0] *= background_weights / train_background_weights
        valid_weights[valid_labels == 1] *= signal_weights / valid_signal_weights
        valid_weights[valid_labels == 0] *= background_weights / valid_background_weights
        mu_calc_set_weights[mu_calc_set_labels == 1] *= signal_weights / mu_calc_set_signal_weights
        mu_calc_set_weights[mu_calc_set_labels == 0] *= background_weights / mu_calc_set_background_weights

        self.train_df = train_df

        self.train_set = {
            "data": train_df,
            "labels": train_labels,
            "weights": train_weights,
            "settings": self.train_set["settings"]
        }

        self.eval_set = [(self.train_set['data'], self.train_set['labels']), (valid_df.to_numpy(), valid_labels)]
        
        train_df = train_df.copy()
        train_df["weights"] = train_weights
        train_df["labels"] = train_labels
        train_df = postprocess(train_df)

        train_weights = train_df.pop('weights')
        train_labels = train_df.pop('labels')
        

        mu_calc_set_df = mu_calc_set_df.copy()
        mu_calc_set_df["weights"] = mu_calc_set_weights
        mu_calc_set_df["labels"] = mu_calc_set_labels
        mu_calc_set_df = postprocess(mu_calc_set_df)

        mu_calc_set_weights = mu_calc_set_df.pop('weights')
        mu_calc_set_labels = mu_calc_set_df.pop('labels')


        self.mu_calc_set = {
                "data": mu_calc_set_df,
                "labels": mu_calc_set_labels,
                "weights": mu_calc_set_weights
            }

        self.validation_sets = []
        # Loop 10 times to generate 10 validation sets
        for i in range(0, 20):
            tes = round(np.random.uniform(0.9, 1.10), 2)
            # apply systematics
            valid_df_temp = valid_df.copy()
            valid_df_temp["weights"] = valid_weights
            valid_df_temp["labels"] = valid_labels

            valid_with_systematics_temp = self.systematics(
                data=valid_df_temp,
                tes=tes
            ).data

            valid_labels_temp = valid_with_systematics_temp.pop('labels')
            valid_weights_temp = valid_with_systematics_temp.pop('weights')
            valid_with_systematics = valid_with_systematics_temp.copy()

            self.validation_sets.append({
                "data": valid_with_systematics,
                "labels": valid_labels_temp,
                "weights": valid_weights_temp,
                "settings": self.train_set["settings"],
                "tes": tes
            })
            del valid_with_systematics_temp
            del valid_df_temp


        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()
        valid_signal_weights = valid_weights[valid_labels == 1].sum()
        valid_background_weights = valid_weights[valid_labels == 0].sum()
        mu_calc_set_signal_weights = mu_calc_set_weights[mu_calc_set_labels == 1].sum()
        mu_calc_set_background_weights = mu_calc_set_weights[mu_calc_set_labels == 0].sum()

        print(f"[*] --- original signal: {signal_weights} --- original background: {background_weights}")
        print(f"[*] --- train signal: {train_signal_weights} --- train background: {train_background_weights}")
        print(f"[*] --- valid signal: {valid_signal_weights} --- valid background: {valid_background_weights}")
        print(f"[*] --- mu_calc_set signal: {mu_calc_set_signal_weights} --- mu_calc_set background: {mu_calc_set_background_weights}")

    def _train(self):

        tes_sets = []

        for i in range(0, 10):
            tes_set = self.train_df.copy()
            tes_set = pd.DataFrame(tes_set)
            tes_set["weights"] = self.train_set["weights"]
            tes_set["labels"] = self.train_set["labels"]

            # adding systematics to the tes set
            # Extract the TES information from the JSON file
            tes = round(np.random.uniform(0.9, 1.10), 2)
            # tes = 1.0

            syst_set = tes_set.copy()
            data_syst = self.systematics(
                data=syst_set,
                verbose=0,
                tes=tes
            ).data

            data_syst = data_syst.round(3)
            tes_set = data_syst.copy()
            tes_set['tes'] = (tes*10)*2
            tes_sets.append(tes_set)
            del data_syst
            del tes_set

        tes_sets_df = pd.concat(tes_sets)

        train_tes_data = shuffle(tes_sets_df)

        tes_label_1 = train_tes_data.pop('tes')
        tes_label_2 = train_tes_data.pop('labels')
        tes_label = [tes_label_1, tes_label_2]
        tes_label = np.array(tes_label).T
        tes_weights = train_tes_data.pop('weights')

        weights_train = tes_weights.copy()

        class_weights_train = (weights_train[tes_label_2 == 0].sum(), weights_train[tes_label_2 == 1].sum())

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[tes_label_2 == i] *= max(class_weights_train) / class_weights_train[i]
            # test dataset : increase test weight to compensate for sampling

        print("[*] --- Training Model")
        train_tes_data = self.scaler.fit_transform(train_tes_data)

        self._fit(train_tes_data, tes_label, weights_train)

        print("[*] --- Predicting Train set")
        self.train_set['predictions'] = (self.train_set['data'], self.best_theta)

        self.train_set['score'] = self._return_score(self.train_set['data'])

    def _fit(self, X, y, w):
        print("[*] --- Fitting Model")
        self.model.fit(X, y, sample_weight=w)

    def _return_score(self, X):
        y_predict = self.model.predict(X)[:, 1]
        return y_predict

    def _predict(self, X, theta):
        Y_predict = self._return_score(X)
        predictions = np.where(Y_predict > theta, 1, 0)
        return predictions

    def N_calc_2(self, weights, n=100):
        total_weights = []
        for i in range(n):
            bootstrap_weights = bootstrap(weights=weights, seed=42+i)
            bootstrap_weights_array = np.array(bootstrap_weights)
            effective_number = (bootstrap_weights_array).sum()
            total_weights.append(effective_number)
        n_calc_array = np.array(total_weights)
        return n_calc_array

    def mu_hat_calc(self):

        self.mu_calc_set['data'] = self.scaler.transform(self.mu_calc_set['data'])
        Y_hat_mu_calc_set = self._predict(self.mu_calc_set['data'], self.best_theta)
        Y_mu_calc_set = self.mu_calc_set['labels']
        weights_mu_calc_set = self.mu_calc_set['weights']

        # compute gamma_roi
        weights_mu_calc_set_signal = weights_mu_calc_set[Y_mu_calc_set == 1]
        weights_mu_calc_set_bkg = weights_mu_calc_set[Y_mu_calc_set == 0]

        Y_hat_mu_calc_set_signal = Y_hat_mu_calc_set[Y_mu_calc_set == 1]
        Y_hat_mu_calc_set_bkg = Y_hat_mu_calc_set[Y_mu_calc_set == 0]

        self.gamma_roi = (weights_mu_calc_set_signal[Y_hat_mu_calc_set_signal == 1]).sum()

        # compute beta_roi
        self.beta_roi = (weights_mu_calc_set_bkg[Y_hat_mu_calc_set_bkg == 1]).sum()
        if self.gamma_roi == 0:
            self.gamma_roi = EPSILON

    def get_meta_validation_set(self):

        meta_validation_data = []
        meta_validation_labels = []
        meta_validation_weights = []

        for valid_set in self.validation_sets:
            meta_validation_data.append(valid_set['data'])
            meta_validation_labels = np.concatenate((meta_validation_labels, valid_set['labels']))
            meta_validation_weights = np.concatenate((meta_validation_weights, valid_set['weights']))

        return {
            'data': pd.concat(meta_validation_data),
            'labels': meta_validation_labels,
            'weights': meta_validation_weights
        }

    def _choose_theta(self):

        print("[*] Choose best theta")
        meta_validation_set = self.get_meta_validation_set()
        theta_sigma_squared = []

        # Loop over theta candidates
        # try each theta on meta-validation set
        # choose best theta
        for theta in self.theta_candidates:
            meta_validation_set_df = self.scaler.transform(meta_validation_set["data"])    
            # Get predictions from trained model
            Y_hat_valid = self._predict(meta_validation_set_df, theta)
            Y_valid = meta_validation_set["labels"]

            weights_valid = meta_validation_set["weights"].copy()

            # get region of interest
            nu_roi = (weights_valid[Y_hat_valid == 1]).sum()/10

            weights_valid_signal = weights_valid[Y_valid == 1]  
            weights_valid_bkg = weights_valid[Y_valid == 0]

            Y_hat_valid_signal = Y_hat_valid[Y_valid == 1]  
            Y_hat_valid_bkg = Y_hat_valid[Y_valid == 0] 

            # compute gamma_roi
            gamma_roi = (weights_valid_signal[Y_hat_valid_signal == 1]).sum()/10

            # compute beta_roi
            beta_roi = (weights_valid_bkg[Y_hat_valid_bkg == 1]).sum()/10

            # Compute sigma squared mu hat
            sigma_squared_mu_hat = nu_roi/np.square(gamma_roi)

            # get N_ROI from predictions
            theta_sigma_squared.append(sigma_squared_mu_hat)

            print(f"\n[*] --- theta: {theta}--- nu_roi: {nu_roi} --- beta_roi: {beta_roi} --- gamma_roi: {gamma_roi} --- sigma squared: {sigma_squared_mu_hat}")

        # Choose theta with min sigma squared
        try:
            index_of_least_sigma_squared = np.nanargmin(theta_sigma_squared)
        except:
            print("[!] - WARNING! All sigma squared are nan")
            index_of_least_sigma_squared = np.argmin(theta_sigma_squared)

        self.best_theta = self.theta_candidates[index_of_least_sigma_squared]
        print(f"[*] --- Best theta : {self.best_theta}")

    def _validate(self):
        for valid_set in self.validation_sets:
            valid_set['data'] = self.scaler.transform(valid_set['data'])
            valid_set['predictions'] = self._predict(valid_set['data'], self.best_theta)
            valid_set['score'] = self._return_score(valid_set['data'])

    def _compute_validation_result(self):
        print("[*] - Computing Validation result")

        self.validation_delta_mu_hats = []
        for valid_set in self.validation_sets:
            Y_hat_train = self.train_set["predictions"]
            Y_train = self.train_set["labels"]
            Y_hat_valid = valid_set["predictions"]
            Y_valid = valid_set["labels"]
            Score_train = self.train_set["score"]
            Score_valid = valid_set["score"]

            # print(f"[*] --- PRI_had_pt : {valid_set['had_pt']}")
            # del Score_valid
            weights_train = self.train_set["weights"].copy()
            weights_valid = valid_set["weights"].copy()

            signal_valid = weights_valid[Y_valid == 1]
            background_valid = weights_valid[Y_valid == 0]

            Y_hat_valid_signal = Y_hat_valid[Y_valid == 1]
            Y_hat_valid_bkg = Y_hat_valid[Y_valid == 0]

            signal = signal_valid[Y_hat_valid_signal == 1].sum()
            background = background_valid[Y_hat_valid_bkg == 1].sum()

            # get n_roi
            n_roi = self.N_calc_2(weights_valid[Y_hat_valid == 1])

            mu_hat = ((n_roi - self.beta_roi)/self.gamma_roi).mean()
            # get region of interest
            nu_roi = self.beta_roi + self.gamma_roi

            gamma_roi = self.gamma_roi

            # compute beta_roi
            beta_roi = self.beta_roi
            if gamma_roi == 0:
                gamma_roi = EPSILON

            # Compute mu_hat

            # Compute delta mu hat (absolute value)
            delta_mu_hat = np.abs(valid_set["settings"]["ground_truth_mu"] - mu_hat)

            self.validation_delta_mu_hats.append(delta_mu_hat)

            print(f"[*] --- mu: {np.round(valid_set['settings']['ground_truth_mu'], 4)} --- mu_hat: {np.round(mu_hat, 4)} --- delta_mu_hat: {np.round(delta_mu_hat, 4)}")

        print(f"[*] --- validation delta_mu_hat (avg): {np.round(np.mean(self.validation_delta_mu_hats), 4)}")
