import os
from sys import path
import numpy as np
import pandas as pd
from math import sqrt, log
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
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
        self.validation_set = None
        self.theta_candidates = np.linspace(0.8, 1.0, 100)
        self.best_theta = 0.8
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
        # self.mu_hat_calc()
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

        print("[*] - Testing")
        test_df = test_set['data']
        test_df = self.scaler.transform(test_df)
        Y_hat_test = self._predict(test_df, self.best_theta)

        print("[*] - Computing Test result")
        weights_train = self.train_set["weights"].copy()
        weights_test = test_set["weights"].copy()

        print(f"[*] --- total weight test: {weights_test.sum()}") 
        print(f"[*] --- total weight train: {weights_train.sum()}")
        print(f"[*] --- total weight mu_cals_set: {self.mu_calc_set['weights'].sum()}")

        weight = weights_test*(Y_hat_test)
        # get n_roi

        mu_hat,mu_p16,mu_p84 = self._compute_result(weight)
        delta_mu_hat = np.abs(mu_p84 - mu_p16)

        print(f"[*] --- mu_hat: {mu_hat.mean()}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        return {
            "mu_hat": mu_hat.mean(),
            "delta_mu_hat": delta_mu_hat,
            "p16": mu_p16,
            "p84": mu_p84
        }

    def _init_model(self):
        print("[*] - Intialize Baseline Model (XBM Classifier Model)")

        self.model = XGBClassifier(
            tree_method="hist",
            use_label_encoder=False,
            eval_metric=['logloss', 'auc']
        )
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
            test_size=0.2,
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

        # valid_df = valid_df.copy()
        # valid_df["weights"] = valid_weights
        # valid_df["labels"] = valid_labels

        # valid_df = postprocess(valid_df)

        # valid_weights = valid_df.pop('weights')
        # valid_labels = valid_df.pop('labels')




        self.train_df = train_df

        self.train_set = {
            "data": train_df,
            "labels": train_labels,
            "weights": train_weights,
            "settings": self.train_set["settings"]
        }

        self.eval_set = [(self.train_set['data'], self.train_set['labels']), (valid_df.to_numpy(), valid_labels)]

        self.mu_calc_set = {
                "data": mu_calc_set_df,
                "labels": mu_calc_set_labels,
                "weights": mu_calc_set_weights
            }

        self.validation_sets = []
        for i in range(10):
            # Loop 10 times to generate 10 validation sets
            tes = round(np.random.uniform(0.9, 1.10), 2)
            # apply systematics
            valid_df_temp = valid_df.copy()
            valid_df_temp["weights"] = valid_weights
            valid_df_temp["labels"] = valid_labels

            # valid_with_systematics_temp = self.systematics(
            #     data=valid_df_temp,
            #     tes=tes
            # ).data
            valid_with_systematics_temp = postprocess(valid_df_temp)

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

        self.validation_set = self.validation_sets[1]

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


        weights_train = self.train_set["weights"].copy()
        train_labels = self.train_set["labels"].copy()
        train_data = self.train_set["data"].copy()
        class_weights_train = (weights_train[train_labels == 0].sum(), weights_train[train_labels == 1].sum())

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= max(class_weights_train) / class_weights_train[i]
            # test dataset : increase test weight to compensate for sampling

        print("[*] --- Training Model")
        train_data = self.scaler.fit_transform(train_data)

        print("[*] --- shape of train tes data", train_data.shape)

        self._fit(train_data, train_labels, weights_train)

        print("[*] --- Predicting Train set")
        self.train_set['predictions'] = (self.train_set['data'], self.best_theta)

        self.train_set['score'] = self._return_score(self.train_set['data'])

        auc_train = roc_auc_score(
            y_true=self.train_set['labels'],
            y_score=self.train_set['score'],
            sample_weight=self.train_set['weights']
        )
        print(f"[*] --- AUC train : {auc_train}")

    def _fit(self, X, y, w):
        print("[*] --- Fitting Model")
        self.model.fit(X, y, sample_weight=w)

    def _return_score(self, X):
        y_predict = self.model.predict(X)
        return y_predict

    def _predict(self, X, theta):
        Y_predict = self._return_score(X)
        predictions = (Y_predict > theta).astype(int)
        return predictions

    def N_calc_2(self, weights, n=1000):
        total_weights = []
        for i in range(n):
            bootstrap_weights = bootstrap(weights=weights, seed=42+i)
            total_weights.append(np.array(bootstrap_weights).sum())
        n_calc_array = np.array(total_weights)
        return n_calc_array


    # def mu_hat_calc(self):  

    #     self.mu_calc_set['data'] = self.scaler.transform(self.mu_calc_set['data'])
    #     Y_hat_mu_calc_set = self._predict(self.mu_calc_set['data'], self.best_theta)
    #     Y_mu_calc_set = self.mu_calc_set['labels']
    #     weights_mu_calc_set = self.mu_calc_set['weights']

    #     # compute gamma_roi
    #     weights_mu_calc_set_signal = weights_mu_calc_set[Y_mu_calc_set == 1]
    #     weights_mu_calc_set_bkg = weights_mu_calc_set[Y_mu_calc_set == 0]

    #     Y_hat_mu_calc_set_signal = Y_hat_mu_calc_set[Y_mu_calc_set == 1]
    #     Y_hat_mu_calc_set_bkg = Y_hat_mu_calc_set[Y_mu_calc_set == 0]

    #     self.gamma_roi = (weights_mu_calc_set_signal[Y_hat_mu_calc_set_signal == 1]).sum()

    #     # compute beta_roi
    #     self.beta_roi = (weights_mu_calc_set_bkg[Y_hat_mu_calc_set_bkg == 1]).sum()
    #     if self.gamma_roi == 0:
    #         self.gamma_roi = EPSILON

    def amsasimov_x(self, s, b):
        '''
        This function calculates the Asimov crossection significance for a given number of signal and background events.
        Parameters: s (float) - number of signal events

        Returns:    float - Asimov crossection significance
        '''

        if b <= 0 or s <= 0:
            return 0
        try:
            return s/sqrt(s+b)
        except ValueError:
            print(1+float(s)/b)
            print(2*((s+b)*log(1+float(s)/b)-s))
        # return s/sqrt(s+b)

    def del_mu_stat(self, s, b):
        '''
        This function calculates the statistical uncertainty on the signal strength.
        Parameters: s (float) - number of signal events
                    b (float) - number of background events

        Returns:    float - statistical uncertainty on the signal strength

        '''
        return (np.sqrt(s + b)/s)

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

        meta_validation_set = self.validation_set
        val_min = 1
        # Loop over theta candidates
        # try each theta on meta-validation set
        # choose best theta
        for theta in self.theta_candidates:
            meta_validation_set_df = self.scaler.transform(meta_validation_set["data"])    
            # Get predictions from trained model


            # get region of interest

            # predict probabilities for holdout
            X_holdout_sc = self.scaler.transform(self.mu_calc_set['data'])
            w_holdout = self.mu_calc_set['weights']
            y_holdout = self.mu_calc_set['labels']
            y_pred = self._predict(X_holdout_sc, theta)
            

            gamma_roi = (w_holdout*(y_pred * y_holdout)).sum()
            beta_roi = (w_holdout*(y_pred * (1-y_holdout))).sum()


            Y_hat_valid = self._predict(meta_validation_set_df, theta)
            weights_valid = meta_validation_set["weights"].copy() 

            weight = weights_valid*(Y_hat_valid)

            mu_scan = np.linspace(0, 3, 100)
            hist_llr = self.calculate_NLL(mu_scan, weight,gamma_roi,beta_roi)
            hist_llr = np.array(hist_llr)

            val =  np.abs(mu_scan[np.argmin(hist_llr)] - 1)

            if val < val_min:
                val_min = val
                print("val: ", val)
                print("gamma_roi: ", gamma_roi)
                print("beta_roi: ", beta_roi)
                print("Uncertainity", np.sqrt(gamma_roi + beta_roi)/gamma_roi)
                Beta_roi = beta_roi.copy()
                Gamma_roi = gamma_roi.copy()
                self.best_theta = theta


        theta = self.best_theta
        # predict probabilities for holdout
        X_holdout_sc = self.scaler.transform(self.mu_calc_set['data'])
        w_holdout = self.mu_calc_set['weights']
        y_holdout = self.mu_calc_set['labels']
        y_pred = self._predict(X_holdout_sc, theta)
        

        gamma_roi = (w_holdout*(y_pred * y_holdout)).sum()
        beta_roi = (w_holdout*(y_pred * (1-y_holdout))).sum()


        Y_hat_valid = self._predict(meta_validation_set_df, theta)
        weights_valid = meta_validation_set["weights"].copy() 

        weight = weights_valid*(Y_hat_valid)

        mu_scan = np.linspace(0, 3, 100)
        hist_llr = self.calculate_NLL(mu_scan, weight,gamma_roi,beta_roi)
        hist_llr = np.array(hist_llr)

        val =  np.abs(mu_scan[np.argmin(hist_llr)] - 1)
        print("val: ", val)
        print("gamma_roi: ", gamma_roi)
        print("beta_roi: ", beta_roi)
        print("Uncertainity", np.sqrt(gamma_roi + beta_roi)/gamma_roi)
        self.beta_roi = beta_roi.copy()
        self.gamma_roi = gamma_roi.copy()

        self.force_correction_term = val
        print(f"[*] --- Best theta : {self.best_theta}")

    def _validate(self):
        for valid_set in self.validation_sets:
            valid_set['data'] = self.scaler.transform(valid_set['data'])
            valid_set['predictions'] = self._predict(valid_set['data'], self.best_theta)
            valid_set['score'] = self._return_score(valid_set['data'])


    def calculate_NLL( self,mu_scan, weight_data,gamma_roi,beta_roi):
        def _sigma_asimov_SR(mu):
            return mu*gamma_roi + beta_roi

        sum_data_total_SR = weight_data.sum()
        comb_llr = []
        for i, mu in enumerate(mu_scan):
            hist_llr = (
                -2
                * sum_data_total_SR
                * np.log((_sigma_asimov_SR(mu) / _sigma_asimov_SR(1.0)))
            ) + (2 * (_sigma_asimov_SR(mu) - _sigma_asimov_SR(1.0)))

            comb_llr.append(hist_llr )

        comb_llr = np.array(comb_llr)
        comb_llr = comb_llr - np.amin(comb_llr)

        return comb_llr

    
    def _sigma_asimov_SR(self,mu):
        return mu*self.gamma_roi + self.beta_roi

    def _compute_result(self,weights):
        mu_scan = np.linspace(0, 5, 100)
        sum_data_total_SR = weights.sum()
        comb_llr = []
        for i, mu in enumerate(mu_scan):
            hist_llr = (
                -2
                * sum_data_total_SR
                * np.log((self._sigma_asimov_SR(mu) / self._sigma_asimov_SR(1.0)))
            ) + (2 * (self._sigma_asimov_SR(mu) - self._sigma_asimov_SR(1.0)))
            comb_llr.append(hist_llr)

        comb_llr = np.array(comb_llr)
        comb_llr = comb_llr - np.amin(comb_llr)


        if (mu_scan[np.where((comb_llr <= 9.0) & (comb_llr >= 0.0))].size == 0):
            p16 = 0
            p84 = 0
            mu = 0
        else:
            p16 = min(mu_scan[np.where((comb_llr <= 9.0) & (comb_llr >= 0.0))])
            p84 = max(mu_scan[np.where((comb_llr <= 9.0) & (comb_llr >= 0.0))]) 
            mu = mu_scan[np.argmin(comb_llr)]

        mu = mu - self.force_correction_term
        p16 = p16 - self.force_correction_term
        p84 = p84 - self.force_correction_term
        return mu, p16, p84

    def _compute_validation_result(self):
        print("[*] - Computing Validation result")

        self.validation_delta_mu_hats = []
        for valid_set in self.validation_sets:
            Y_hat_valid = valid_set["predictions"]
            Score_valid = valid_set["score"]

            auc_valid = roc_auc_score(y_true=valid_set["labels"], y_score=Score_valid,sample_weight=valid_set['weights'])
            print(f"\n[*] --- AUC validation : {auc_valid} --- tes : {valid_set['tes']}")

            # print(f"[*] --- PRI_had_pt : {valid_set['had_pt']}")
            # del Score_valid
            weights_valid = valid_set["weights"].copy()

            weight = weights_valid*(Y_hat_valid)

            mu_hat,mu_p16,mu_p84 = self._compute_result(weight)



            # Compute delta mu hat (absolute value)
            delta_mu_hat = np.abs(valid_set["settings"]["ground_truth_mu"] - mu_hat)

            self.validation_delta_mu_hats.append(delta_mu_hat)


            print(f"[*] --- mu: {np.round(valid_set['settings']['ground_truth_mu'], 4)} --- mu_hat: {np.round(mu_hat, 4)} --- delta_mu_hat: {np.round(delta_mu_hat, 4)}")
            print(f"[*] --- p16: {np.round(mu_p16, 4)} --- p84: {np.round(mu_p84, 4)} --- mu_hat_: {np.round(mu_hat, 4)}")

        print(f"[*] --- validation delta_mu_hat (avg): {np.round(np.mean(self.validation_delta_mu_hats), 4)}")
