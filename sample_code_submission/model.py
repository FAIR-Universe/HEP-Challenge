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
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


# ------------------------------
# Absolute path to submission dir
# ------------------------------
submissions_dir = os.path.dirname(os.path.abspath(__file__))
path.append(submissions_dir)

from systematics import postprocess
# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps

hist_analysis_dir = os.path.dirname(submissions_dir)
path.append(hist_analysis_dir)

from hist_analysis import compute_results_syst, compute_result



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
        self.model_name = "XGB_NLL"
        # Intialize class variables
        self.validation_set = None
        self.theta_candidates = np.linspace(0.5, 1.0, 100)
        self.mu_scan = np.linspace(0, 3.92, 100)
        self.threshold = 0.8
        self.bins = 5
        self.bin_nums = 5
        self.force_correction = 0
        self.plot_count = 0
        self.variable = "DER_deltar_lep_had"
        self.scaler = StandardScaler()

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
        self._predict_holdout()
        self.plot_score_holdout()
        # self._choose_theta()
        self.mu_hat_calc()
        # self._validate()
        # self._compute_validation_result()

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
        Y_hat_test = self._return_score(test_df)
        test_set['score'] = Y_hat_test

        if self.plot_count < 2:
            # self.plot_score(test_set)
            self.plot_count += 1


        print("[*] - Computing Test result")
        weights_train = self.train_set["weights"].copy()
        weights_test = test_set["weights"].copy()

        # print(f"[*] --- total weight test: {weights_test.sum()}") 
        # print(f"[*] --- total weight train: {weights_train.sum()}")
        # print(f"[*] --- total weight mu_cals_set: {self.holdout['weights'].sum()}")

        weight_clean = weights_test[Y_hat_test > self.threshold]
        test_df = test_set['data'][Y_hat_test > self.threshold]
        test_array = Y_hat_test[Y_hat_test > self.threshold]

        # test_array = test_df[self.variable]


        test_hist ,_ = np.histogram(test_array,
                    bins=self.bins, density=False, weights=weight_clean)



        mu_hat, mu_p16, mu_p84 = compute_results_syst(test_hist,self.fit_function_dict)
        delta_mu_hat = mu_p84 - mu_p16

        print(f"[*] --- mu_hat: {mu_hat}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        return {
            "mu_hat": mu_hat,
            "delta_mu_hat": delta_mu_hat,
            "p16": mu_p16,
            "p84": mu_p84
        }

    def _init_model(self):
        print("[*] - Intialize Baseline Model (XBM Classifier Model)")

        self.model = XGBClassifier(
            tree_method="hist",
            use_label_encoder=False,
            eval_metric=['logloss', 'auc'],
            n_thread=1,
            n_jobs=1,
        )
    def _generate_validation_sets(self):
        print("[*] - Generating Validation sets")
        print("[*] -- Basic Parameters")
        print(f"[*] --- theta candidates: {min(self.theta_candidates)} - {max(self.theta_candidates)} --- len: {len(self.theta_candidates)}")
        print(f"[*] --- bins: {self.bins}")
        
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

        train_df, holdout_df, train_labels, holdout_labels, train_weights, holdout_weights = train_test_split(
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
        holdout_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_background_weights = holdout_weights[holdout_labels == 0].sum()

        # Balance the sum of weights for signal and background in the training and validation sets
        train_weights[train_labels == 1] *= signal_weights / train_signal_weights
        train_weights[train_labels == 0] *= background_weights / train_background_weights
        valid_weights[valid_labels == 1] *= signal_weights / valid_signal_weights
        valid_weights[valid_labels == 0] *= background_weights / valid_background_weights
        holdout_weights[holdout_labels == 1] *= signal_weights / holdout_signal_weights
        holdout_weights[holdout_labels == 0] *= background_weights / holdout_background_weights

        train_df = train_df.copy()
        train_df["weights"] = train_weights
        train_df["labels"] = train_labels
        train_df = postprocess(train_df)

        train_weights = train_df.pop('weights')
        train_labels = train_df.pop('labels')
        

        holdout_df = holdout_df.copy()
        holdout_df["weights"] = holdout_weights
        holdout_df["labels"] = holdout_labels

        holdout_df = postprocess(holdout_df)

        holdout_weights = holdout_df.pop('weights')
        holdout_labels = holdout_df.pop('labels')

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

        self.holdout = {
                "data": holdout_df,
                "labels": holdout_labels,
                "weights": holdout_weights
            }

        self.validation_sets = []
        for _ in range(10):
            # Loop 10 times to generate 10 validation sets
            tes = round(np.random.uniform(0.9, 1.10), 2)
            # apply systematics
            valid_df_temp = valid_df.copy()
            valid_df_temp["weights"] = valid_weights
            valid_df_temp["labels"] = valid_labels

            valid_with_systematics_temp = self.systematics(
                data=valid_df_temp,
                tes=tes
            ).data
            # valid_with_systematics_temp = postprocess(valid_df_temp)

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
        holdout_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_background_weights = holdout_weights[holdout_labels == 0].sum()

        print(f"[*] --- original signal: {signal_weights} --- original background: {background_weights}")
        print(f"[*] --- train signal: {train_signal_weights} --- train background: {train_background_weights}")
        print(f"[*] --- valid signal: {valid_signal_weights} --- valid background: {valid_background_weights}")
        print(f"[*] --- holdout signal: {holdout_signal_weights} --- holdout background: {holdout_background_weights}")

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
        self.train_set['predictions'] = (self.train_set['data'], self.threshold)

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
        y_predict = self.model.predict_proba(X)[:, 1]
        return y_predict

    def _predict(self, X, theta):
        Y_predict = self._return_score(X)
        predictions = (Y_predict > theta).astype(int)
        return predictions


    def _predict_holdout(self):
        print("[*] --- Predicting Holdout set")
        X_holdout = self.holdout['data']
        X_holdout_sc = self.scaler.transform(X_holdout)
        self.holdout['score'] = self._return_score(X_holdout_sc)
        print("[*] --- Predicting Holdout set done")
        print("[*] --- score = ", self.holdout['score'])

    def plot_score_holdout(self):
        _, ax = plt.subplots()

        bins = int(self.bin_nums/(1 - self.threshold))

        plt.hist(
            self.holdout['score'][self.holdout['labels'] == 1], 
            bins=bins, density=False, alpha=0.6, color='b',
            weights=self.holdout['weights'][self.holdout['labels'] == 1]
        )
        plt.hist(
            self.holdout['score'][self.holdout['labels'] == 0], 
            bins=bins, density=False, alpha=0.6, color='r',
            weights=self.holdout['weights'][self.holdout['labels'] == 0]
        )

        plt.vlines(self.threshold,0,100000,linestyles='dashed',color='k',label = "threshold")


        plt.legend()
        plt.title(self.model_name + ' Scores')
        plt.xlabel(" Score ")
        plt.ylabel(" Events ")
        ax.set_yscale('log')
        plt.show()

        # ...

    def plot_score(self,test_set):

        _, ax = plt.subplots()

        bins = int(self.bin_nums/(1 - self.threshold))
        high_low = (0,1)
        density = False

        holdout_val = self.holdout['score']
        label_holdout = self.holdout['labels']

        weights_holdout_signal = self.holdout['weights'][label_holdout == 1]
        weights_holdout_background = self.holdout['weights'][label_holdout == 0]

        test_val = test_set['score']
        weights_test = test_set['weights']

        signal_val = holdout_val[label_holdout == 1]    
        background_val = holdout_val[label_holdout == 0]

        plt.hist([signal_val, background_val], bins=bins,histtype='bar', 
                 stacked=True, label=['holdout_signal', 'holdout_background'], alpha=0.7,
                 weights=weights_holdout_signal,color=['cyan', 'magenta'])

        

        # plt.hist(holdout_val[label_holdout == 0], bins=bins,histtype='bar', 
        #          stacked=True, label=['holdout_background'], alpha=0.7, 
        #          weights=weights_holdout_background,color='cyan')
        # plt.hist(holdout_val[label_holdout == 1], bins=bins,histtype='step', 
        #          stacked=True, label=['holdout_signal'], alpha=0.7, 
        #          weights=weights_holdout_signal,edgecolor='red', fill=False)

        hist, bins = np.histogram(test_val,
                                    bins=bins, range=high_low, density=density, weights=weights_test)
        scale = len(test_val) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        center = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(center, hist, yerr=err, fmt='o', c='k', label='pseudo-data')

        plt.legend()
        plt.title(self.model_name + ' Scores')
        plt.xlabel(" Score ")
        plt.ylabel(" Events ")
        ax.set_yscale('log')
        plt.show()

    def mu_hat_calc(self):  
        Y_hat_holdout = self.holdout['score']
        Y_holdout = self.holdout['labels']
        weights_holdout = self.holdout['weights']

        mu_calc_df = self.holdout['data'][Y_hat_holdout > self.threshold]
        # compute gamma_roi

        weights = weights_holdout[Y_hat_holdout > self.threshold]
        # mu_calc_array = mu_calc_df[self.variable]
        mu_calc_array = Y_hat_holdout[Y_hat_holdout > self.threshold]


        mu_calc_hist , bins = np.histogram(mu_calc_array,
                    bins= self.bins, density=False, weights=weights)


        self.theta_function()


        mu_hat, mu_p16, mu_p84 = compute_results_syst(mu_calc_hist,self.fit_function_dict)

        print(f"[*] --- mu_hat: {mu_hat} --- mu_p16: {mu_p16} --- mu_p84: {mu_p84}")



    def nominal_histograms(self,mu,theta):

        X_holdout = self.holdout['data']

        X_holdout['weights'] = self.holdout['weights'].copy()
        X_holdout['labels'] = self.holdout['labels'].copy()

        holdout_syst = self.systematics(
            data=X_holdout.copy(),
            tes=theta
        ).data


        Y_holdout = holdout_syst.pop('labels')
        weights_holdout = holdout_syst.pop('weights')

        X_holdout_sc = self.scaler.transform(holdout_syst)
        Y_hat_holdout = self._return_score(X_holdout_sc)


        mu_calc_df = holdout_syst[Y_hat_holdout > self.threshold]
        # compute gamma_roi

        weights_holdout = weights_holdout[Y_hat_holdout > self.threshold]
        # holdout_val = mu_calc_df[self.variable]
        holdout_val = Y_hat_holdout[Y_hat_holdout > self.threshold]

        label_holdout = Y_holdout[Y_hat_holdout > self.threshold]

        weights_holdout_signal = weights_holdout[label_holdout == 1]
        weights_holdout_background = weights_holdout[label_holdout == 0]

        holdout_signal_hist , self.bins = np.histogram(holdout_val[label_holdout == 1],
                    bins= self.bins, density=False, weights=weights_holdout_signal)
        
        holdout_background_hist , self.bins = np.histogram(holdout_val[label_holdout == 0],
                    bins= self.bins, density=False, weights=weights_holdout_background)


        return holdout_signal_hist , holdout_background_hist


    def theta_function(self,plot_count=0):

        fit_line_s_list = []
        fit_line_b_list = []
        theta_list = np.linspace(0.9,1.1,10)
        s_list = [[] for _ in range(self.bins)]
        b_list = [[] for _ in range(self.bins)]
        
        for theta in tqdm(theta_list):
            mu_hat = 1.0
            s , b = self.nominal_histograms(mu_hat,theta)
            # print(f"[*] --- s: {s}")
            # print(f"[*] --- b: {b}")

            for i in range(len(s)):
                s_list[i].append(s[i])
                b_list[i].append(b[i])

        print(f"[*] --- s_list shape: {np.array(s_list).shape}")
        print(f"[*] --- b_list shape: {np.array(b_list).shape}")
        print(f"[*] --- theta_list shape: {np.array(theta_list).shape}")

        for i in range(len(s_list)):
            s_array = np.array(s_list[i])
            b_array = np.array(b_list[i])

            coef_s = np.polyfit(theta_list,s_array,1)

            coef_b = np.polyfit(theta_list,b_array,1)

            fit_line_s_list.append(np.poly1d(coef_s))
            fit_line_b_list.append(np.poly1d(coef_b))

        if plot_count > 0:
            for i in range(min(plot_count,len(s_list))):
                plt.plot(theta_list,s_list[i])
                plt.show()

                plt.plot(theta_list,b_list[i])
                plt.show()


        print(f"[*] --- fit_line_s_list: {fit_line_s_list}")
        print(f"[*] --- fit_line_b_list: {fit_line_b_list}")

        self.fit_function_dict = {
            "gamma_roi": fit_line_s_list,
            "beta_roi": fit_line_b_list
        }




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
        for theta in tqdm(self.theta_candidates):
            meta_validation_set_df_sc = self.scaler.transform(meta_validation_set["data"])
            meta_validation_set['score'] = self._return_score(meta_validation_set_df_sc)

            weights_valid = meta_validation_set["weights"].copy()
            valid_df = meta_validation_set["data"][meta_validation_set['score'] > theta]
            valid_array = valid_df[self.variable]
            weights_valid = weights_valid[meta_validation_set['score'] > theta]  
            Y_hat_valid = meta_validation_set['score'][meta_validation_set['score'] > theta]
             
            # Get predictions from trained model


            # get region of interest

            # predict probabilities for holdout
            holdout_val = self.holdout['data'][self.variable]
            Y_hat_holdout = self.holdout['score']
            Y_holdout = self.holdout['labels']
            weights_holdout = self.holdout['weights']
            # compute gamma_roi

            weights_holdout = weights_holdout[Y_hat_holdout > theta]
            holdout_val = holdout_val[Y_hat_holdout > theta]

            Y_holdout = Y_holdout[Y_hat_holdout > theta]

            weights_holdout_signal = weights_holdout[Y_holdout == 1]
            weights_holdout_bkg = weights_holdout[Y_holdout == 0]
            bins = self.bins
            
            gamma_roi ,bins = np.histogram(holdout_val[Y_holdout == 1],
                        bins=bins, density=False, weights=weights_holdout_signal)
            
            beta_roi , bins = np.histogram(holdout_val[Y_holdout == 0],
                        bins=bins, density=False, weights=weights_holdout_bkg)
            

            
            hist_llr = self.calculate_NLL(weights_valid,Y_hat_valid,beta_roi,gamma_roi)

            val =  np.abs(self.mu_scan[np.argmin(hist_llr)] - 1)

            if val < val_min:
                print("val: ", val)
                print("gamma_roi: ", gamma_roi)
                print("beta_roi: ", beta_roi)
                print("theta: ", theta)
                print("Uncertainity", np.sqrt(gamma_roi + beta_roi)/gamma_roi)
                val_min = val
                self.threshold = theta

        print(f"[*] --- best theta: {self.threshold}")



    def _validate(self):
        for valid_set in self.validation_sets:
            Scaled_valid_df = self.scaler.transform(valid_set['data'])
            valid_set['predictions'] = self._predict(Scaled_valid_df, self.threshold)
            valid_set['score'] = self._return_score(Scaled_valid_df)



    def _compute_validation_result(self):
        print("[*] - Computing Validation result")

        self.validation_delta_mu_hats = []
        for valid_set in self.validation_sets:
            Y_hat_valid = valid_set["predictions"]
            Score_valid = valid_set["score"]

            auc_valid = roc_auc_score(y_true=valid_set["labels"], y_score=Score_valid,sample_weight=valid_set['weights'])
            print(f"\n[*] --- AUC validation : {auc_valid} --- tes : {valid_set['tes']}")

            weights_valid = valid_set["weights"].copy()


            weight_clean = weights_valid[Y_hat_valid > self.threshold]
            valid_df_clean = valid_set['data'][Y_hat_valid > self.threshold]

            valid_array = valid_df_clean[self.variable]

            valid_hist ,_ = np.histogram(valid_array,
                    bins=self.bins, density=False, weights=weight_clean)


            mu_hat, mu_p16, mu_p84 = compute_result(valid_hist,self.fit_function_dict)

            # Compute delta mu hat (absolute value)
            delta_mu_hat = np.abs(valid_set["settings"]["ground_truth_mu"] - mu_hat)

            self.validation_delta_mu_hats.append(delta_mu_hat)


            print(f"[*] --- p16: {np.round(mu_p16, 4)} --- p84: {np.round(mu_p84, 4)} --- mu_hat: {np.round(mu_hat, 4)}")

        print(f"[*] --- validation delta_mu_hat (avg): {np.round(np.mean(self.validation_delta_mu_hats), 4)}")
