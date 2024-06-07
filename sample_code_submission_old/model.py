import os
from sys import path
import numpy as np
import pandas as pd
from math import sqrt, log
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import json
#import mplhep as hep

#hep.set_style("ATLAS")
# ------------------------------
# Absolute path to submission dir
# ------------------------------
submissions_dir = os.path.dirname(os.path.abspath(__file__))
path.append(submissions_dir)

# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps

hist_analysis_dir = os.path.dirname(submissions_dir)
path.append(hist_analysis_dir)

from hist_analysis import compute_result,plot_score



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
            get_train_set=None,
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
        self.train_set = get_train_set
        self.systematics = systematics
        self.model_name = "XGB_NLL"
        # Intialize class variables
        self.threshold = 0.9
        self.bins = 30
        self.bin_nums = 30
        self.plot_count = 1
        self.calibration = 0
        self.scaler = StandardScaler()
        self.SYST = True

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """

        self._generate_holdout_sets()
        self._init_model()
        self._train()
        self.mu_hat_calc()
        self.save_model()

        self.plot_dir = os.path.join(submissions_dir, "plots/")
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

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
        test_score = self._return_score(test_df)
        test_set['score'] = test_score

        print("[*] - Computing Test result")
        weights_test = test_set["weights"].copy()
        
        print( "sum of weight",weights_test.sum())


        test_hist ,bins = np.histogram(test_score,
                    bins=self.bins, density=False, weights=weights_test)
        
        test_hist_control = test_hist[-self.control_bins:]

        mu_hat, mu_p16, mu_p84, alpha = compute_result(test_hist_control,self.fit_dict_control,SYST=True)
        delta_mu_hat = mu_p84 - mu_p16
        
        mu_p16 = mu_p16-self.calibration
        mu_p84 = mu_p84-self.calibration
        mu_hat = mu_hat-self.calibration

        if self.plot_count > 0:
            hist_fit_s = []
            hist_fit_b = []

            for i in range(self.bin_nums):
                hist_fit_s.append(self.fit_dict["gamma_roi"][i](alpha))
                hist_fit_b.append(self.fit_dict["beta_roi"][i](alpha))

            hist_fit_s = np.array(hist_fit_s)
            hist_fit_b = np.array(hist_fit_b)

            plot_score(test_hist,hist_fit_s,hist_fit_b,mu_hat,bins,threshold=self.threshold,save_path=(self.plot_dir + f"XGB_score_{self.plot_count}.png"))

            self.plot_count = self.plot_count - 1 
            
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
            n_thread=1,
            n_jobs=1,
        )

    def _generate_holdout_sets(self):
        print("[*] - Generating Validation sets")

        # Calculate the sum of weights for signal and background in the original dataset
        signal_weights = self.train_set["weights"][self.train_set["labels"] == 1].sum()
        background_weights = self.train_set["weights"][self.train_set["labels"] == 0].sum()
        detailed_label = np.array(self.train_set["detailed_labels"])
        
        keys = np.unique(detailed_label)
        
        for key in keys:
            weight_keys =  self.train_set["weights"][detailed_label == key]
            print("  ", key ," ",weight_keys.sum())
        
        # Split the data into training and holdout sets while preserving the proportion of samples with respect to the target variable
        train_df, holdout_df, train_labels, holdout_labels, train_weights, holdout_weights =  train_test_split(
            self.train_set["data"],
            self.train_set["labels"],
            self.train_set["weights"],
            test_size=0.1,
            stratify=self.train_set["labels"]
        )



        # Calculate the sum of weights for signal and background in the training and holdout sets
        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()

        holdout_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_background_weights = holdout_weights[holdout_labels == 0].sum()

        # Balance the sum of weights for signal and background in the training and holdout sets
        train_weights[train_labels == 1] *= signal_weights / train_signal_weights
        train_weights[train_labels == 0] *= background_weights / train_background_weights

        holdout_weights[holdout_labels == 1] *= signal_weights / holdout_signal_weights
        holdout_weights[holdout_labels == 0] *= background_weights / holdout_background_weights

        
        self.train_set = {
            "data": train_df,
            "labels": train_labels,
            "weights": train_weights,
            "settings": self.train_set["settings"]
        }

        self.train_set = self.systematics(
        data_set=self.train_set,
        tes=1.0,
        jes=1.0,
        )
        
        print(self.train_set['data'].columns)

        train_weights = self.train_set['weights']
        train_labels = self.train_set['labels']

        self.holdout = {
                "data": holdout_df,
                "labels": holdout_labels,
                "weights": holdout_weights
            }

        
        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()

        holdout_set_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_set_background_weights = holdout_weights[holdout_labels == 0].sum()

        print(f"[*] --- original signal: {signal_weights} --- original background: {background_weights}")
        print(f"[*] --- train signal: {train_signal_weights} --- train background: {train_background_weights}")
        print(f"[*] --- holdout_set signal: {holdout_set_signal_weights} --- holdout_set background: {holdout_set_background_weights}")
  
  
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

    def mu_hat_calc(self):

        holdout_post = self.systematics(
            data_set=self.holdout,
            tes=1.0,
            jes=1.0,
        )

        weights_holdout  = holdout_post['weights']

        X_holdout_sc = self.scaler.transform(holdout_post['data'])
        holdout_array = self._return_score(X_holdout_sc)
        print("[*] --- Predicting Holdout set done")
        print("[*] --- score = ", holdout_array)

        # compute gamma_roi

        self.control_bins = int(self.bin_nums * (1 - self.threshold))

        if self.SYST:
            self.alpha_function(plot_count=2)

        else:
            s , b = self.nominal_histograms(1)
            self.fit_dict = {
                "gamma_roi": s,
                "beta_roi": b,
                "error_s": [0 for _ in range(self.bins)],
                "error_b": [0 for _ in range(self.bins)]
            }

            self.fit_dict_control = {
                "gamma_roi": s[-self.control_bins:],
                "beta_roi": b[-self.control_bins:],
                "error_s": [0 for _ in range(self.control_bins)],
                "error_b": [0 for _ in range(self.control_bins)]
            }

            

        holdout_hist , _ = np.histogram(holdout_array,
                    bins = self.bins, density=False, weights=weights_holdout)
        
        
        holdout_hist_control = holdout_hist[-self.control_bins:]
        # holdout_hist_control = (s + b)[-self.control_bins:]

        mu_hat, mu_p16, mu_p84, alpha = compute_result(holdout_hist_control,self.fit_dict_control,SYST=self.SYST)

        self.calibration = mu_hat - 1
        
        print(f"[*] --- mu_hat: {mu_hat} --- mu_p16: {mu_p16} --- mu_p84: {mu_p84} --- alpha: {alpha}")

        del self.holdout


    def nominal_histograms(self,alpha):

        holdout_syst = self.systematics(
            data_set=self.holdout,
            tes=alpha,
            jes=1.0,
        )

        label_holdout = holdout_syst['labels']
        weights_holdout = holdout_syst['weights']

        X_holdout_sc = self.scaler.transform(holdout_syst['data'])
        holdout_val = self._return_score(X_holdout_sc)

        weights_holdout_signal = weights_holdout[label_holdout == 1]
        weights_holdout_background = weights_holdout[label_holdout == 0]

        holdout_signal_hist , _ = np.histogram(holdout_val[label_holdout == 1],
                    bins= self.bins, density=False, weights=weights_holdout_signal)
        
        holdout_background_hist , _ = np.histogram(holdout_val[label_holdout == 0],
                    bins= self.bins, density=False, weights=weights_holdout_background)


        return holdout_signal_hist , holdout_background_hist


    def alpha_function(self,plot_count=25):

        fit_line_s_list = []
        fit_line_b_list = []
        self.coef_b_list = []
        self.coef_s_list = []

        error_s = []
        error_b = []

        alpha_list = np.linspace(0.9,1.1,10)
        s_array = np.zeros((len(alpha_list),self.bins))
        b_array = np.zeros((len(alpha_list),self.bins))
        
        for i in tqdm(range(len(alpha_list))):
            s_array[i] , s_array[i] = self.nominal_histograms(alpha_list[i])

        s_array = s_array.T
        b_array = b_array.T

        print(f"[*] --- s_array shape: {s_array.shape}")
        print(f"[*] --- b_array shape: {b_array.shape}")
        

        for i in range(s_array.shape[0]):

            coef_s = np.polyfit(alpha_list, s_array[i], 3)
            coef_b = np.polyfit(alpha_list, b_array[i], 3)

            fit_fun_s = np.poly1d(coef_s)
            fit_fun_b = np.poly1d(coef_b)

            fit_line_s_list.append(fit_fun_s)
            fit_line_b_list.append(fit_fun_b)

            coef_b_ = coef_b.tolist()
            coef_s_ = coef_s.tolist()

            self.coef_s_list.append(coef_s_)
            self.coef_b_list.append(coef_b_)


        for i in range(min(plot_count,self.bins)):

            _, ax = plt.subplots()

            plt.plot(alpha_list,s_array[i],'b.',label="s")
            plt.plot(alpha_list,fit_line_s_list[i](alpha_list),'cyan',label="fit s")
            plt.legend()
            plt.title(f"Bin {i}")
            plt.xlabel("alpha")
            plt.ylabel("Events")
            # hep.atlas.text(loc=1, text='Internal')
            save_path = os.path.join(submissions_dir, "plots/")
            plot_file = os.path.join(save_path, f"NN_s_{i}.png")
            plt.savefig(plot_file)
            # plt.show()

            _, ax = plt.subplots()

            plt.plot(alpha_list,b_array[i],'r.',label="b")
            plt.plot(alpha_list,fit_line_b_list[i](alpha_list),'orange',label="fit b")
            plt.legend()
            plt.title(f"Bin {i}")
            plt.xlabel("alpha")
            plt.ylabel("Events")
            # hep.atlas.text(loc=1, text='Internal')
            save_path = os.path.join(submissions_dir, "plots/")
            plot_file = os.path.join(save_path, f"NN_b_{i}.png")
            plt.savefig(plot_file)
            # plt.show()



            plot_count = plot_count - 1

            if plot_count <= 0:
                break
        


        self.fit_dict = {
            "gamma_roi": fit_line_s_list,
            "beta_roi": fit_line_b_list,
        }

        print(f"[*] --- number of bins: {self.bins}")
        print(f"[*] --- number of control bins: {self.control_bins}")

        self.fit_dict_control = {
            "gamma_roi": fit_line_s_list[-self.control_bins:],
            "beta_roi": fit_line_b_list[-self.control_bins:],
        }

    def save_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_dir = os.path.join(parent_dir, "XGB_saved")   
        model_path = os.path.join(model_dir, "model.pkl")
        settings_path = os.path.join(model_dir, "settings.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")


        print("[*] - Saving Model")
        print(f"[*] --- model path: {model_path}")
        print(f"[*] --- settings path: {settings_path}")
        print(f"[*] --- scaler path: {scaler_path}")

        

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(model_path, "wb") as model_file:
            pickle.dump(self.model, model_file)

        if self.SYST:
            settings = {
                "threshold": self.threshold,
                "bin_nums": self.bin_nums,
                "control_bins": self.control_bins,
                "coef_s_list": self.coef_s_list,
                "coef_b_list": self.coef_b_list,
            }
        else:
            settings = {
                "threshold": self.threshold,
                "bin_nums": self.bin_nums,
                "control_bins": self.control_bins,
                "gamma_roi": self.fit_dict["gamma_roi"],
                "beta_roi": self.fit_dict["beta_roi"],
            }


        pickle.dump(settings, open(settings_path, "wb"))

        pickle.dump(self.scaler, open(scaler_path, "wb"))

        print("[*] - Model saved")

