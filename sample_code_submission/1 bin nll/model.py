# ------------------------------
# Imports
# ------------------------------
import os
from sys import path
import numpy as np
from sklearn.preprocessing import StandardScaler
from systematics import postprocess


# ------------------------------
# Import local modules
# ------------------------------
module_dir= os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(module_dir)
path.append(root_dir)
from bootstrap import bootstrap


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
    3) predict: predict mu_hat and delta_mu_hat

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods
    """

    def __init__(
            self,
            train_set=None,
            systematics=None,
            model_name="one-bin",

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

            model_name:

                name of the model, default: BDT


        Returns:
            None
        """

        # Set class variables from parameters
        self.model_name = model_name
        self.train_set = train_set
        self.systematics = systematics

        # Intialize class variables
        self.force_correction = 0
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

        self._init_model()
        self._predict()
    def predict(self, test_set):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hats
                - delta_mu_hats
                - p16
                - p84
        """

        self._test(test_set)

        return {
            "mu_hat": self.mu_hat,
            "delta_mu_hat": self.delta_mu_hat,
            "p16": self.p16,
            "p84": self.p84
        }

    def _init_model(self):
        print("[*] - Intialize model")
        train_df = self.train_set["data"].copy()
        train_df['weights'] = self.train_set["weights"].copy()
        train_df['labels'] = self.train_set["labels"].copy()

        train_df = postprocess(train_df)

        self.train_set["labels"] = train_df.pop("labels")
        self.train_set["weights"] = train_df.pop("weights")
        self.train_set["data"] = train_df.copy()

    def _fit(self):
        print("[*] --- Fitting Model")
        

    def _predict(self):
        print("[*] --- Predicting")
        train_weights = self.train_set["weights"].copy()
        train_labels = self.train_set["labels"].copy()

        self.gamma_roi = (train_weights*(train_labels)).sum()
        self.beta_roi = (train_weights*(1-train_labels)).sum()   

        print(f"[*] --- gamma_roi: {self.gamma_roi}")
        print(f"[*] --- beta_roi: {self.beta_roi}")

        significance = self.gamma_roi / np.sqrt(self.gamma_roi + self.beta_roi)
        print(f"[*] --- significance: {significance}")
        
        mu_hat, mu_p16, mu_p84 = self._compute_result(train_weights)
        delta_mu_hat = mu_p84 - mu_p16
        val = mu_hat - 1.0
        print(f"[*] --- mu_hat: {mu_hat}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")

        # self.force_correction =  val



    def _sigma_asimov_SR(self,mu):
        return mu*self.gamma_roi + self.beta_roi


    def calculate_NLL(self,mu_scan, weight_data):
        sum_data_total_SR = weight_data.sum()
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

        return comb_llr
    
    def _compute_result(self,weights):
        mu_scan = np.linspace(0, 5, 100)
        nll = self.calculate_NLL(mu_scan, weights)
        hist_llr = np.array(nll)

        if (mu_scan[np.where((hist_llr <= 1.0) & (hist_llr >= 0.0))].size == 0):
            p16 = 0
            p84 = 0
            mu = 0
        else:
            p16 = min(mu_scan[np.where((hist_llr <= 1.0) & (hist_llr >= 0.0))])
            p84 = max(mu_scan[np.where((hist_llr <= 1.0) & (hist_llr >= 0.0))]) 
            mu = mu_scan[np.argmin(hist_llr)]

        mu = mu - self.force_correction
        p16 = p16 - self.force_correction
        p84 = p84 - self.force_correction

        return mu, p16, p84
    
    def _test(self, test_set=None):
        print("[*] - Testing")
        weights_test = test_set["weights"].copy()
        print(f"[*] --- weights_test: {weights_test.sum()}")

        mu_hat, mu_p16, mu_p84 = self._compute_result(weights_test)
        delta_mu_hat = mu_p84 - mu_p16
        
        print(f"[*] --- mu_hat: {mu_hat.mean()}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        self.mu_hat = mu_hat.mean()
        self.delta_mu_hat = delta_mu_hat
        self.p16 = mu_p16
        self.p84 = mu_p84
