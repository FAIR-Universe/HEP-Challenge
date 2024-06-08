import numpy as np
from sys import path
from systematics import systematics
import pickle
from iminuit import Minuit
import matplotlib.pyplot as plt

path.append("../")
path.append("../ingestion_program")

class StatisticalAnalysis:
    """
    A class that performs statistical analysis on a given model and holdout set.

    Args:
        model: The model used for prediction.
        holdout_set (dict): Dictionary containing holdout set data and labels.
        bins (int, optional): Number of bins for histogram calculation. Defaults to 10.

    Attributes:
        model: The model used for prediction.
        bins (int): Number of bins for histogram calculation.
        syst_settings (dict): Dictionary containing the systematic settings.
        alpha_ranges (dict): Dictionary containing the alpha ranges for each systematic parameter.
        holdout_set (dict): Dictionary containing holdout set data and labels.
        fit_function_s (list): List of lists containing the fit functions for signal events.
        fit_function_b (list): List of lists containing the fit functions for background events.
        saved_info (dict): Dictionary containing the saved information for mu calculation.

    Methods:
        compute_mu: Perform calculations to calculate mu.
        calculate_saved_info: Calculate the saved_info dictionary for mu calculation.
        nominal_histograms: Calculate the nominal histograms for signal and background events.
        fit_functions: Fits polynomial functions to the given data for a specific key.
        alpha_function: Calculate the alpha functions for signal and background events.
        save: Save the saved_info dictionary to a file.
        load: Load the saved_info dictionary from a file.
    """
    def __init__(self,model,holdout_set,bins=10):
        self.model = model
        self.bins = bins
        self.syst_settings = {
            'tes': 1.0,
            'bkg_scale': 1.0,
            'jes': 1.0,
            'soft_met': 0.0,
            'ttbar_scale': 1.0,
            'diboson_scale': 1.0,
        }
        
        self.alpha_ranges = {
            "tes": np.linspace(0.9, 1.1, 10),
            "bkg_scale": np.linspace(0.995, 1.005, 10),
            "jes": np.linspace(0.9, 1.1, 10),
            "soft_met": np.linspace(0.0, 5.0, 10),
            "ttbar_scale": np.linspace(0.5, 2.0, 10),
            "diboson_scale": np.linspace(0.5, 2.0, 10),
        }
            
        
        holdout_set["data"].reset_index(drop=True, inplace=True)
        
        self.holdout_set = holdout_set
        
    def compute_mu(self,score, weight, plot=False):
        """
        Perform calculations to calculate mu using the profile likelihood method.
        

        Args:
            score (numpy.ndarray): Array of scores.
            weight (numpy.ndarray): Array of weights.

        Returns:
            dict: Dictionary containing calculated values of mu_hat, delta_mu_hat, p16, and p84.
        """
        
        N_obs, _ = np.histogram(score, bins=self.bins, density=False, weights=weight)
                    
        def combined_fit_function_s(x):
            for j in range(len(x)):
                combined_function_s = np.zeros(self.bins)
                combined_function_s_bin = np.zeros(self.bins)
                for i in range(self.bins):
                    combined_function_s_bin[i] = self.fit_function_s[j][i](x[j])
                combined_function_s += combined_function_s_bin
            return combined_function_s/len(self.syst_settings.keys())
            
        def combined_fit_function_b(x):
            for j in range(len(x)):
                combined_function_b = np.zeros(self.bins)
                combined_function_b_bin = np.zeros(self.bins)
                for i in range(self.bins):
                    combined_function_b_bin[i] = self.fit_function_b[j][i](x[j])
                combined_function_b += combined_function_b_bin
            return combined_function_b/len(self.syst_settings.keys())
            
        def sigma_asimov(mu,alpha):
                return mu*combined_fit_function_s(alpha) + combined_fit_function_b(alpha)

        
        def NLL(mu, tes, bkg_scale, jes, soft_met, ttbar_scale, diboson_scale):
            """
            Calculate the negative log-likelihood (NLL) for a given set of parameters.

            Parameters:
            mu (float): Signal strength parameter.
            tes (float): Tau energy scale parameter.
            bkg_scale (float): Background scale parameter.
            jes (float): Jet energy scale parameter.
            soft_met (float): Soft MET parameter.
            ttbar_scale (float): ttbar scale parameter.
            diboson_scale (float): Diboson scale parameter.

            Returns:
            float: The negative log-likelihood value.
            """

            alpha = [tes, bkg_scale, jes, soft_met, ttbar_scale, diboson_scale]

            sigma_asimov_mu = sigma_asimov(mu, alpha)

            hist_llr = (
                - N_obs
                * np.log((sigma_asimov_mu))
            ) + (sigma_asimov_mu)

            return hist_llr.sum()

        result = Minuit(NLL,
                        mu=1.0,
                        tes=1.0,
                        bkg_scale=1.0,
                        jes=1.0,
                        soft_met=0.0,
                        ttbar_scale=1.0,
                        diboson_scale=1.0,
                        )

        result.errordef = Minuit.LIKELIHOOD
        result.migrad()

        mu_hat = result.values['mu']
        mu_p16 = mu_hat - result.errors['mu']
        mu_p84 = mu_hat  + result.errors['mu']

        if plot:
            result.draw_profile('mu')
            result.draw_mnprofile('mu')
            plt.show()
        
        return {
            "mu_hat": mu_hat,
            "delta_mu_hat" : result.errors['mu'] * 2,
            "p16": mu_p16,
            "p84": mu_p84,
        }
        


    def calculate_saved_info(self):
        """
        Calculate the saved_info dictionary for mu calculation.

        Args:
            model: The model used for prediction.
            train_set (dict): Dictionary containing training set data and labels.
            file_path (str, optional): File path to save the calculated saved_info dictionary. Defaults to "saved_info.pkl".

        Returns:
            dict: Dictionary containing calculated values of beta and gamma.
        """

        self.saved_info = {}
        for key in self.syst_settings.keys():
            
            coef_s_list, coef_b_list = self.fit_functions(key)
            self.saved_info[key] = {
                "coef_s": coef_s_list,
                "coef_b": coef_b_list,
            }
                 
        self.alpha_function()

    def nominal_histograms(self, alpha, key):
        """
        Calculate the nominal histograms for signal and background events.

        Parameters:
        - alpha (float): The value of the systematic parameter.
        - key (str): The key corresponding to the systematic parameter.

        Returns:
        - holdout_signal_hist (numpy.ndarray): The histogram of signal events in the holdout set.
        - holdout_background_hist (numpy.ndarray): The histogram of background events in the holdout set.
        """
        syst_settings = self.syst_settings.copy()
        syst_settings[key] = alpha
        holdout_syst = systematics(
            self.holdout_set.copy(),
            tes=syst_settings['tes'],        
            bkg_scale=syst_settings['bkg_scale'],
            jes=syst_settings['jes'],
            soft_met=syst_settings['soft_met'],
            ttbar_scale=syst_settings['ttbar_scale'],
            diboson_scale=syst_settings['diboson_scale'],
        )

        label_holdout = holdout_syst['labels']
        weights_holdout = holdout_syst['weights']

        holdout_val = self.model.predict(holdout_syst['data'])

        weights_holdout_signal = weights_holdout[label_holdout == 1]
        weights_holdout_background = weights_holdout[label_holdout == 0]

        holdout_signal_hist , _ = np.histogram(holdout_val[label_holdout == 1],
                    bins=self.bins, density=False, weights=weights_holdout_signal)
        
        holdout_background_hist , _ = np.histogram(holdout_val[label_holdout == 0],
                    bins=self.bins, density=False, weights=weights_holdout_background)

        return holdout_signal_hist, holdout_background_hist

    def fit_functions(self, key):
        """
        Fits polynomial functions to the given data for a specific key.

        Parameters:
            key (str): The key to identify the data.

        Returns:
            tuple: A tuple containing two lists of coefficients. The first list contains the coefficients for the polynomial fit of the 's_array' data, and the second list contains the coefficients for the polynomial fit of the 'b_array' data.
        """
        coef_b_list = []
        coef_s_list = []

        alpha_list = self.alpha_ranges[key]
        
        s_array = np.zeros((len(alpha_list), self.bins))
        b_array = np.zeros((len(alpha_list), self.bins))
        
        for i in range(len(alpha_list)):
            s_array[i], s_array[i] = self.nominal_histograms(alpha_list[i], key)

        s_array = s_array.T
        b_array = b_array.T

        for i in range(self.bins):
            coef_s = np.polyfit(alpha_list, s_array[i], 3)
            coef_b = np.polyfit(alpha_list, b_array[i], 3)

            coef_s_list.append(coef_s.tolist())
            coef_b_list.append(coef_b.tolist())
            
        print(f"[*] --- coef_s_list shape: {len(coef_s_list)}")
        
        return coef_s_list, coef_b_list
       
    def alpha_function(self):
        
        self.fit_function_s = [[] for _ in range(len(self.syst_settings.keys()))]
        self.fit_function_b = [[] for _ in range(len(self.syst_settings.keys()))]
        
        for key in self.syst_settings.keys():
            
            coef_s_list = self.saved_info[key]['coef_s']
            coef_b_list = self.saved_info[key]['coef_b']
            
            index = list(self.syst_settings.keys()).index(key)
            
            for i in range(self.bins):
                coef_s = coef_s_list[i]
                coef_b = coef_b_list[i]
                self.fit_function_s[index].append(np.poly1d(coef_s))
                self.fit_function_b[index].append(np.poly1d(coef_b))
                
           
    def save(self, file_path):
        """
        Save the saved_info dictionary to a file.

        Args:
            file_path (str): File path to save the object.

        Returns:
            None
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.saved_info, f)
            
    def load(self, file_path):
        """
        Load the saved_info dictionary from a file.

        Args:
            file_path (str): File path to load the object.

        Returns:
            None
        """
        with open(file_path, "rb") as f:
            self.saved_info = pickle.load(f)
        self.alpha_function()