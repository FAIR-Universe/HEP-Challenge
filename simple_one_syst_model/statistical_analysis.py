import os

import numpy as np
from sys import path
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
        fit_function_s (dict): Dictionary of lists containing the fit functions for signal events.
        fit_function_b (dict): Dictionary of lists containing the fit functions for background events.
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

    def __init__(self, model, bins=10, stat_only=False, systematics=None, fixed_syst=None):
        
        self.model = model
        self.bins = bins
        self.bin_edges = np.linspace(0, 1, bins + 1)
        self.syst_settings = {
            'tes': 1.0,
            'bkg_scale': 1.0,
            'jes': 1.0,
            'soft_met': 0.0,
            'ttbar_scale': 1.0,
            'diboson_scale': 1.0,
        }
        self.systematics = systematics  # Function to apply systematics
        
        self.alpha_ranges = {
            "tes": {
                "range": np.linspace(0.9, 1.1, 15),
                "mean": 1.0,
                "std": 0.03,
            },
            "bkg_scale": {
                "range": np.linspace(0.99, 1.01, 15),
                "mean": 1.0,
                "std": 0.003,
            },
            "jes": {
                "range": np.linspace(0.9, 1.1, 15),
                "mean": 1.0,
                "std": 0.03,
            },
            "soft_met": {
                "range": np.linspace(0.0, 5.0, 15),
                "mean": 0.0,
                "std": 3.0,
            },
            "ttbar_scale": {
                "range": np.linspace(0.8, 1.2, 15),
                "mean": 1.0,
                "std": 0.06,
            },
            "diboson_scale": {
                "range": np.linspace(0.0, 2.0, 15),
                "mean": 1.0,
                "std": 0.75,
            },
        }

        # If stat_only is set to True, the systematic parameters will be fixed to the nominal values.
        self.stat_only = stat_only
        # If syst_fixed_setting is set, the systematic parameters will be fixed to the given values.
        self.syst_fixed_setting = fixed_syst

        self.run_syst = None
        self.fit_function_s = {k: [] for k in self.syst_settings.keys()}
        self.fit_function_b = {k: [] for k in self.syst_settings.keys()}
        self.saved_info = {}
        self.syst_load = {syst: False for syst in self.syst_settings.keys()}

    def compute_mu(self, score, weight, plot=None):
        """
        Perform calculations to calculate mu using the profile likelihood method.
        

        Args:
            score (numpy.ndarray): Array of scores.
            weight (numpy.ndarray): Array of weights.

        Returns:
            dict: Dictionary containing calculated values of mu_hat, delta_mu_hat, p16, and p84.
        """


        N_obs, bins = np.histogram(score, bins=self.bin_edges, density=False, weights=weight)

        def combined_fit_function_s(alpha):
            combined_function_s = np.zeros(self.bins)

            if self.run_syst is not None:
                alpha_ = {self.run_syst: alpha[self.run_syst]}
            else:
                alpha_ = alpha

            for syst, value in alpha_.items():
                combined_function_s_bin = np.zeros(self.bins)
                for i in range(self.bins):
                    combined_function_s_bin[i] = self.fit_function_s[syst][i](value)
                combined_function_s += combined_function_s_bin

            return combined_function_s / len(alpha_.keys())

        def combined_fit_function_b(alpha):
            combined_function_b = np.zeros(self.bins)

            if self.run_syst is not None:
                alpha_ = {self.run_syst: alpha[self.run_syst]}
            else:
                alpha_ = alpha

            for syst, value in alpha_.items():
                combined_function_b_bin = np.zeros(self.bins)
                for i in range(self.bins):
                    combined_function_b_bin[i] = self.fit_function_b[syst][i](value)
                combined_function_b += combined_function_b_bin

            return combined_function_b / len(alpha_.keys())

        def sigma_asimov(mu, alpha):
            return mu * combined_fit_function_s(alpha) + combined_fit_function_b(alpha)

        def gaussian_constraint(theta, theta_hat, sigma):
            return np.power(theta - theta_hat, 2) / (2 * sigma ** 2)

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

            alpha = {
                'tes': tes,
                'bkg_scale': bkg_scale,
                'jes': jes,
                'soft_met': soft_met,
                'ttbar_scale': ttbar_scale,
                'diboson_scale': diboson_scale,
            }

            sigma_asimov_mu = sigma_asimov(mu, alpha)

            # Add a small epsilon to avoid log(0) or very small values
            epsilon = 1e-10
            sigma_asimov_mu = np.clip(sigma_asimov_mu, epsilon, None)

            gaus_term = 0
            for syst in self.syst_settings.keys():
                gaus_term += gaussian_constraint(
                    alpha[syst], self.alpha_ranges[syst]['mean'],
                    self.alpha_ranges[syst]['std']
                )

            # adding Gaussian constraint
            hist_llr = (- N_obs * np.log(sigma_asimov_mu)) + sigma_asimov_mu + gaus_term

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

        for key, value in self.alpha_ranges.items():
            result.limits[key] = (value['range'][0], value['range'][-1])
        result.limits['mu'] = (0, 10)

        if self.syst_fixed_setting is not None:
            for key, value in self.syst_fixed_setting.items():
                result.fixto(key, value)
                # print(f"[*] - Fixed {key} to {value}")

        if self.stat_only:
            result.fixed = True
            result.fixed['mu'] = False
            # print("[*] - Fixed all systematics to nominal values.")

        result.errordef = Minuit.LIKELIHOOD
        result.migrad()

        if not result.fmin.is_valid:
            print("Warning: migrad did not converge. Hessian errors might be unreliable.")
            return {
                "mu_hat": -999,
                "delta_mu_hat": -999,
                "p16": -999,
                "p84": -999,
            }

        mu_hat = result.values['mu']
        mu_p16 = mu_hat - result.errors['mu']
        mu_p84 = mu_hat + result.errors['mu']

        if plot:
            print(result)
            result.draw_profile('mu')
            result.draw_mnprofile('mu')
            plt.show()

            os.makedirs("plots", exist_ok=True)
            alpha_test = {syst: result.values[syst] for syst in self.syst_settings.keys()}
            self.plot_stacked_histogram(
                bins,
                combined_fit_function_s(alpha_test),
                combined_fit_function_b(alpha_test),
                mu=mu_hat,
                N_obs=N_obs,
                save_name=f"plots/{plot}.png"
            )

        return {
            "mu_hat": mu_hat,
            "delta_mu_hat": result.errors['mu'] * 2,
            "p16": mu_p16,
            "p84": mu_p84,
        }

    def calculate_saved_info(self, holdout_set, file_path):
        """
        Calculate the saved_info dictionary for mu calculation.

        Args:
            model: The model used for prediction.
            train_set (dict): Dictionary containing training set data and labels.
            file_path (str, optional): File path to save the calculated saved_info dictionary. Defaults to "saved_info.pkl".

        Returns:
            dict: Dictionary containing calculated values of beta and gamma.
        """

        holdout_set["data"].reset_index(drop=True, inplace=True)

        def nominal_histograms(alpha, key):
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
            holdout_syst = self.systematics(
                holdout_set.copy(),
                **syst_settings
            )

            label_holdout = holdout_syst['labels']
            weights_holdout = holdout_syst['weights']

            holdout_val = self.model.predict(holdout_syst['data'])

            weights_holdout_signal = weights_holdout[label_holdout == 1]
            weights_holdout_background = weights_holdout[label_holdout == 0]

            holdout_signal_hist, bins_signal = np.histogram(holdout_val[label_holdout == 1],
                                                            bins=self.bin_edges, density=False,
                                                            weights=weights_holdout_signal)

            holdout_background_hist, bins_background = np.histogram(holdout_val[label_holdout == 0],
                                                                    bins=self.bin_edges, density=False,
                                                                    weights=weights_holdout_background)

            return holdout_signal_hist, holdout_background_hist

        def fit_functions(key):
            """
            Fits polynomial functions to the given data for a specific key.

            Parameters:
                key (str): The key to identify the data.

            Returns:
                tuple: A tuple containing two lists of coefficients. The first list contains the coefficients for the polynomial fit of the 's_array' data, and the second list contains the coefficients for the polynomial fit of the 'b_array' data.
            """
            coef_b_list = []
            coef_s_list = []

            alpha_list = self.alpha_ranges[key]["range"]

            s_array = np.zeros((len(alpha_list), self.bins))
            b_array = np.zeros((len(alpha_list), self.bins))

            for i in range(len(alpha_list)):
                s_array[i], b_array[i] = nominal_histograms(alpha_list[i], key)

            s_array = s_array.T
            b_array = b_array.T

            def calculate_r_squared(y_true, y_pred):
                residuals = y_true - y_pred
                ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
                ss_residual = np.sum(residuals ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                return r_squared

            def calculate_rmse(y_true, y_pred):
                residuals = y_true - y_pred
                mse = np.mean(residuals ** 2)
                rmse = np.sqrt(mse)
                return rmse

            for i in range(self.bins):
                coef_s = np.polyfit(alpha_list, s_array[i], 5)
                coef_b = np.polyfit(alpha_list, b_array[i], 5)

                coef_s_list.append(coef_s.tolist())
                coef_b_list.append(coef_b.tolist())

                # Calculate R^2 and RMSE for coef_s
                poly_vals_s = np.polyval(coef_s, alpha_list)
                r_squared_s = calculate_r_squared(s_array[i], poly_vals_s)
                rmse_s = calculate_rmse(s_array[i], poly_vals_s)

                # Calculate R^2 and RMSE for coef_b
                poly_vals_b = np.polyval(coef_b, alpha_list)
                r_squared_b = calculate_r_squared(b_array[i], poly_vals_b)
                rmse_b = calculate_rmse(b_array[i], poly_vals_b)

                # Print the results
                print(f"[*] Bin {i + 1}:")
                print(f"    Sig: R^2: {r_squared_s:.4f}, RMSE: {rmse_s:.4f}")
                print(f"    Bkg: R^2: {r_squared_b:.4f}, RMSE: {rmse_b:.4f}")

            print(f"[*] --- coef_s_list shape: {len(coef_s_list)}")

            os.makedirs("plots/fittings", exist_ok=True)

            self.visualize_fit(
                alpha_list=alpha_list, array=s_array, coefficient_list=coef_s_list, alpha_name=f'Signal: {key}',
                save_name=f"plots/fittings/signal_{key}.png"
            )
            self.visualize_fit(
                alpha_list=alpha_list, array=b_array, coefficient_list=coef_b_list, alpha_name=f'Background: {key}',
                log_y=True, save_name=f"plots/fittings/background_{key}.png"
            )

            return coef_s_list, coef_b_list

        for key in self.syst_settings.keys():

            if self.syst_load[key]:
                print(f"[***] - Loading template for {key}")
                continue

            print(f"[***] - Calculating template for {key}")
            coef_s_list, coef_b_list = fit_functions(key)
            self.saved_info[key] = {
                "coef_s": coef_s_list,
                "coef_b": coef_b_list,
            }

            with open(os.path.join(file_path, f'{key}.pkl'), "wb") as f:
                pickle.dump(self.saved_info[key], f)

    def alpha_function(self):

        for key in self.syst_settings.keys():
            coef_s_list = self.saved_info[key]['coef_s']
            coef_b_list = self.saved_info[key]['coef_b']

            for i in range(self.bins):
                coef_s = coef_s_list[i]
                coef_b = coef_b_list[i]
                self.fit_function_s[key].append(np.poly1d(coef_s))
                self.fit_function_b[key].append(np.poly1d(coef_b))

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

        for key in self.syst_settings.keys():
            if os.path.exists(os.path.join(file_path, f"{key}.pkl")):
                self.syst_load[key] = True
                with open(os.path.join(file_path, f"{key}.pkl"), "rb") as f:
                    self.saved_info[key] = pickle.load(f)

        # return if all the systematics are loaded
        return all(self.syst_load.values())

    def plot_stacked_histogram(self, bins, signal_fit, background_fit, mu, N_obs, save_name=None):
        """
        Plot a stacked histogram with combined signal and background fits and observed data points.

        Parameters:
            bins (numpy.ndarray): Bin edges.
            signal_fit (numpy.ndarray): Combined signal fit values.
            background_fit (numpy.ndarray): Combined background fit values.
            mu (float): Multiplicative factor for the signal.
            N_obs (numpy.ndarray): Observed data points.
            save_name (str, optional): Name of the file to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15), gridspec_kw={'height_ratios': [2, 1]})

        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bin_widths = np.diff(bins)

        # Main plot: stacked histograms for signal and background
        ax1.bar(bin_centers, signal_fit, width=bin_widths, color='g', align='center', label='Signal')
        ax1.bar(bin_centers, background_fit, width=bin_widths, alpha=0.5, label='Background', color='b', align='center')
        ax1.bar(bin_centers, signal_fit * mu, width=bin_widths, alpha=0.5, label=f'Signal * {mu:.1f}', color='r',
                align='center', bottom=background_fit)

        # Plot observed data points
        ax1.errorbar(bin_centers, N_obs, yerr=np.sqrt(N_obs), fmt='o', color='k', label='Observed Data')

        ax1.set_xlabel('Score')
        ax1.set_ylabel('Counts')
        ax1.set_yscale('log')  # Set y-axis to logarithmic scale
        ax1.set_title('Stacked Histogram: Signal and Background Fits with Observed Data')
        ax1.legend()

        # Subplot: distribution of (N_obs - background_fit) and signal_fit
        diff = N_obs - background_fit
        ax2.errorbar(bin_centers, diff, yerr=np.sqrt(np.abs(diff)), fmt='o', color='k', label='N_obs - Background')

        # ax2.bar(bin_centers, diff, width=bin_widths, color='purple', align='center', label='N_obs - Background')
        ax2.bar(bin_centers, signal_fit, width=bin_widths, color='g', align='center', alpha=0.5, label='Signal')

        ax2.set_xlabel('Score')
        ax2.set_ylabel('Counts')
        ax2.legend()

        # Save the plot
        if save_name:
            plt.savefig(save_name)
        plt.show()

    def visualize_fit(self, alpha_list, array, coefficient_list, alpha_name=None, log_y=False, save_name=None):
        # Prepare the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex='col')

        # Plotting for s_array
        for i in range(self.bins):
            coef_s = coefficient_list[i]
            poly_s = np.poly1d(coef_s)

            # Offset alpha values for each bin
            current_alpha = np.linspace(0.1, 0.9, len(alpha_list)) + i

            # Plot original data points and fitted curve
            ax1.plot(current_alpha, array[i], 'o', alpha=0.5, label=f'Bin {i + 1} data')
            ax1.plot(current_alpha, poly_s(alpha_list), '-', label=f'Bin {i + 1} fit')

            # Calculate relative error and plot in the second subplot
            fitted_values = poly_s(alpha_list)
            relative_error = (array[i] - fitted_values) / array[i] * 100
            ax2.plot(current_alpha, relative_error, 'o', alpha=0.5, label=f'Bin {i + 1} error')

            # print the value for array[i] and fitted_values for alpha_list == 1
            print(
                f'[*] - Bin {i + 1} - raw: {array[i][5]:.2f}, fitted: {fitted_values[5]:.2f}, relative error: {relative_error[5]:.2f}%')

        ax1.set_ylabel('MVA distribution')
        # ax1.legend(loc='upper right')
        if log_y:
            ax1.set_yscale('log')

        ax2.set_ylabel('Relative Error [%]')
        # ax2.set_xlabel('Extended alpha')
        ax2.axhline(y=0, color='grey', linestyle='--')  # Add a dashed grey line at y=0
        # ax2.legend(loc='upper right')

        for i in range(self.bins):
            ax1.axvline(x=i, color='grey', linestyle='--', alpha=0.25)
            ax2.axvline(x=i, color='grey', linestyle='--', alpha=0.25)

        # add annotation for ax1
        if alpha_name is not None:
            ax1.text(
                0.95, 0.95, alpha_name,
                transform=ax1.transAxes,
                fontsize=20,
                verticalalignment='top',
                horizontalalignment='right'
            )

        plt.tight_layout()

        if save_name:
            fig.savefig(save_name)
        plt.show()
