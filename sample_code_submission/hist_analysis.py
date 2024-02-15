import numpy as np
from iminuit import Minuit
import matplotlib.pyplot as plt



def compute_result(holdout_signal_hist,holdout_background_hist,N_obs):

    def sigma_asimov(mu,alpha,beta_roi,gamma_roi):
        return mu*gamma_roi + alpha*beta_roi

    
    def NLL(mu,alpha,beta_roi,gamma_roi):

        sigma_asimov_mu = sigma_asimov(mu,alpha,beta_roi,gamma_roi)
    
        hist_llr = (
            - N_obs
            * np.log((sigma_asimov_mu ))
        ) + (sigma_asimov_mu)

        return hist_llr.sum()

    def NLL_minuit_mu(mu,alpha):
        par = alpha
        # par = 1.0
        return NLL(mu,par,holdout_background_hist,holdout_signal_hist)
    
    
    result = Minuit(NLL_minuit_mu, mu=1.0, alpha=1.0)
    result.errordef = Minuit.LIKELIHOOD
    # result.draw_mnprofile("mu")
    # plt.show()
    result.migrad()

    mu_hat = result.values['mu']

    sigma_mu_hat = 0
    mu_p16 = mu_hat - sigma_mu_hat - result.errors['mu']
    mu_p84 = mu_hat + sigma_mu_hat + result.errors['mu']

    return mu_hat, mu_p16, mu_p84

def compute_results_syst(N_obs,alpha_fun_dict=None):

    bins = len(N_obs)
    def sigma_asimov(mu,alpha):

        Gamma_list = alpha_fun_dict["gamma_roi"]
        Beta_list = alpha_fun_dict["beta_roi"]

        gamma_roi = np.zeros(bins)
        beta_roi = np.zeros(bins)
        for i in range(bins):
            Gamma = Gamma_list[i]
            Beta = Beta_list[i]

            # print(f"[*] --- Gamma: {Gamma}")
            # print(f"[*] --- Beta: {Beta}")

            gamma_roi[i] = Gamma(alpha)
            beta_roi[i] = Beta(alpha)

        return mu*gamma_roi + alpha*beta_roi


    def NLL(mu,alpha):

        sigma_asimov_mu = sigma_asimov(mu,alpha)
    
        hist_llr = (
            - N_obs
            * np.log((sigma_asimov_mu ))
        ) + (sigma_asimov_mu)

        return hist_llr.sum()

    def NLL_minuit_mu(mu,alpha):
        par = alpha
        # par = 1.0
        return NLL(mu,par)
    
    
    result = Minuit(NLL_minuit_mu, mu=1.0, alpha=1.0)
    result.errordef = Minuit.LIKELIHOOD
    # result.draw_mnprofile("mu")
    # plt.show()
    result.migrad()

    mu_hat = result.values['mu']

    mu_p16 = mu_hat  - (result.errors['mu']**2 + result.errors['alpha']**2)**0.5
    mu_p84 = mu_hat + (result.errors['mu']**2 + result.errors['alpha']**2)**0.5


    return mu_hat, mu_p16, mu_p84


if __name__ == "__main__":
    mu = 1.0
    alpha = 1.0
    beta_roi = 1.0
    gamma_roi = 1.0
    test_hist = np.random.rand(10)
    print(compute_result(test_hist,beta_roi,gamma_roi))
