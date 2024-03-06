import numpy as np
from iminuit import Minuit
import matplotlib.pyplot as plt
import os
# import mplhep as hep

# hep.style.use('ATLAS')



def compute_result(N_obs,asimov_dict=None,sigma_mu_hat=0.0,SYST=False,PLOT=False):

    bins = len(N_obs)

    gamma_roi = asimov_dict["gamma_roi"]
    beta_roi = asimov_dict["beta_roi"]

    if SYST:
        def sigma_asimov(mu,alpha):

            Gamma_list = asimov_dict["gamma_roi"]
            Beta_list = asimov_dict["beta_roi"]

            gamma_roi = np.zeros(bins)
            beta_roi = np.zeros(bins)
            for i in range(bins):
                Gamma = Gamma_list[i]
                Beta = Beta_list[i]

                # print(f"[*] --- Gamma: {Gamma}")
                # print(f"[*] --- Beta: {Beta}")

                gamma_roi[i] = Gamma(alpha)
                beta_roi[i] = Beta(alpha)

            return mu*gamma_roi + beta_roi
        
    else:
        def sigma_asimov(mu,alpha):
            return mu*gamma_roi + alpha*beta_roi

    
    def NLL(mu,alpha):

        sigma_asimov_mu = sigma_asimov(mu,alpha)
    
        hist_llr = (
            - N_obs
            * np.log((sigma_asimov_mu ))
        ) + (sigma_asimov_mu)

        return hist_llr.sum()

    def NLL_minuit_mu(mu):
        return NLL(mu,alpha = 1.0)
    
    if SYST:
        result = Minuit(NLL, mu=1.0, alpha=1.0)

        result.errordef = Minuit.LIKELIHOOD
        plt.show()
        result.migrad()
        alpha = result.values['alpha']

    else:
        result = Minuit(NLL_minuit_mu, mu=1.0)

        result.errordef = Minuit.LIKELIHOOD
        result.migrad()

        alpha = 1.0

    if PLOT:
        _, ax = plt.subplots()
        result.draw_mnprofile("mu")
        hep.atlas.text(loc=1, text='Internal')
        plt.show()
        
        result.draw_mnmatrix(cl=[1,2])

    mu_hat = result.values['mu']
    mu_p16 = mu_hat - sigma_mu_hat - result.errors['mu']
    mu_p84 = mu_hat + sigma_mu_hat + result.errors['mu']

    return mu_hat, mu_p16, mu_p84, alpha

def plot_score(test_hist,hist_s,hist_b,mu_hat,bins,threshold=0,save_path=f"XGB_score.png"):

    _, ax = plt.subplots()
    
    print (f"[*] --- len(hist): {len(test_hist)}")
    print (f"[*] --- len(bins): {len(bins)}")
    
    
    plt.stairs(hist_s * mu_hat + hist_b, bins, fill=False, 
                color='orange', label=f"$H \\rightarrow \\tau \\tau (\mu = {mu_hat:.3f})$")
    
    plt.stairs(hist_s + hist_b, bins, fill=False, 
                color='red', label=f"$H \\rightarrow \\tau \\tau (\mu = {1.0:.3f})$")
    
    plt.stairs(hist_b, bins,fill=True, color='blue',
                label="$Z \\rightarrow \\tau \\tau$")
    
    yerr = np.sqrt(test_hist)

    xerr = (bins[1] - bins[0]) / 2

    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, test_hist, yerr = yerr,xerr = xerr, fmt='o', c='k', label=f'pseudo-data')

    plt.vlines(threshold, 1e3, 1.5e4, colors='k', linestyles='dashed', label='Threshold')

    plt.legend(loc='upper right')
    plt.xlabel(" BDT Score")
    plt.xlim(0,1)
    plt.ylabel(" Events ")
    plt.ylim(1e3, 1e6)
    ax.set_yscale('log')
    plot_file = os.path.join(save_path)
    # plt.savefig(plot_file)
    plt.show()
    plt.close()


if __name__ == "__main__":
    mu = 1.0
    alpha = 1.0
    beta_roi = 1.0
    gamma_roi = 1.0
    test_hist = np.random.rand(10)
    print(compute_result(test_hist,beta_roi,gamma_roi))
