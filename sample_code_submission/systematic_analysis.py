import numpy as np


Plotteur = "False"

nb_points_soft_met_for_fitting = 15
nb_points_tes_for_fitting=1000
nb_points_jes_for_fitting=10


n_jobs_distrib=18

from joblib import Parallel, delayed
from iminuit import Minuit
# Automatic error propagation at each x
from iminuit.util import propagate



def Polynomial_Reg_Model_soft_met(x, c, a1, a2):
    return c + a1 * x + a2 * x * x  # +a3*x*x*x#+a4*x*x*x*x#+a5*x*x*x*x*x


def Polynomial_Reg_Model(x, c, a1, a2):
    return c + a1 * x + a2 * x * x


def Polynomial_Reg_Model_forced_jes_tes(x, a2, b1  ):
    return (-a2 - b1) + b1 * x + a2 * x * x


def Polynomial_Reg_Model_forced_soft_met(x, a2, b1   ):
    return b1 * x + a2 * x * x  

#########################################################
#Classic Sig VS bkg 
#########################################################
def regression_tes(dataset, model, systematics, nb_bins=20, threshold=0):
    import matplotlib.pyplot as plt
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))


    #################Be careful
    if "score" in dataset["data"].columns:
            dataset["data"] = dataset["data"].drop(columns=["score"])
    ######################

    dataset_tamp=dataset.copy()
    dataset_tamp = systematics(dataset_tamp, tes=1)
    if "score" in dataset_tamp["data"].columns:
        dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
    data_score = model.predict(dataset_tamp["data"])

    weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
    label_Roiscore = dataset_tamp["labels"][data_score > threshold]
    score_ROIscore = data_score[data_score > threshold]
    Bins_edges = np.linspace(0, 1, nb_bins + 1)  # A changer au besoin
    signal_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 1],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 1],
    )[0]
    bkg_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 0],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 0],
    )[0]
    N_obs_ref = np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[
        0
    ]
    
    delta_sig_ref =np.array([ np.sqrt( np.sum( np.power(weight_ROIscore[ (
        (score_ROIscore >= Bins_edges[j])
        & (score_ROIscore < Bins_edges[j + 1])
        & (label_Roiscore == 1) )],2) ) ) for j in range(nb_bins) ])

    delta_bkg_ref =np.array([ np.sqrt( np.sum( np.power(weight_ROIscore[ (
        (score_ROIscore >= Bins_edges[j])
        & (score_ROIscore < Bins_edges[j + 1])
        & (label_Roiscore == 0) ) ],2) ) ) for j in range(nb_bins) ])
    
    delta_N_ref =np.array([ np.sqrt( np.sum( np.power(weight_ROIscore[ (
        (score_ROIscore >= Bins_edges[j])
        & (score_ROIscore < Bins_edges[j + 1])
        )],2) ) ) for j in range(nb_bins) ])

    sigma = np.linspace(-0.106, 0.096, nb_points_tes_for_fitting)
    var_lenght=len(sigma)
    tes = [np.exp(sigma[i]) for i in range(var_lenght)]

    # data_score=[model.predict(systematics(dataset,tes=tes[i]))  for i in range(len(sigma)) ]  #Alternative to the for loop
    # weight_ROIscore=[dataset["weights"][data_score[i] > threshold] for i in range(len(sigma))]
    # label_Roiscore=[dataset["labels"][data_score[i]>threshold] for i in range(len(sigma))]
    # score_ROIscore=[data_score[data_score[i]>threshold]  for i in range(len(sigma))]
    # Bins_edges=[np.linspace(np.min(score_ROIscore[i]),np.max(score_ROIscore[i]),nb_bins+1) for i in range(len(sigma))]
    # signal_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==1], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==1])[0] for i in range(len(sigma))]
    # bkg_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==0], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==0])[0] for i in range(len(sigma))]
    # N_obs=[np.histogram(score_ROIscore[i], bins=Bins_edges[i], weights=weight_ROIscore[i])[0] for i in range(len(sigma))]

    # We create the modified observed list so we can fit on
    def tes_hist_syst_points(i):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp = systematics(dataset, tes=tes[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        label_Roiscore = dataset_tamp["labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
        signal = (
            np.histogram(
                score_ROIscore[label_Roiscore == 1],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 1],
            )[0]
            - signal_obs_ref
        )  # 3 lignes below new
        bkg = (
            np.histogram(
                score_ROIscore[label_Roiscore == 0],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 0],
            )[0]
            - bkg_obs_ref
        )
        N = (
            np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[0]
            - N_obs_ref
        )


        d_sig =np.array([ np.sqrt( np.sum( np.power(weight_ROIscore[ (
        (score_ROIscore >= Bins_edges[j])
        & (score_ROIscore < Bins_edges[j + 1])
        & (label_Roiscore == 1) )],2) ) ) for j in range(nb_bins) ])

        d_bkg =np.array([ np.sqrt( np.sum( np.power(weight_ROIscore[ (
        (score_ROIscore >= Bins_edges[j])
        & (score_ROIscore < Bins_edges[j + 1])
        & (label_Roiscore == 0) )],2) ) ) for j in range(nb_bins) ])

        d_N=np.array([ np.sqrt( np.sum( np.power(weight_ROIscore[ (
        (score_ROIscore >= Bins_edges[j])
        & (score_ROIscore < Bins_edges[j + 1])
        )],2) ) ) for j in range(nb_bins) ])

        return signal, bkg, N , d_sig, d_bkg,d_N

    results_tes_hist_syst_points = Parallel(n_jobs=n_jobs_distrib)(delayed(tes_hist_syst_points)(i) for i in range(var_lenght))
    print("End of the tes hist")
    del dataset, dataset_tamp

    signal_obs = [None] * var_lenght
    bkg_obs    = [None] * var_lenght
    N_obs      = [None] * var_lenght
    delta_sig  = [None] * var_lenght
    delta_bkg  = [None] * var_lenght
    
    delta_N = [None] * var_lenght

    for i, (signal, bkg, N, d_sig, d_bkg,d_N) in enumerate(results_tes_hist_syst_points):
        signal_obs[i] = signal
        bkg_obs[i]    = bkg
        N_obs[i]      = N
        delta_sig[i] = d_sig
        delta_bkg[i] = d_bkg
        delta_N[i]   = d_N


    y_obs_name = ["signal_obs", "bkg_obs", " total_obs"]
    signal_obs_order = np.array( [
        [signal_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)
        ])
    bkg_obs_order = np.array( [[bkg_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)])
    N_obs_order = np.array([[N_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)])

    delta_sig_order =np.array([[delta_sig[j][i] for j in range(var_lenght)] for i in range(nb_bins)])
    delta_bkg_order =np.array([[delta_bkg[j][i] for j in range(var_lenght)] for i in range(nb_bins)])
    delta_N_order =np.array([[delta_N[j][i] for j in range(var_lenght)] for i in range(nb_bins)])

    delta_sig_ref_order=np.array([[delta_sig_ref[i] for j in range(var_lenght)] for i in range(nb_bins)] )
    delta_bkg_ref_order=np.array([[delta_bkg_ref[i] for j in range(var_lenght)] for i in range(nb_bins)] )
    delta_N_ref_order=np.array([[delta_N_ref[i] for j in range(var_lenght)] for i in range(nb_bins)] )

    y_obs = [signal_obs_order, bkg_obs_order, N_obs_order]
    delta_y_obs=[delta_sig_order,delta_bkg_order,delta_N_order ]
    delta_y_ref=[delta_sig_ref_order,delta_bkg_ref_order,delta_N_ref_order ]
    """
    y_obs[0:2] = sig, bkg, tot
    y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un TES précis*
    """
    #for i in range(len(y_obs)):  # =3
    #for j in range(nb_bins):
    def tes_fit_and_plot_over_bins (i,j) :
        # Fit
        x_data = np.array(tes)
        y_data = y_obs[i][j]
        
        def chi2(a, b):
            y_fit = Polynomial_Reg_Model_forced_jes_tes(x_data, a, b)
            return np.sum((y_data - y_fit)**2)

        # Minimize
        m = Minuit(chi2, a=0, b=0)
        m.errordef = Minuit.LEAST_SQUARES
        m.migrad()
        m.hesse() 

        fit_params = [m.values["a"], m.values["b"]]
        fit_cov = m.covariance

        tes_fit = np.linspace(min(tes), max(tes), 40)
        y_fit = Polynomial_Reg_Model_forced_jes_tes(tes_fit, *(fit_params))


        results_propa = [propagate(lambda p: Polynomial_Reg_Model_forced_jes_tes(xi, p[0], p[1]), m.values, fit_cov) for xi in tes_fit]

        y = np.array([r[0] for r in results_propa])
        ycov = np.array([r[1] for r in results_propa])  # shape (len(tes_fit), 1, 1)
        y_std = ycov**0.5
        
        # Plot
        if Plotteur == "True" and (j in [2 ,nb_bins-2]):

            plt.scatter(
                tes, y_obs[i][j], label="%s" % (y_obs_name[i]), color="dodgerblue", s=4,
            )
            plt.fill_between (
                tes,
                y_obs[i][j] - delta_y_obs[i][j]-delta_y_ref[i][j],
                y_obs[i][j] + delta_y_obs[i][j]+delta_y_ref[i][j],
                color="blue",
                alpha=0.3,
                label="Data total uncertainty",
            )
            plt.fill_between (
                tes,
                y_obs[i][j] - delta_y_obs[i][j],
                y_obs[i][j] + delta_y_obs[i][j],
                color="black",
                alpha=0.5,
                label="syst data uncertainty (no ref)",
            )
            plt.plot(tes_fit, y_fit, color="darkorange", label="Fit")
            plt.fill_between(
                tes_fit,
                y - y_std,
                y + y_std,
                color="orange",
                alpha=0.3,
                label="Fitting uncertainty",
            )
            plt.ylabel("data observed")
            plt.xlabel("Tes factor")
            plt.title("Data observed with a variation of Tes (bin n°%s)" % (j + 1))
            plt.legend()
            plt.grid()

            if not os.path.exists("%s/Images/Fitting/TES/one_bkg"%(current_dir)):
                os.makedirs("%s/Images/Fitting/TES/one_bkg"%(current_dir))
            plt.savefig("%s/Images/Fitting/TES/one_bkg/TES_%s_Bins%s_over_%s_1bkg"%(current_dir,y_obs_name[i],j,nb_bins))
            plt.close()
        
        return j, fit_params, fit_cov
    

    fit_param = [[None] * len(y_obs) for i in range(nb_bins)]
    fit_cov = [[None] * len(y_obs) for i in range(nb_bins)]

    for i in range(len(y_obs)):
        results = Parallel(n_jobs=n_jobs_distrib)(
            delayed(tes_fit_and_plot_over_bins)(i, j) for j in range(nb_bins)
            )
        
        for j, params, cov in results:
            fit_param[j][i] = params
            fit_cov[j][i] = cov
    
    print("End of the fit of tes")

    # Save fiting param
    if not os.path.exists("%s/Fitting_Parameters/one_bkg"%(current_dir) ):
        os.makedirs("%s/Fitting_Parameters/one_bkg"%(current_dir))

    np.savez(
        "%s/Fitting_Parameters/one_bkg/TES_%sbins_2orderFittingParam.npz" % (current_dir,nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])



def regression_jes(dataset, model, systematics, nb_bins=20, threshold=0):
    from utils import histogram_dataset
    from statistical_analysis import calculate_saved_info
    import matplotlib.pyplot as plt

    dataset_tamp=dataset
    dataset_tamp = systematics(dataset_tamp, jes=1)
    if "score" in dataset_tamp["data"].columns:
        dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
    data_score = model.predict(dataset_tamp["data"])

    weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
    label_Roiscore = dataset_tamp["labels"][data_score > threshold]
    score_ROIscore = data_score[data_score > threshold]
    Bins_edges = np.linspace(0, 1, nb_bins + 1)  # A changer au besoin
    signal_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 1],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 1],
    )[0]
    bkg_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 0],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 0],
    )[0]
    N_obs_ref = np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[
        0
    ]
    

    sigma = np.linspace(-0.106, 0.096, nb_points_jes_for_fitting)
    var_lenght=len(sigma)
    jes = [np.exp(sigma[i]) for i in range(var_lenght)]
    signal_obs = [None] * var_lenght
    bkg_obs = [None] * var_lenght
    N_obs = [None] * var_lenght

    # data_score=[model.predict(systematics(dataset,jes=jes[i]))  for i in range(len(sigma)) ]  #Alternative to the for loop
    # weight_ROIscore=[dataset["weights"][data_score[i] > threshold] for i in range(len(sigma))]
    # label_Roiscore=[dataset["labels"][data_score[i]>threshold] for i in range(len(sigma))]
    # score_ROIscore=[data_score[data_score[i]>threshold]  for i in range(len(sigma))]
    # Bins_edges=[np.linspace(np.min(score_ROIscore[i]),np.max(score_ROIscore[i]),nb_bins+1) for i in range(len(sigma))]
    # signal_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==1], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==1])[0] for i in range(len(sigma))]
    # bkg_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==0], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==0])[0] for i in range(len(sigma))]
    # N_obs=[np.histogram(score_ROIscore[i], bins=Bins_edges[i], weights=weight_ROIscore[i])[0] for i in range(len(sigma))]

    # We create the modified observed list so we can fit on
    for i in range(var_lenght):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp = systematics(dataset, jes=jes[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        label_Roiscore = dataset_tamp["labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
        signal_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 1],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 1],
            )[0]
            - signal_obs_ref
        )  # 3 lignes below new
        bkg_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 0],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 0],
            )[0]
            - bkg_obs_ref
        )
        N_obs[i] = (
            np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[0]
            - N_obs_ref
        )

    y_obs_name = ["signal obs", "bkg obs", " total obs"]
    signal_obs_order = [
        [signal_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)
    ]
    bkg_obs_order = [[bkg_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    N_obs_order = [[N_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    y_obs = [signal_obs_order, bkg_obs_order, N_obs_order]
    # y_obs[0:2] = sig, bkg, tot
    # y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    # y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un JES précis

    fit_param = [[None] * len(y_obs) for i in range(nb_bins)]
    fit_cov = [[None] * len(y_obs) for i in range(nb_bins)]
    for i in range(len(y_obs)):  # =3
        for j in range(nb_bins):
            # Fit
            from iminuit import Minuit

            
            x_data = np.array(jes)
            y_data = y_obs[i][j]
           
            
            def chi2(a, b  ):
                y_fit = Polynomial_Reg_Model_forced_jes_tes(x_data, a, b )
                return np.sum((y_data - y_fit)**2)

            
            m = Minuit(chi2, a=0, b=0 )
            m.errordef = Minuit.LEAST_SQUARES
            m.migrad()
            m.hesse() 

            fit_param[j][i] = [m.values["a"],m.values["b"]  ]
            print(fit_param[j][i])
            fit_cov[j][i] =  m.covariance


            jes_fit = np.linspace(min(jes), max(jes), 40)
            y_fit = Polynomial_Reg_Model_forced_jes_tes(jes_fit, *(fit_param[j][i]))

            # Automatic error propagation at each x
            from iminuit.util import propagate

            # y_std = np.array([
            #     propagate(lambda a, b: model(xi, a, b), m.values, fit_cov[j][i])
            #     for xi in jes_fit])   

            results = [propagate(lambda p: Polynomial_Reg_Model_forced_jes_tes(xi, p[0], p[1]), m.values, fit_cov[j][i]) for xi in jes_fit]

            y = np.array([r[0] for r in results])
            ycov = np.array([r[1] for r in results])  # shape (len(jes_fit), 1, 1)

            y_std = ycov**0.5
            


                            # fit_param_tamp, fit_cov_tamp = curve_fit(
                            #     Polynomial_Reg_Model_forced_jes_tes, jes, y_obs[i][j]
                            # )
                            # fit_param[j][i] = fit_param_tamp
                            # fit_cov[j][i] = fit_cov_tamp

                            # # popt = best-fit parameters [a0, a1, a2]
                            # # pcov = covariance matrix of parameters

                            # # Calculate uncertainty (standard deviation) of parameters
                            # param_err = np.sqrt(np.diag(fit_cov[j][i]))

                            # print("Fit parameters:", fit_param[j][i])
                            # print("Parameter uncertainties:", param_err)

                            # # Predict values
                            # jes_fit = np.linspace(min(jes), max(jes), 200)
                            # y_fit = Polynomial_Reg_Model_forced_jes_tes(jes_fit, *(fit_param[j][i]))

                            # # To get uncertainty on the fit curve, propagate errors:
                            # # Compute Jacobian matrix at each x_fit
                            # J = np.vstack(
                            #     [jes_fit**k for k in range(len(fit_param[j][i]))]
                            # ).T  # shape (num_points, 4)
                            # print("J:", J)

                            # # y_var = np.sum(J @ fit_cov[j][i] * J, axis=1)
                            # y_var = np.einsum("ij,jk,ik->i", J, fit_cov[j][i], J)
                            
                            # y_std = np.sqrt(y_var)


            # Plot

            if Plotteur == "True" and (j in [0,10,nb_bins-2]):

                plt.scatter(
                    jes, y_obs[i][j], label="%s" % (y_obs_name[i]), color="dodgerblue"
                )
                plt.plot(jes_fit, y_fit, color="darkorange", label="Fit")
                plt.fill_between(
                    jes_fit,
                    y - y_std,
                    y + y_std,
                    color="orange",
                    alpha=0.3,
                    label="Fitting uncertainty",
                )
                plt.ylabel("data observed")
                plt.xlabel("jes factor")
                plt.title("Data observed with a variation of Jes (bin n°%s)" % (j + 1))
                plt.legend()
                plt.grid()
                plt.show()

    # Save fiting param
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/Fitting_Parameters/one_bkg"%(current_dir)):
        os.makedirs("current_dir/Fitting_Parameters/one_bkg")
    np.savez(
        "%s/Fitting_Parameters/one_bkg/JES_%sbins_2orderFittingParam.npz" % (current_dir,nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])






def regression_soft_met(dataset, model, systematics, nb_bins=20, threshold=0):
    from utils import histogram_dataset
    from statistical_analysis import calculate_saved_info
    import matplotlib.pyplot as plt

    dataset_tamp=dataset
    dataset_tamp = systematics(dataset_tamp, soft_met=0)
    if "score" in dataset_tamp["data"].columns:
        dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
    data_score = model.predict(dataset_tamp["data"])

    weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
    label_Roiscore = dataset_tamp["labels"][data_score > threshold]
    score_ROIscore = data_score[data_score > threshold]
    Bins_edges = np.linspace(0, 1, nb_bins + 1)  # A changer au besoin
    signal_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 1],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 1],
    )[0]
    bkg_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 0],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 0],
    )[0]
    N_obs_ref = np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[
        0
    ]
    

    soft_met = np.linspace(0, 5, nb_points_soft_met_for_fitting)
    var_lenght=len(soft_met)
    signal_obs = [None] * var_lenght
    bkg_obs = [None] * var_lenght
    N_obs = [None] * var_lenght

    # data_score=[model.predict(systematics(dataset,soft_met=soft_met[i]))  for i in range(len(sigma)) ]  #Alternative to the for loop
    # weight_ROIscore=[dataset["weights"][data_score[i] > threshold] for i in range(len(sigma))]
    # label_Roiscore=[dataset["labels"][data_score[i]>threshold] for i in range(len(sigma))]
    # score_ROIscore=[data_score[data_score[i]>threshold]  for i in range(len(sigma))]
    # Bins_edges=[np.linspace(np.min(score_ROIscore[i]),np.max(score_ROIscore[i]),nb_bins+1) for i in range(len(sigma))]
    # signal_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==1], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==1])[0] for i in range(len(sigma))]
    # bkg_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==0], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==0])[0] for i in range(len(sigma))]
    # N_obs=[np.histogram(score_ROIscore[i], bins=Bins_edges[i], weights=weight_ROIscore[i])[0] for i in range(len(sigma))]

    # We create the modified observed list so we can fit on
    for i in range(var_lenght):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp = systematics(dataset, soft_met=soft_met[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        label_Roiscore = dataset_tamp["labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
        signal_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 1],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 1],
            )[0]
            - signal_obs_ref
        )  # 3 lignes below new
        bkg_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 0],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 0],
            )[0]
            - bkg_obs_ref
        )
        N_obs[i] = (
            np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[0]
            - N_obs_ref
        )

    y_obs_name = ["signal obs", "bkg obs", " total obs"]
    signal_obs_order = [
        [signal_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)
    ]
    bkg_obs_order = [[bkg_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    N_obs_order = [[N_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    y_obs = [signal_obs_order, bkg_obs_order, N_obs_order]
    # y_obs[0:2] = sig, bkg, tot
    # y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    # y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un soft met précis

    fit_param = [[None] * len(y_obs) for i in range(nb_bins)]
    fit_cov = [[None] * len(y_obs) for i in range(nb_bins)]
    for i in range(len(y_obs)):  # =3
        for j in range(nb_bins):
            # Fit
            from iminuit import Minuit

            
            x_data = np.array(soft_met)
            y_data = y_obs[i][j]
           
            
            def chi2(a, b):
                y_fit = Polynomial_Reg_Model_forced_soft_met(x_data, a, b )
                return np.sum((y_data - y_fit)**2)

            
            m = Minuit(chi2, a=0, b=0 )
            m.errordef = Minuit.LEAST_SQUARES
            m.migrad()
            m.hesse() 

            fit_param[j][i] = [m.values["a"],m.values["b"]]
            fit_cov[j][i] =  m.covariance


            soft_met_fit = np.linspace(min(soft_met), max(soft_met), 60)
            y_fit = Polynomial_Reg_Model_forced_soft_met(soft_met_fit, *(fit_param[j][i]))

            # Automatic error propagation at each x
            from iminuit.util import propagate

            # y_std = np.array([
            #     propagate(lambda a, b: model(xi, a, b), m.values, fit_cov[j][i])
            #     for xi in soft_met_fit])   

            results = [propagate(lambda p: Polynomial_Reg_Model_forced_soft_met(xi, p[0], p[1]), m.values, fit_cov[j][i]) for xi in soft_met_fit]

            y = np.array([r[0] for r in results])
            ycov = np.array([r[1] for r in results])  # shape (len(soft_met_fit), 1, 1)import matplotlib.pyplot as plt

            y_std = ycov**0.5
            


                            # fit_param_tamp, fit_cov_tamp = curve_fit(
                            #     Polynomial_Reg_Model_forced_soft_met, soft_met, y_obs[i][j]
                            # )
                            # fit_param[j][i] = fit_param_tamp
                            # fit_cov[j][i] = fit_cov_tamp

                            # # popt = best-fit parameters [a0, a1, a2]
                            # # pcov = covariance matrix of parameters

                            # # Calculate uncertainty (standard deviation) of parameters
                            # param_err = np.sqrt(np.diag(fit_cov[j][i]))

                            # print("Fit parameters:", fit_param[j][i])
                            # print("Parameter uncertainties:", param_err)

                            # # Predict values
                            # soft_met_fit = np.linspace(min(soft_met), max(soft_met), 200)
                            # y_fit = Polynomial_Reg_Model_forced_soft_met(soft_met_fit, *(fit_param[j][i]))

                            # # To get uncertainty on the fit curve, propagate errors:
                            # # Compute Jacobian matrix at each x_fit
                            # J = np.vstack(
                            #     [soft_met_fit**k for k in range(len(fit_param[j][i]))]
                            # ).T  # shape (num_points, 4)
                            # print("J:", J)

                            # # y_var = np.sum(J @ fit_cov[j][i] * J, axis=1)
                            # y_var = np.einsum("ij,jk,ik->i", J, fit_cov[j][i], J)
                            
                            # y_std = np.sqrt(y_var)


            # Plot

            if Plotteur == "True" and (j in [2,11,nb_bins-2]):
                plt.scatter(
                    soft_met, y_obs[i][j], label="%s" % (y_obs_name[i]), color="dodgerblue"
                )
                plt.plot(soft_met_fit, y_fit, color="darkorange", label="Fit")
                plt.fill_between(
                    soft_met_fit,
                    y - y_std,
                    y + y_std,
                    color="orange",
                    alpha=0.3,
                    label="Fitting uncertainty",
                )
                plt.ylabel("data observed")
                plt.xlabel("soft met factor")
                plt.title("Data observed with a variation of soft met (bin n°%s)" % (j + 1))
                plt.legend()
                plt.grid()
                plt.show()

    # Save fiting param
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/Fitting_Parameters/one_bkg"%(current_dir)):
        os.makedirs("%s/Fitting_Parameters/one_bkg"%(current_dir))
    np.savez(
        "%s/Fitting_Parameters/one_bkg/SOFTMET_%sbins_2orderFittingParam.npz" % (current_dir,nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])


#########################################################
#Normalization bkg : Sig Vs Diboson VS Ztautau VS ttbar
#########################################################

def regression_tes_3bkg (dataset, model, systematics, nb_bins=20, threshold=0):
    from utils import histogram_dataset
    from statistical_analysis import calculate_saved_info
    import matplotlib.pyplot as plt
    
    dataset_tamp=dataset
    dataset_tamp = systematics(dataset_tamp, tes=1)
    if "score" in dataset_tamp["data"].columns:
        dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
    data_score = model.predict(dataset_tamp["data"])

    weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
    detailed_labels_Roiscore = dataset_tamp["detailed_labels"][data_score > threshold]
    score_ROIscore = data_score[data_score > threshold]
    bkg_channel_list=["ztautau" , "ttbar", "diboson"]
    
    Bins_edges = np.linspace(0, 1, nb_bins + 1)  # A changer au besoin
    signal_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "htautau"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "htautau"],
    )[0]
    ztautau_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "ztautau"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "ztautau"],
    )[0]
    ttbar_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "ttbar"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "ttbar"],
    )[0]
    diboson_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "diboson"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "diboson"],
    )[0]
    N_obs_ref = np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[
        0
    ]
    

    sigma = np.linspace(-0.106, 0.096, nb_points_tes_for_fitting)
    var_lenght=len(sigma)
    tes = [np.exp(sigma[i]) for i in range(var_lenght)]
    signal_obs = [None] * var_lenght
    ztautau_obs = [None] * var_lenght
    ttbar_obs = [None] * var_lenght
    diboson_obs = [None] * var_lenght
    N_obs = [None] * var_lenght

    # We create the modified observed list so we can fit on
    for i in range(var_lenght):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp=dataset
        dataset_tamp = systematics(dataset_tamp, tes=tes[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        detailed_labels_Roiscore = dataset_tamp["detailed_labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
        signal_obs[i] = ( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "htautau"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "htautau"],
            )[0] 
            -signal_obs_ref
                     )
        ztautau_obs[i] = ( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "ztautau"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "ztautau"],
            )[0] 
            -ztautau_obs_ref
                      )
        ttbar_obs[i] =( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "ttbar"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "ttbar"],
            )[0] 
            -ttbar_obs_ref 
                   )
        diboson_obs[i] =( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "diboson"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "diboson"],
            )[0] 
            -diboson_obs_ref
                    )
            
        N_obs[i] = (
            np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[0]
            - N_obs_ref
        )
   
    y_obs_name = ["signal (htautau) obs", "ztautau obs","ttbar obs","diboson obs", " total obs"]
    signal_obs_order = [[signal_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    ztautau_obs_order = [[ztautau_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    ttbar_obs_order = [[ttbar_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    diboson_obs_order = [[diboson_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    N_obs_order = [[N_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    y_obs = [signal_obs_order, ztautau_obs_order,ttbar_obs_order,diboson_obs_order, N_obs_order]
    # y_obs[0:2] = sig, bkg, tot
    # y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    # y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un TES précis

    fit_param = [[None] * len(y_obs) for i in range(nb_bins)]
    fit_cov = [[None] * len(y_obs) for i in range(nb_bins)]
    for i in range(len(y_obs)):  # =5
        for j in range(nb_bins):
            # Fit
            from iminuit import Minuit

            
            x_data = np.array(tes)
            y_data = y_obs[i][j]
            
            def chi2(a, b):
                y_fit = Polynomial_Reg_Model_forced_jes_tes(x_data, a, b)
                return np.sum((y_data - y_fit)**2)

            # Minimize
            m = Minuit(chi2, a=0, b=0)
            m.errordef = Minuit.LEAST_SQUARES
            m.migrad()
            m.hesse() 

            fit_param[j][i] = [m.values["a"],m.values["b"]]
            fit_cov[j][i] =  m.covariance


            tes_fit = np.linspace(min(tes), max(tes), 40)
            y_fit = Polynomial_Reg_Model_forced_jes_tes(tes_fit, *(fit_param[j][i]))

            # Automatic error propagation at each x
            from iminuit.util import propagate

            # y_std = np.array([
            #     propagate(lambda a, b: model(xi, a, b), m.values, fit_cov[j][i])
            #     for xi in tes_fit])   

            results = [propagate(lambda p: Polynomial_Reg_Model_forced_jes_tes(xi, p[0], p[1]), m.values, fit_cov[j][i]) for xi in tes_fit]

            y = np.array([r[0] for r in results])
            ycov = np.array([r[1] for r in results])  # shape (len(tes_fit), 1, 1)

            y_std = ycov**0.5
            
            # Plot

            if Plotteur == "True" and (j in [2 ,nb_bins-2]):

                plt.scatter(
                    tes, y_obs[i][j], label="%s" % (y_obs_name[i]), color="dodgerblue"
                )
                plt.plot(tes_fit, y_fit, color="darkorange", label="Fit")
                plt.fill_between(
                    tes_fit,
                    y - y_std,
                    y + y_std,
                    color="orange",
                    alpha=0.3,
                    label="Fitting uncertainty",
                )
                plt.ylabel("data observed")
                plt.xlabel("TES factor")
                plt.title("Data observed with a variation of TES (bin n°%s)" % (j + 1))
                plt.legend()
                plt.grid()
                plt.show()

    # Save fiting param
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/Fitting_Parameters/bkg_subchannel"%(current_dir)):
        os.makedirs("%s/Fitting_Parameters/bkg_subchannel"%(current_dir))
    np.savez(
        "%s/Fitting_Parameters/bkg_subchannel/TES_3bkg_%sbins_2orderFittingParam.npz" % (current_dir,nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])



def regression_jes_3bkg (dataset, model, systematics, nb_bins=20, threshold=0):
    from utils import histogram_dataset
    from statistical_analysis import calculate_saved_info
    import matplotlib.pyplot as plt
    
    dataset_tamp=dataset
    dataset_tamp = systematics(dataset_tamp, jes=1)
    if "score" in dataset_tamp["data"].columns:
        dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
    data_score = model.predict(dataset_tamp["data"])

    weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
    detailed_labels_Roiscore = dataset_tamp["detailed_labels"][data_score > threshold]
    score_ROIscore = data_score[data_score > threshold]
    bkg_channel_list=["ztautau" , "ttbar", "diboson"]
    
    Bins_edges = np.linspace(0, 1, nb_bins + 1)  # A changer au besoin
    signal_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "htautau"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "htautau"],
    )[0]
    ztautau_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "ztautau"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "ztautau"],
    )[0]
    ttbar_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "ttbar"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "ttbar"],
    )[0]
    diboson_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "diboson"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "diboson"],
    )[0]
    N_obs_ref = np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[
        0
    ]
    

    sigma = np.linspace(-0.106, 0.096, nb_points_jes_for_fitting)
    var_lenght=len(sigma)
    jes = [np.exp(sigma[i]) for i in range(var_lenght)]
    signal_obs = [None] * var_lenght
    ztautau_obs = [None] * var_lenght
    ttbar_obs = [None] * var_lenght
    diboson_obs = [None] * var_lenght
    N_obs = [None] * var_lenght

    # We create the modified observed list so we can fit on
    for i in range(var_lenght):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp=dataset
        dataset_tamp = systematics(dataset_tamp, jes=jes[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        detailed_labels_Roiscore = dataset_tamp["detailed_labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
        signal_obs[i] = ( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "htautau"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "htautau"],
            )[0] 
            -signal_obs_ref
                     )
        ztautau_obs[i] = ( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "ztautau"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "ztautau"],
            )[0] 
            -ztautau_obs_ref
                      )
        ttbar_obs[i] =( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "ttbar"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "ttbar"],
            )[0] 
            -ttbar_obs_ref 
                   )
        diboson_obs[i] =( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "diboson"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "diboson"],
            )[0] 
            -diboson_obs_ref
                    )
            
        N_obs[i] = (
            np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[0]
            - N_obs_ref
        )
   
    y_obs_name = ["signal (htautau) obs", "ztautau obs","ttbar obs","diboson obs", " total obs"]
    signal_obs_order = [[signal_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    ztautau_obs_order = [[ztautau_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    ttbar_obs_order = [[ttbar_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    diboson_obs_order = [[diboson_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    N_obs_order = [[N_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    y_obs = [signal_obs_order, ztautau_obs_order,ttbar_obs_order,diboson_obs_order, N_obs_order]
    # y_obs[0:2] = sig, bkg, tot
    # y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    # y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un JES précis

    fit_param = [[None] * len(y_obs) for i in range(nb_bins)]
    fit_cov = [[None] * len(y_obs) for i in range(nb_bins)]
    for i in range(len(y_obs)):  # =5
        for j in range(nb_bins):
            # Fit
            from iminuit import Minuit

            
            x_data = np.array(jes)
            y_data = y_obs[i][j]
            
            def chi2(a, b):
                y_fit = Polynomial_Reg_Model_forced_jes_tes(x_data, a, b)
                return np.sum((y_data - y_fit)**2)

            # Minimize
            m = Minuit(chi2, a=0, b=0)
            m.errordef = Minuit.LEAST_SQUARES
            m.migrad()
            m.hesse() 

            fit_param[j][i] = [m.values["a"],m.values["b"]]
            fit_cov[j][i] =  m.covariance


            jes_fit = np.linspace(min(jes), max(jes), 40)
            y_fit = Polynomial_Reg_Model_forced_jes_tes(jes_fit, *(fit_param[j][i]))

            # Automatic error propagation at each x
            from iminuit.util import propagate

            # y_std = np.array([
            #     propagate(lambda a, b: model(xi, a, b), m.values, fit_cov[j][i])
            #     for xi in jes_fit])   

            results = [propagate(lambda p: Polynomial_Reg_Model_forced_jes_tes(xi, p[0], p[1]), m.values, fit_cov[j][i]) for xi in jes_fit]

            y = np.array([r[0] for r in results])
            ycov = np.array([r[1] for r in results])  # shape (len(jes_fit), 1, 1)

            y_std = ycov**0.5
            
            # Plot

            if Plotteur == "True" and (j in [2 ,nb_bins-2]):

                plt.scatter(
                    jes, y_obs[i][j], label="%s" % (y_obs_name[i]), color="dodgerblue"
                )
                plt.plot(jes_fit, y_fit, color="darkorange", label="Fit")
                plt.fill_between(
                    jes_fit,
                    y - y_std,
                    y + y_std,
                    color="orange",
                    alpha=0.3,
                    label="Fitting uncertainty",
                )
                plt.ylabel("data observed")
                plt.xlabel("JES factor")
                plt.title("Data observed with a variation of JES (bin n°%s)" % (j + 1))
                plt.legend()
                plt.grid()
                plt.show()

    # Save fiting param
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/Fitting_Parameters/bkg_subchannel"%(current_dir)):
        os.makedirs("%s/Fitting_Parameters/bkg_subchannel"%(current_dir))
    np.savez(
        "%s/Fitting_Parameters/bkg_subchannel/JES_3bkg_%sbins_2orderFittingParam.npz" % (current_dir,nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])


def regression_soft_met_3bkg (dataset, model, systematics, nb_bins=25, threshold=0):
    from utils import histogram_dataset
    from statistical_analysis import calculate_saved_info
    import matplotlib.pyplot as plt
    
    dataset_tamp=dataset
    dataset_tamp = systematics(dataset_tamp, soft_met=0)
    if "score" in dataset_tamp["data"].columns:
        dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
    data_score = model.predict(dataset_tamp["data"])

    weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
    detailed_labels_Roiscore = dataset_tamp["detailed_labels"][data_score > threshold]
    score_ROIscore = data_score[data_score > threshold]
    bkg_channel_list=["ztautau" , "ttbar", "diboson"]
    
    Bins_edges = np.linspace(0, 1, nb_bins + 1)  # A changer au besoin
    signal_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "htautau"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "htautau"],
    )[0]
    ztautau_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "ztautau"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "ztautau"],
    )[0]
    ttbar_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "ttbar"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "ttbar"],
    )[0]
    diboson_obs_ref = np.histogram(
        score_ROIscore[detailed_labels_Roiscore == "diboson"],
        bins=Bins_edges,
        weights=weight_ROIscore[detailed_labels_Roiscore == "diboson"],
    )[0]
    N_obs_ref = np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[
        0
    ]
    

    soft_met = np.linspace(0, 5, nb_points_soft_met_for_fitting)
    var_lenght=len(soft_met)
    signal_obs = [None] * var_lenght
    ztautau_obs = [None] * var_lenght
    ttbar_obs = [None] * var_lenght
    diboson_obs = [None] * var_lenght
    N_obs = [None] * var_lenght

    # We create the modified observed list so we can fit on
    for i in range(var_lenght):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp=dataset
        dataset_tamp = systematics(dataset_tamp, soft_met=soft_met[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        detailed_labels_Roiscore = dataset_tamp["detailed_labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
        signal_obs[i] = ( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "htautau"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "htautau"],
            )[0] 
            -signal_obs_ref
                     )
        ztautau_obs[i] = ( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "ztautau"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "ztautau"],
            )[0] 
            -ztautau_obs_ref
                      )
        ttbar_obs[i] =( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "ttbar"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "ttbar"],
            )[0] 
            -ttbar_obs_ref 
                   )
        diboson_obs[i] =( np.histogram(
            score_ROIscore[detailed_labels_Roiscore == "diboson"],
            bins=Bins_edges,
            weights=weight_ROIscore[detailed_labels_Roiscore == "diboson"],
            )[0] 
            -diboson_obs_ref
                    )
            
        N_obs[i] = (
            np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[0]
            - N_obs_ref
        )
   
    y_obs_name = ["signal (htautau) obs", "ztautau obs","ttbar obs","diboson obs", " total obs"]
    signal_obs_order = [[signal_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    ztautau_obs_order = [[ztautau_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    ttbar_obs_order = [[ttbar_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    diboson_obs_order = [[diboson_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    N_obs_order = [[N_obs[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
    y_obs = [signal_obs_order, ztautau_obs_order,ttbar_obs_order,diboson_obs_order, N_obs_order]
    # y_obs[0:2] = sig, bkg, tot
    # y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    # y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un soft_met précis

    fit_param = [[None] * len(y_obs) for i in range(nb_bins)]
    fit_cov = [[None] * len(y_obs) for i in range(nb_bins)]
    for i in range(len(y_obs)):  # =5
        for j in range(nb_bins):
            # Fit
            from iminuit import Minuit

            
            x_data = np.array(soft_met)
            y_data = y_obs[i][j]
            
            def chi2(a, b):
                y_fit = Polynomial_Reg_Model_forced_soft_met(x_data, a, b)
                return np.sum((y_data - y_fit)**2)

            # Minimize
            m = Minuit(chi2, a=0, b=0)
            m.errordef = Minuit.LEAST_SQUARES
            m.migrad()
            m.hesse() 

            fit_param[j][i] = [m.values["a"],m.values["b"]]
            fit_cov[j][i] =  m.covariance


            soft_met_fit = np.linspace(min(soft_met), max(soft_met), 60)
            y_fit = Polynomial_Reg_Model_forced_soft_met(soft_met_fit, *(fit_param[j][i]))

            # Automatic error propagation at each x
            from iminuit.util import propagate

            # y_std = np.array([
            #     propagate(lambda a, b: model(xi, a, b), m.values, fit_cov[j][i])
            #     for xi in soft_met_fit])   

            results = [propagate(lambda p: Polynomial_Reg_Model_forced_soft_met(xi, p[0], p[1]), m.values, fit_cov[j][i]) for xi in soft_met_fit]

            y = np.array([r[0] for r in results])
            ycov = np.array([r[1] for r in results])  # shape (len(soft_met_fit), 1, 1)

            y_std = ycov**0.5
            
            # Plot

            if Plotteur == "True" and (j in [2 ,nb_bins-2]):

                plt.scatter(
                    soft_met, y_obs[i][j], label="%s" % (y_obs_name[i]), color="dodgerblue"
                )
                plt.plot(soft_met_fit, y_fit, color="darkorange", label="Fit")
                plt.fill_between(
                    soft_met_fit,
                    y - y_std,
                    y + y_std,
                    color="orange",
                    alpha=0.3,
                    label="Fitting uncertainty",
                )
                plt.ylabel("data observed")
                plt.xlabel("Soft MET factor")
                plt.title("Data observed with a variation of Soft MET (bin n°%s)" % (j + 1))
                plt.legend()
                plt.grid()
                plt.show()

    # Save fiting param
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/Fitting_Parameters/bkg_subchannel"%(current_dir)):
        os.makedirs("%s/Fitting_Parameters/bkg_subchannel"%(current_dir))
    np.savez(
        "%s/Fitting_Parameters/bkg_subchannel/SOFTMET_3bkg_%sbins_2orderFittingParam.npz" % (current_dir,nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])


#################
#Fitter
###########################################"




def tes_fitter(
    score,  # model previously
    train_set,
    systematics,
):
    """
    Task 1 : Analysis TES Uncertainty
    1. Loop over different values of tes and make store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of tes and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given TES

    """
    syst_set = systematics(train_set, tes=1)
    # score = model.predict(syst_set["data"])

    histogram = np.histogram(score, bins=100, range=(0, 1))

    # Write a function to loop over different values of tes and histogram and make fit function which transforms the histogram for any given TES

    def fit_function(array, tes):
        # Dummy fit function, replace with actual fitting procedure
        return [array[i] * f[i](tes) for i in range(len(array))]

    return fit_function


def jes_fitter(
    score,
    train_set,
    systematics,
):
    """
    Task 1 : Analysis JES Uncertainty
    1. Loop over different values of jes and store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of JES and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given jes

    """
    syst_set = systematics(train_set, jes=1)
    # score = model.predict(syst_set["data"])

    histogram = np.histogram(score, bins=100, range=(0, 1))

    # Write a function to loop over different values of jes and histogram and make fit function which transforms the histogram for any given JES

    def fit_function(array, jes):
        # Dummy fit function, replace with actual fitting procedure
        return array * jes

    return fit_function









"""
def regression_jes(dataset, model, nb_bins=25, threshold=0):
    from utils import histogram_dataset
    from statistical_analysis import calculate_saved_info
    import matplotlib.pyplot as plt

    dataset_tamp = systematics(dataset, jes=1)
    if "score" in dataset_tamp["data"].columns:
        dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
    data_score = model.predict(dataset_tamp["data"])

    weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
    label_Roiscore = dataset_tamp["labels"][data_score > threshold]
    score_ROIscore = data_score[data_score > threshold]
    Bins_edges = np.linspace(0, 1, nb_bins + 1)  # A changer au besoin
    signal_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 1],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 1],
    )[0]
    bkg_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 0],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 0],
    )[0]
    N_obs_ref = np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[
        0
    ]

    sigma = np.linspace(-0.106, 0.096, 100)
    jes = [np.exp(sigma[i]) for i in range(len(sigma))]
    signal_obs = [None] * len(sigma)
    bkg_obs = [None] * len(sigma)
    N_obs = [None] * len(sigma)

    # data_score=[model.predict(systematics(dataset,tes=tes[i]))  for i in range(len(sigma)) ]  #Alternative to the for loop
    # weight_ROIscore=[dataset["weights"][data_score[i] > threshold] for i in range(len(sigma))]
    # label_Roiscore=[dataset["labels"][data_score[i]>threshold] for i in range(len(sigma))]
    # score_ROIscore=[data_score[data_score[i]>threshold]  for i in range(len(sigma))]
    # Bins_edges=[np.linspace(np.min(score_ROIscore[i]),np.max(score_ROIscore[i]),nb_bins+1) for i in range(len(sigma))]
    # signal_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==1], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==1])[0] for i in range(len(sigma))]
    # bkg_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==0], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==0])[0] for i in range(len(sigma))]
    # N_obs=[np.histogram(score_ROIscore[i], bins=Bins_edges[i], weights=weight_ROIscore[i])[0] for i in range(len(sigma))]

    # We create the modified observed list so we can fit on
    for i in range(len(sigma)):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp = systematics(dataset, jes=jes[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        label_Roiscore = dataset_tamp["labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
        signal_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 1],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 1],
            )[0]
            - signal_obs_ref
        )  # 3 lignes below new
        bkg_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 0],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 0],
            )[0]
            - bkg_obs_ref
        )
        N_obs[i] = (
            np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[0]
            - N_obs_ref
        )

    y_obs_name = ["signal obs", "bkg obs", " total obs"]
    signal_obs_order = [
        [signal_obs[j][i] for j in range(len(sigma))] for i in range(nb_bins)
    ]
    bkg_obs_order = [[bkg_obs[j][i] for j in range(len(sigma))] for i in range(nb_bins)]
    N_obs_order = [[N_obs[j][i] for j in range(len(sigma))] for i in range(nb_bins)]
    y_obs = [signal_obs_order, bkg_obs_order, N_obs_order]
    # y_obs[0:2] = sig, bkg, tot
    # y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    # y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un TES précis

    fit_param = [[None] * 3 for i in range(nb_bins)]
    fit_cov = [[None] * 3 for i in range(nb_bins)]
    for i in range(len(y_obs)):  # =3
        for j in range(nb_bins):
            # Fit
            fit_param_tamp, fit_cov_tamp = curve_fit(
                Polynomial_Reg_Model_forced_jes_tes, jes, y_obs[i][j]
            )
            fit_param[j][i] = fit_param_tamp
            fit_cov[j][i] = fit_cov_tamp

            # popt = best-fit parameters [a0, a1, a2]
            # pcov = covariance matrix of parameters

            # Calculate uncertainty (standard deviation) of parameters
            param_err = np.sqrt(np.diag(fit_cov[j][i]))

            print("Fit parameters:", fit_param[j][i])
            print("Parameter uncertainties:", param_err)

            # Predict values
            jes_fit = np.linspace(min(jes), max(jes), 200)
            y_fit = Polynomial_Reg_Model_forced_jes_tes(jes_fit, *(fit_param[j][i]))

            # To get uncertainty on the fit curve, propagate errors:
            # Compute Jacobian matrix at each x_fit
            J = np.vstack(
                [jes_fit**k for k in range(len(fit_param[j][i]))]
            ).T  # shape (num_points, 4)
            print("J:", J)
            y_var = np.sum(J @ fit_cov[j][i] * J, axis=1)
            y_std = np.sqrt(y_var)
            print("Y _std:", y_std)

            # Plot
            if Plotteur == "True" and (j in [2, int(nb_bins/2),nb_bins-2]):
                print(np.size(jes), " jes size")
                print("y_obs[i][j] size")
                plt.scatter(
                    jes, y_obs[i][j], label="%s" % (y_obs_name[i]), color="dodgerblue"
                )
                plt.plot(jes_fit, y_fit, color="darkorange", label="Fit")
                plt.fill_between(
                    jes_fit,
                    y_fit - y_std,
                    y_fit + y_std,
                    color="orange",
                    alpha=0.3,
                    label="Fitting uncertainty",
                )
                plt.ylabel("data observed")
                plt.xlabel("jes factor")
                plt.title("Data observed with a variation of Jes (bin n°%s)" % (j + 1))
                plt.legend()
                plt.grid()
                plt.show()

    # Save fiting param
    np.savez(
        "JES_%sbins_2orderFittingParam.npz" % (nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])


def regression_soft_met(dataset, model, nb_bins=25, threshold=0):
    from utils import histogram_dataset
    from statistical_analysis import calculate_saved_info
    import matplotlib.pyplot as plt

    dataset_tamp = systematics(dataset, soft_met=0)
    if "score" in dataset_tamp["data"].columns:
        dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
    data_score = model.predict(dataset_tamp["data"])

    weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
    label_Roiscore = dataset_tamp["labels"][data_score > threshold]
    score_ROIscore = data_score[data_score > threshold]
    Bins_edges = np.linspace(0, 1, nb_bins + 1)  # A changer au besoin
    signal_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 1],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 1],
    )[0]
    bkg_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 0],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 0],
    )[0]
    N_obs_ref = np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[
        0
    ]

    soft_met = np.linspace(0, 5, 250)
    signal_obs = [None] * len(soft_met)
    bkg_obs = [None] * len(soft_met)
    N_obs = [None] * len(soft_met)

    # data_score=[model.predict(systematics(dataset,tes=tes[i]))  for i in range(len(sigma)) ]  #Alternative to the for loop
    # weight_ROIscore=[dataset["weights"][data_score[i] > threshold] for i in range(len(sigma))]
    # label_Roiscore=[dataset["labels"][data_score[i]>threshold] for i in range(len(sigma))]
    # score_ROIscore=[data_score[data_score[i]>threshold]  for i in range(len(sigma))]
    # Bins_edges=[np.linspace(np.min(score_ROIscore[i]),np.max(score_ROIscore[i]),nb_bins+1) for i in range(len(sigma))]
    # signal_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==1], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==1])[0] for i in range(len(sigma))]
    # bkg_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==0], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==0])[0] for i in range(len(sigma))]
    # N_obs=[np.histogram(score_ROIscore[i], bins=Bins_edges[i], weights=weight_ROIscore[i])[0] for i in range(len(sigma))]

    # We create the modified observed list so we can fit on
    for i in range(len(soft_met)):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp = systematics(dataset, soft_met=soft_met[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        label_Roiscore = dataset_tamp["labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
        signal_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 1],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 1],
            )[0]
            - signal_obs_ref
        )  # 3 lignes below new
        bkg_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 0],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 0],
            )[0]
            - bkg_obs_ref
        )
        N_obs[i] = (
            np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[0]
            - N_obs_ref
        )

    y_obs_name = ["signal obs", "bkg obs", " total obs"]
    signal_obs_order = [
        [signal_obs[j][i] for j in range(len(soft_met))] for i in range(nb_bins)
    ]
    bkg_obs_order = [
        [bkg_obs[j][i] for j in range(len(soft_met))] for i in range(nb_bins)
    ]
    N_obs_order = [[N_obs[j][i] for j in range(len(soft_met))] for i in range(nb_bins)]
    y_obs = [signal_obs_order, bkg_obs_order, N_obs_order]
    # y_obs[0:2] = sig, bkg, tot
    # y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    # y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un TES précis

    fit_param = [[None] * 3 for i in range(nb_bins)]
    fit_cov = [[None] * 3 for i in range(nb_bins)]
    for i in range(len(y_obs)):  # =3
        for j in range(nb_bins):
            # Fit
            fit_param_tamp, fit_cov_tamp = curve_fit(
                Polynomial_Reg_Model_forced_soft_met, soft_met, y_obs[i][j]
            )
            fit_param[j][i] = fit_param_tamp
            fit_cov[j][i] = fit_cov_tamp

            # popt = best-fit parameters [a0, a1, a2]
            # pcov = covariance matrix of parameters

            # Calculate uncertainty (standard deviation) of parameters
            param_err = np.sqrt(np.diag(fit_cov[j][i]))

            print("Fit parameters:", fit_param[j][i])
            print("Parameter uncertainties:", param_err)

            # Predict values
            soft_met_fit = np.linspace(min(soft_met), max(soft_met), 200)
            y_fit = Polynomial_Reg_Model_forced_soft_met(
                soft_met_fit, *(fit_param[j][i])
            )

            # To get uncertainty on the fit curve, propagate errors:
            # Compute Jacobian matrix at each x_fit
            J = np.vstack(
                [soft_met_fit**k for k in range(len(fit_param[j][i]))]
            ).T  # shape (num_points, 4)
            print("J:", J)
            y_var = np.sum(J @ fit_cov[j][i] * J, axis=1)
            y_std = np.sqrt(y_var)
            print("Y _std:", y_std)

            # Plot
            if Plotteur == "True" and (j in [2, int(nb_bins/2),nb_bins-2]):
                print(np.size(soft_met), " soft met size")
                print("y_obs[i][j] size")
                plt.scatter(
                    soft_met,
                    y_obs[i][j],
                    label="%s" % (y_obs_name[i]),
                    color="dodgerblue",
                )
                plt.plot(soft_met_fit, y_fit, color="darkorange", label="Fit")
                plt.fill_between(
                    soft_met_fit,
                    y_fit - y_std,
                    y_fit + y_std,
                    color="orange",
                    alpha=0.3,
                    label="Fitting uncertainty",
                )
                plt.ylabel("data observed")
                plt.xlabel("soft met factor")
                plt.title(
                    "Data observed with a variation of Soft Met (bin n°%s)" % (j + 1)
                )
                plt.legend()
                plt.grid()
                plt.show()

    # Save fiting param
    np.savez(
        "SOFTMET_%sbins_2orderFittingParam.npz" % (nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])
















def regression_tes_old(dataset, model, nb_bins=25, threshold=0):
    from utils import histogram_dataset
    from statistical_analysis import calculate_saved_info
    import matplotlib.pyplot as plt

    dataset_tamp=dataset
    dataset_tamp = systematics(dataset_tamp, tes=1)
    if "score" in dataset_tamp["data"].columns:
        dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
    data_score = model.predict(dataset_tamp["data"])

    weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
    label_Roiscore = dataset_tamp["labels"][data_score > threshold]
    score_ROIscore = data_score[data_score > threshold]
    Bins_edges = np.linspace(0, 1, nb_bins + 1)  # A changer au besoin
    signal_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 1],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 1],
    )[0]
    bkg_obs_ref = np.histogram(
        score_ROIscore[label_Roiscore == 0],
        bins=Bins_edges,
        weights=weight_ROIscore[label_Roiscore == 0],
    )[0]
    N_obs_ref = np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[
        0
    ]
    

    sigma = np.linspace(-0.106, 0.096, 100)
    tes = [np.exp(sigma[i]) for i in range(len(sigma))]
    signal_obs = [None] * len(sigma)
    bkg_obs = [None] * len(sigma)
    N_obs = [None] * len(sigma)

    # data_score=[model.predict(systematics(dataset,tes=tes[i]))  for i in range(len(sigma)) ]  #Alternative to the for loop
    # weight_ROIscore=[dataset["weights"][data_score[i] > threshold] for i in range(len(sigma))]
    # label_Roiscore=[dataset["labels"][data_score[i]>threshold] for i in range(len(sigma))]
    # score_ROIscore=[data_score[data_score[i]>threshold]  for i in range(len(sigma))]
    # Bins_edges=[np.linspace(np.min(score_ROIscore[i]),np.max(score_ROIscore[i]),nb_bins+1) for i in range(len(sigma))]
    # signal_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==1], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==1])[0] for i in range(len(sigma))]
    # bkg_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==0], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==0])[0] for i in range(len(sigma))]
    # N_obs=[np.histogram(score_ROIscore[i], bins=Bins_edges[i], weights=weight_ROIscore[i])[0] for i in range(len(sigma))]

    # We create the modified observed list so we can fit on
    for i in range(len(sigma)):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp = systematics(dataset, tes=tes[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        label_Roiscore = dataset_tamp["labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
        signal_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 1],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 1],
            )[0]
            - signal_obs_ref
        )  # 3 lignes below new
        bkg_obs[i] = (
            np.histogram(
                score_ROIscore[label_Roiscore == 0],
                bins=Bins_edges,
                weights=weight_ROIscore[label_Roiscore == 0],
            )[0]
            - bkg_obs_ref
        )
        N_obs[i] = (
            np.histogram(score_ROIscore, bins=Bins_edges, weights=weight_ROIscore)[0]
            - N_obs_ref
        )

    y_obs_name = ["signal obs", "bkg obs", " total obs"]
    signal_obs_order = [
        [signal_obs[j][i] for j in range(len(sigma))] for i in range(nb_bins)
    ]
    bkg_obs_order = [[bkg_obs[j][i] for j in range(len(sigma))] for i in range(nb_bins)]
    N_obs_order = [[N_obs[j][i] for j in range(len(sigma))] for i in range(nb_bins)]
    y_obs = [signal_obs_order, bkg_obs_order, N_obs_order]
    # y_obs[0:2] = sig, bkg, tot
    # y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    # y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un TES précis

    fit_param = [[None] * len(y_obs) for i in range(nb_bins)]
    fit_cov = [[None] * len(y_obs) for i in range(nb_bins)]
    for i in range(len(y_obs)):  # =3
        for j in range(nb_bins):
            # Fit
            from iminuit import Minuit

            # Data
            x_data = np.array(tes)
            y_data = y_obs[i][j]
            x0, y0 = 1, 0  

            # Constrained model
            def model(x, a, b):
                c = y0 - a * np.power(x0,2) - b * x0
                return a * np.power(x,2) + b * x + c

            # Chi2 function
            def chi2(a, b):
                y_model = model(x_data, a, b)
                return np.sum((y_data - y_model)**2)

            # Minimize
            m = Minuit(chi2, a=0, b=0)
            m.errordef = Minuit.LEAST_SQUARES
            m.migrad()
            m.hesse() 

            fit_param[j][i] = [m.values["a"],m.values["b"]]
            fit_cov[j][i] =  m.covariance


            tes_fit = np.linspace(min(tes), max(tes), 200)
            y_fit = model(tes_fit, *(fit_param[j][i]))

            # Automatic error propagation at each x
            from iminuit.util import propagate

            # y_std = np.array([
            #     propagate(lambda a, b: model(xi, a, b), m.values, fit_cov[j][i])
            #     for xi in tes_fit])   

            results = [propagate(lambda p: model(xi, p[0], p[1]), m.values, fit_cov[j][i]) for xi in tes_fit]

            y = np.array([r[0] for r in results])
            ycov = np.array([r[1] for r in results])  # shape (len(tes_fit), 1, 1)

            y_std = ycov**0.5
            


                            # fit_param_tamp, fit_cov_tamp = curve_fit(
                            #     Polynomial_Reg_Model_forced_jes_tes, tes, y_obs[i][j]
                            # )
                            # fit_param[j][i] = fit_param_tamp
                            # fit_cov[j][i] = fit_cov_tamp

                            # # popt = best-fit parameters [a0, a1, a2]
                            # # pcov = covariance matrix of parameters

                            # # Calculate uncertainty (standard deviation) of parameters
                            # param_err = np.sqrt(np.diag(fit_cov[j][i]))

                            # print("Fit parameters:", fit_param[j][i])
                            # print("Parameter uncertainties:", param_err)

                            # # Predict values
                            # tes_fit = np.linspace(min(tes), max(tes), 200)
                            # y_fit = Polynomial_Reg_Model_forced_jes_tes(tes_fit, *(fit_param[j][i]))

                            # # To get uncertainty on the fit curve, propagate errors:
                            # # Compute Jacobian matrix at each x_fit
                            # J = np.vstack(
                            #     [tes_fit**k for k in range(len(fit_param[j][i]))]
                            # ).T  # shape (num_points, 4)
                            # print("J:", J)

                            # # y_var = np.sum(J @ fit_cov[j][i] * J, axis=1)
                            # y_var = np.einsum("ij,jk,ik->i", J, fit_cov[j][i], J)
                            
                            # y_std = np.sqrt(y_var)
            print("Y _std:", y_std)

            # Plot

            if Plotteur == "True" and (j in [2, int(nb_bins/2),nb_bins-2]):
                print(np.size(tes), " tes size")
                print("y_obs[i][j] size")
                plt.scatter(
                    tes, y_obs[i][j], label="%s" % (y_obs_name[i]), color="dodgerblue"
                )
                plt.plot(tes_fit, y_fit, color="darkorange", label="Fit")
                plt.fill_between(
                    tes_fit,
                    y - y_std,
                    y + y_std,
                    color="orange",
                    alpha=0.3,
                    label="Fitting uncertainty",
                )
                plt.ylabel("data observed")
                plt.xlabel("Tes factor")
                plt.title("Data observed with a variation of Tes (bin n°%s)" % (j + 1))
                plt.legend()
                plt.grid()
                plt.show()

    # Save fiting param
    np.savez(
        "TES_%sbins_2orderFittingParam.npz" % (nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])



#


def regression_SoftMET_old(dataset, model, nb_bins=25, threshold=0.93):
    from utils import histogram_dataset
    from statistical_analysis import calculate_saved_info
    import matplotlib.pyplot as plt

    SoftMET = np.linspace(0, 5, 100)
    signal_obs = [None] * len(SoftMET)
    bkg_obs = [None] * len(SoftMET)
    N_obs = [None] * len(SoftMET)

    # data_score=[model.predict(systematics(dataset,tes=tes[i]))  for i in range(len(sigma)) ]  #Alternative to the for loop
    # weight_ROIscore=[dataset["weights"][data_score[i] > threshold] for i in range(len(sigma))]
    # label_Roiscore=[dataset["labels"][data_score[i]>threshold] for i in range(len(sigma))]
    # score_ROIscore=[data_score[data_score[i]>threshold]  for i in range(len(sigma))]
    # Bins_edges=[np.linspace(np.min(score_ROIscore[i]),np.max(score_ROIscore[i]),nb_bins+1) for i in range(len(sigma))]
    # signal_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==1], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==1])[0] for i in range(len(sigma))]
    # bkg_obs=[np.histogram(score_ROIscore[i][label_Roiscore[i]==0], bins=Bins_edges[i], weights=weight_ROIscore[i][label_Roiscore[i]==0])[0] for i in range(len(sigma))]
    # N_obs=[np.histogram(score_ROIscore[i], bins=Bins_edges[i], weights=weight_ROIscore[i])[0] for i in range(len(sigma))]

    # We create the modified observed list so we can fit on
    for i in range(len(SoftMET)):
        ##We need the for loop due to the addition of the score cell in the data_score
        dataset_tamp = systematics(dataset, soft_met=SoftMET[i])
        if "score" in dataset_tamp["data"].columns:
            dataset_tamp["data"] = dataset_tamp["data"].drop(columns=["score"])
        data_score = model.predict(dataset_tamp["data"])

        weight_ROIscore = dataset_tamp["weights"][data_score > threshold]
        label_Roiscore = dataset_tamp["labels"][data_score > threshold]
        score_ROIscore = data_score[data_score > threshold]
        Bins_edges = np.linspace(
            np.min(score_ROIscore), np.max(score_ROIscore), nb_bins + 1
        )
        signal_obs[i] = np.histogram(
            score_ROIscore[label_Roiscore == 1],
            bins=Bins_edges,
            weights=weight_ROIscore[label_Roiscore == 1],
        )[
            0
        ]  # 3 lignes below new
        bkg_obs[i] = np.histogram(
            score_ROIscore[label_Roiscore == 0],
            bins=Bins_edges,
            weights=weight_ROIscore[label_Roiscore == 0],
        )[0]
        N_obs[i] = np.histogram(
            score_ROIscore, bins=Bins_edges, weights=weight_ROIscore
        )[0]

    from scipy.optimize import curve_fit

    def Polynomial_Reg_Model(tes, c, a1, a2):
        return c + a1 * tes + a2 * tes * tes

    y_obs_name = ["signal obs", "bkg obs", " total obs"]
    signal_obs_order = [
        [signal_obs[j][i] for j in range(len(SoftMET))] for i in range(nb_bins)
    ]
    bkg_obs_order = [
        [bkg_obs[j][i] for j in range(len(SoftMET))] for i in range(nb_bins)
    ]
    N_obs_order = [[N_obs[j][i] for j in range(len(SoftMET))] for i in range(nb_bins)]
    y_obs = [signal_obs_order, bkg_obs_order, N_obs_order]
    # y_obs[0:2] = sig, bkg, tot
    # y_obs[i][0:nb_bins]= y_obs pour le numéro de bin donné
    # y_obs[i][j][0:len(sigma)]=y_obs pour le numéro de bin donné, la valeur de y_obs pour un TES précis

    fit_param = [[None] * 3 for i in range(nb_bins)]
    fit_cov = [[None] * 3 for i in range(nb_bins)]
    for i in range(len(y_obs)):  # =3
        for j in range(nb_bins):
            # Fit
            fit_param_tamp, fit_cov_tamp = curve_fit(
                Polynomial_Reg_Model, SoftMET, y_obs[i][j]
            )
            fit_param[j][i] = fit_param_tamp
            fit_cov[j][i] = fit_cov_tamp

            # popt = best-fit parameters [a0, a1, a2]
            # pcov = covariance matrix of parameters

            # Calculate uncertainty (standard deviation) of parameters
            param_err = np.sqrt(np.diag(fit_cov[j][i]))

            print("Fit parameters:", fit_param[j][i])
            print("Parameter uncertainties:", param_err)

            # Predict values
            SoftMET_fit = np.linspace(min(SoftMET), max(SoftMET), 200)
            y_fit = Polynomial_Reg_Model(SoftMET_fit, *(fit_param[j][i]))

            # To get uncertainty on the fit curve, propagate errors:
            # Compute Jacobian matrix at each x_fit
            J = np.vstack(
                [SoftMET_fit**k for k in range(len(fit_param[j][i]))]
            ).T  # shape (num_points, 4)
            y_var = np.sum(J @ fit_cov[j][i] * J, axis=1)
            y_std = np.sqrt(y_var)

            # Plot
            if Plotteur == "True" or ((j % 5) == 0):
                plt.scatter(
                    SoftMET,
                    y_obs[i][j],
                    label="%s" % (y_obs_name[i]),
                    color="dodgerblue",
                )
                plt.plot(SoftMET_fit, y_fit, color="darkorange", label="Fit")
                plt.fill_between(
                    SoftMET_fit,
                    y_fit - y_std,
                    y_fit + y_std,
                    color="orange",
                    alpha=0.3,
                    label="Fitting uncertainty",
                )
                plt.ylabel("data observed")
                plt.xlabel("Soft MET factor")
                plt.title(
                    "Data observed with a variation of Soft MET (bin n°%s)" % (j + 1)
                )
                plt.legend()
                plt.grid()
                plt.show()

    # Save fiting param
    np.savez(
        "SoftMET_%sbins_2orderFittingParam.npz" % (nb_bins),
        fit_param=np.array(fit_param),
        fit_cov=np.array(fit_cov),
    )
    # # Load
    # data = np.load('4orderFittingParam.npz')
    # print(data['fit_param'])
    # print(data['fit_cov'])

    ############################################################"
    # ################################
    # #############"
    ##Really cool regressor model for free fitting

    # from sklearn.gaussian_process import GaussianProcessRegressor
    # from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RationalQuadratic   #RBF
    # # Data
    # X = tes.reshape(-1, 1)
    # y_sig = signal_obs
    # # Kernel: RBF + noise
    # kernel = ConstantKernel() * RationalQuadratic() + WhiteKernel()
    # # Fit model
    # gp = GaussianProcessRegressor(kernel=kernel,normalize_y=True, n_restarts_optimizer=1)
    # gp.fit(X, y_sig)
    # # Predict
    # x_pred=np.linspace(np.min(tes), np.max(tes), 75).reshape(-1, 1)
    # y_pred, sigma = gp.predict(x_pred, return_std=True)
    # # Plot
    # plt.scatter(X.ravel(), y_sig, label="Sig obs",color="red")
    # plt.plot(x_pred.ravel(), y_pred, label="Sig fit",color="black")
    # plt.fill_between(x_pred.ravel(), y_pred - 1*sigma, y_pred + 1*sigma,color='gray', alpha=0.2, label="1sigma")
    # plt.legend()
    # plt.title("Gaussian Process Regression test")
    # plt.show()
    # import joblib
    # # Save to a file
    # joblib.dump(gp, 'Tes_Sig_gp_fit_model.joblib')

    # #To import
    # gp_loaded = joblib.load('gp_model.joblib')
    # y_pred, sigma = gp_loaded.predict(X_pred, return_std=True)
"""

