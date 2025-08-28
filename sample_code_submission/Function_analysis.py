from parameter_management_scan import Parameter_Distribution
import os

Tamp_parameter = Parameter_Distribution.get_all()
THV_size = Tamp_parameter["THV_size"]
ModelType = Tamp_parameter["ModelType"]
NbTrain = THV_size[0]
NbHoldout = THV_size[1]
NbValidation = THV_size[2]


def Parabola_Likelihood_plot(
    saved_info_hold,
    score_test,
    weight_test,
    Methode_Mu_Compar,
    nb_bins,
    threshold=0,
    mu_init=1.0,
):
    # Plot parabola and sigma
    from statistical_analysis import compute_mu
    import matplotlib.pyplot as plt
    import numpy as np

    if len(Methode_Mu_Compar) > 2:
        cmap = plt.get_cmap("gist_rainbow")
        values = np.linspace(0, 1, len(Methode_Mu_Compar))
        colors = [cmap(v) for v in values]
        linewidth = [7.5, 6, 4.5, 3, 1.5]
    else:
        colors = ["dodgerblue", "orange"]
        linewidth = [7.5, 6]

    plt.figure(1, layout="constrained", figsize=(10, 10))
    for i in range(len(Methode_Mu_Compar)):
        Compute_mu_tamp = compute_mu(
            saved_info_hold=saved_info_hold,
            score_test=score_test,
            weight_test=weight_test,
            method=Methode_Mu_Compar[i],
            nb_bins=nb_bins,
            threshold=threshold,
            mu_init=mu_init,
        )

        log_likelihood_ratio = (
            Compute_mu_tamp["negloglike_values"] - Compute_mu_tamp["negloglike_mu_hat"]
        )
        if (Methode_Mu_Compar[i] == "BNLL") or (Methode_Mu_Compar[i] == "BNLL_syst"):
            plt.plot(
                Compute_mu_tamp["mu_axis_values"],
                log_likelihood_ratio,
                linewidth=linewidth[i],
                color=colors[i],
                label=r"%s %s bins : $\hat{\mu} = %s \pm %s$"
                % (
                    Methode_Mu_Compar[i],
                    nb_bins,
                    np.round(Compute_mu_tamp["mu_hat"], 3),
                    np.round(Compute_mu_tamp["del_mu_stat_sup"], 3),
                ),
            )

        else:
            plt.plot(
                Compute_mu_tamp["mu_axis_values"],
                log_likelihood_ratio,
                linewidth=linewidth[i],
                color=colors[i],
                label=r"%s : $\hat{\mu} = %s \pm %s$"
                % (
                    Methode_Mu_Compar[i],
                    np.round(Compute_mu_tamp["mu_hat"], 3),
                    np.round(Compute_mu_tamp["del_mu_stat_sup"], 3),
                ),
                linestyle="dotted",
            )
        # plt.hlines(0.5, min(mu_axis_values), max(mu_axis_values), linestyle= '-', color= 'black',label=("1sigma =",( np.round(mu_axis_values[np.where(log_likelihood_ratio==np.min(np.abs(log_likelihood_ratio-0.5))+0.5)[0]][0]-mu),3) ) )
        # plt.hlines(2, min(mu_axis_values), max(mu_axis_values), linestyle= '-', color= 'grey',label=("2sigma = ",( np.round(mu_axis_values[np.where(log_likelihood_ratio==np.min(np.abs(log_likelihood_ratio-2))+2)[0]][0]-mu),3) ) )

        # intersection=np.abs(log_likelihood_ratio-0.5)
        # mu_delta_sup=np.argmin(intersection[np.argmin(log_likelihood_ratio):])
        # mu_delta_inf=np.argmin(intersection[:np.argmin(log_likelihood_ratio)])
        # ax1.hlines(0.5, min(mu_axis_values), max(mu_axis_values), linestyle= '-', color= 'black',label="1sigma= inf:%s |sup:%s"%(np.round(-Compute_mu_info["mu"]+mu_axis_values[mu_delta_inf],3),np.round(-Compute_mu_info["mu"]+mu_axis_values[np.argmin(log_likelihood_ratio)+mu_delta_sup],3)))
        # intersection=np.abs(log_likelihood_ratio-2)
        # mu_delta_sup=np.argmin(intersection[np.argmin(log_likelihood_ratio):])
        # mu_delta_inf=np.argmin(intersection[:np.argmin(log_likelihood_ratio)])
        # ax1.hlines(2, min(mu_axis_values), max(mu_axis_values), linestyle= '-', color= 'grey',label="2sigma= inf:%s |sup:%s"%(np.round(-Compute_mu_info["mu"]+mu_axis_values[mu_delta_inf],3),np.round(-Compute_mu_info["mu"]+mu_axis_values[np.argmin(log_likelihood_ratio)+mu_delta_sup],3)))
    plt.axhline(0.5, linestyle="-", color="red", linewidth=1, label=(r"$1\sigma$"))
    plt.axhline(2, linestyle="-", color="black", linewidth=1, label=(r"$2\sigma$"))
    plt.ylabel("log-likelihood ratio", fontsize=18)
    plt.xlabel(r"$\mu$", fontsize=18)
    plt.legend(loc="best", fontsize=18)
    plt.tick_params(axis="both", labelsize=14)
    plt.title(
        "Parabola curve for different method of determination of mu\n(Unormalized convention used for the NLL)",
        fontsize=25,
    )
    plt.grid(True)
    # plt.xlim([0.7,1.3])
    # plt.ylim([0,4])
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/images/Compar/Parabola_Compar" % (current_dir)):
        os.makedirs("%s/images/Compar/Parabola_Compar" % (current_dir))
    plt.savefig(
        "%s/images/Compar/Parabola_Compar/%s_Compar_ParabolaVsMethod_Train=%s_Holdout=%s_Valid=%s.png"
        % (current_dir, ModelType, NbTrain, NbHoldout, NbValidation)
    )
    plt.show()
    plt.close()
    ##End plot parabola and sigma


def Bins_BNLL_varia(
    saved_info_hold,
    score_test,
    weight_test,
    bin_min,
    bin_max,
    value_bin_step,
    mu_init=1.0,
    threshold=0,
):
    from statistical_analysis import compute_mu
    import numpy as np
    import matplotlib.pyplot as plt

    bin_list = np.arange(bin_min, bin_max, value_bin_step)
    mu_list = np.zeros(len(bin_list))
    del_mu_list = np.zeros(len(bin_list))
    for i in range(len(bin_list)):
        print("nbr of bins = ", bin_list[i])
        compute_mu_tamp = compute_mu(
            method="BNLL",
            nb_bins=bin_list[i],
            score_test=score_test,
            weight_test=weight_test,
            saved_info_hold=saved_info_hold,
            mu_init=mu_init,
            threshold=threshold,
        )

        mu_list[i] = compute_mu_tamp["mu_hat"]
        del_mu_list[i] = compute_mu_tamp["del_mu_stat"]

    fig, ax1 = plt.subplots()
    ax1bis = ax1.twinx()
    ax1.axvline(
        bin_list[np.argmin(del_mu_list)],
        color="blue",
        label="Min uncertainty=%s for %s bins"
        % (round(min(del_mu_list), 6), round(bin_list[np.argmin(del_mu_list)], 3)),
    )
    ax1.fill_between(
        bin_list,
        mu_list - del_mu_list,
        mu_list + del_mu_list,
        color="lightskyblue",
        alpha=0.5,
    )
    ax1.plot(bin_list, mu_list, marker=None, color="dodgerblue")
    ax1.set_ylabel(f"$\mu$", color="dodgerblue")  # Fix that
    # ax1.set_ylim([max(0,min(mu_list)-del_mu_tot_list[np.argmin(mu_list)]*1.1),min(3,max(mu_list)+del_mu_tot_list[np.argmax(mu_list)]*1.1)])  #In pratice \mu is between 0.1 and 3
    ax1.tick_params(axis="y", labelcolor="dodgerblue")

    ax1bis.plot(bin_list, del_mu_list, marker=None, color="darkorange")
    ax1bis.set_ylabel(f"$\delta\mu$", color="darkorange")
    ax1bis.tick_params(axis="y", labelcolor="darkorange")
    # ax1.set_ylim([min(min(mu_list-del_mu_tot_list)-0.05,min(del_mu_tot_list)-0.05),min(max(mu_list+del_mu_tot_list)+0.05,3)])
    ax1.legend(loc="upper right")
    plt.suptitle(
        "Impact of the number of bins for Binned NLL (with threshold=%s)" % (threshold)
    )
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/images/Compar/Bin_Compar" % (current_dir)):
        os.makedirs("%s/images/Compar/Bin_Compar" % (current_dir))
    plt.savefig(
        "%s/images/Compar/Bin_Compar/%s_Compar_ParabolaVsMethod_thrsld=%s_Train=%s_Holdout=%s_Valid=%s.png"
        % (current_dir, ModelType, threshold, NbTrain, NbHoldout, NbValidation)
    )
    plt.show()
