import numpy as np

"""
Task 1a : Counting Estimator
1.write the saved_info dictionary such that it contains the following keys
    1. beta
    2. gamma
2. Estimate the mu using the formula
    mu = (sum(score * weight) - beta) / gamma
3. return the mu and its uncertainty

Task 1b : Stat-Only Likelihood Estimator
1. Modify the estimation of mu such that it uses the likelihood function
    1. Write a function for the likelihood function which profiles over mu
    2. Use Minuit to minimize the NLL

Task 2 : Systematic Uncertainty
1. substitute the beta and gamma with the tes_fit and jes_fit functions
2. Write a function to likelihood function which profiles over mu, tes and jes
3. Use Minuit to minimize the NLL
4. return the mu and its uncertainty

"""
import math
from iminuit import Minuit

# from iminuit import cost
# from scipy.stats import poisson, norm
import matplotlib.pyplot as plt

from parameter_management_scan import Parameter_Distribution

Tamp_parameter = Parameter_Distribution.get_all()
THV_size = Tamp_parameter["THV_size"]
ModelType = Tamp_parameter["ModelType"]
NbTrain = THV_size[0]
NbHoldout = THV_size[1]
NbValidation = THV_size[2]


def compute_mu(
    saved_info_hold,
    weight_test,
    score_test=0,
    threshold=0,
    mu_init=1.0,
    method="Direct",
    nb_bins=10,
    score_exp_hold=0,
    weight_exp_hold=0,
    label_exp_hold=0,
    detailed_labels_exp_hold=0,
):

    def Model(mu, sig, bkg):
        return mu * sig + bkg

    score_flat = score_test.flatten() > threshold
    score_flat = score_flat.astype(int)

    mu, del_mu_stat, del_mu_tot, del_mu_sys = (0, 0, 0, 0)

    if method == "Direct":  # Based on N=mu*S+b

        def counting_mu(score, weight, saved_info):
            mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
            del_mu_stat = (
                np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
            )
            return mu, del_mu_stat

        mu, del_mu_stat = counting_mu(score_flat, weight_test, saved_info_hold)

        del_mu_stat_inf = -del_mu_stat
        del_mu_stat_sup = del_mu_stat

    elif method == "UNLL":

        def UNLL_mu(threshold, mu_init, saved_info_hold, weight_test, score_test):

            def UNLL_mu_parameter(saved_info_hold, weight_test, score_test, threshold):
                Parabola_Ploteur = False
                sig_exp_holdout_unll = saved_info_hold["gamma"]
                bkg_exp_holdout_unll = saved_info_hold["beta"]
                n_obs_test_unll = np.sum(weight_test[score_test > threshold])
                print(
                    "Compute mu, sig= ",
                    sig_exp_holdout_unll,
                    " bkg= ",
                    bkg_exp_holdout_unll,
                    " N= ",
                    n_obs_test_unll,
                )

                return n_obs_test_unll, sig_exp_holdout_unll, bkg_exp_holdout_unll

            n_obs_test_unll, sig_exp_holdout_unll, bkg_exp_holdout_unll = (
                UNLL_mu_parameter(
                    saved_info_hold=saved_info_hold,
                    weight_test=weight_test,
                    score_test=score_test,
                    threshold=threshold,
                )
            )

            def Cost_unll(mu):
                N_exp_unll = Model(
                    mu=mu, sig=sig_exp_holdout_unll, bkg=bkg_exp_holdout_unll
                )
                # print("sig_exp_holdout: ",sig_exp_holdout," bkg_exp_holdout: ",bkg_exp_holdout," mu: ",mu," N_exp: ",N_exp," n_obs: ",n_obs_test," N_exp mu=1: ",Model(mu=1, sig=sig_exp_holdout, bkg=bkg_exp_holdout)  )
                return -(-N_exp_unll + n_obs_test_unll * np.log(N_exp_unll))

            def UNLL_mu_computation(mu_init):
                m_unll = Minuit(Cost_unll, mu=mu_init)
                m_unll.limits["mu"] = (0, 4)
                m_unll.errordef = Minuit.LIKELIHOOD

                m_unll.migrad()
                m_unll.hesse()
                m_unll.draw_mnprofile("mu")
                mu = m_unll.values["mu"]
                del_mu_stat = m_unll.errors["mu"]
                del_mu_stat_inf = -del_mu_stat
                del_mu_stat_sup = del_mu_stat
                import os

                current_dir = os.path.dirname(os.path.abspath(__file__))
                if not os.path.exists("%s/images/Parabola/UNLL" % (current_dir)):
                    os.makedirs("%s/images/Parabola/UNLL" % (current_dir))
                plt.savefig(
                    "%s/images/Parabola/UNLL/%s_mu=%s_Compar_ParabolaVsMethod_thrsld=%s_Train=%s_Holdout=%s_Valid=%s.png"
                    % (
                        current_dir,
                        ModelType,
                        round(mu, 4),
                        threshold,
                        NbTrain,
                        NbHoldout,
                        NbValidation,
                    )
                )
                plt.show()
                plt.close()

                return mu, del_mu_stat, del_mu_stat_inf, del_mu_stat_sup

            return UNLL_mu_computation(mu_init=mu_init)

        mu, del_mu_stat, del_mu_stat_inf, del_mu_stat_sup = UNLL_mu(
            threshold=threshold,
            mu_init=mu_init,
            saved_info_hold=saved_info_hold,
            weight_test=weight_test,
            score_test=score_test,
        )

    elif method == "BNLL":

        def BNLL_mu(
            theshold,
            nb_bins,
            mu_init,
            weight_exp_hold,
            label_exp_hold,
            score_exp_hold,
            score_test,
            weight_test,
        ):

            def BNLL_mu_parameter(
                theshold,
                nb_bins,
                weight_exp_hold,
                label_exp_hold,
                score_exp_hold,
                weight_test,
                score_test,
            ):
                weight_ROIscore_exp_holdout = weight_exp_hold[
                    score_exp_hold > threshold
                ]
                label_Roiscore_exp_holdout = label_exp_hold[score_exp_hold > threshold]
                score_ROIscore_exp_holdout = score_exp_hold[score_exp_hold > threshold]
                score_ROIscore_test = score_test[score_test > threshold]
                weight_ROIscore_test = weight_test[score_test > threshold]
                # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
                Bins_edges = np.linspace(0, 1, nb_bins + 1)
                sig_exp_holdout_bnll = np.histogram(
                    score_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 1],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        label_Roiscore_exp_holdout == 1
                    ],
                )[0]
                bkg_exp_holdout_bnll = np.histogram(
                    score_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 0],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        label_Roiscore_exp_holdout == 0
                    ],
                )[0]
                n_obs_test_bnll = np.histogram(
                    score_ROIscore_test, bins=Bins_edges, weights=weight_ROIscore_test
                )[0]

                return n_obs_test_bnll, sig_exp_holdout_bnll, bkg_exp_holdout_bnll

            n_obs_test_bnll, sig_exp_holdout_bnll, bkg_exp_holdout_bnll = (
                BNLL_mu_parameter(
                    theshold=theshold,
                    nb_bins=nb_bins,
                    weight_exp_hold=weight_exp_hold,
                    label_exp_hold=label_exp_hold,
                    score_exp_hold=score_exp_hold,
                    score_test=score_test,
                    weight_test=weight_test,
                )
            )

            def Cost_bnll(mu):
                N_exp_bnll = Model(
                    mu=mu, sig=sig_exp_holdout_bnll, bkg=bkg_exp_holdout_bnll
                )

                return -np.sum(-N_exp_bnll + n_obs_test_bnll * np.log(N_exp_bnll))

            def BNLL_mu_computation(mu_init):
                m_bnll = Minuit(Cost_bnll, mu=mu_init)
                m_bnll.limits["mu"] = (0, None)
                m_bnll.errordef = Minuit.LIKELIHOOD

                m_bnll.migrad()
                m_bnll.hesse()
                # m_bnll.draw_mnprofile("mu")
                mu = m_bnll.values["mu"]
                del_mu_stat = m_bnll.errors["mu"]
                del_mu_stat_inf = -del_mu_stat
                del_mu_stat_sup = del_mu_stat

                return mu, del_mu_stat, del_mu_stat_inf, del_mu_stat_sup

            return BNLL_mu_computation(mu_init=mu_init)

        mu, del_mu_stat, del_mu_stat_inf, del_mu_stat_sup = BNLL_mu(
            theshold=threshold,
            nb_bins=nb_bins,
            mu_init=mu_init,
            weight_exp_hold=weight_exp_hold,
            label_exp_hold=label_exp_hold,
            score_exp_hold=score_exp_hold,
            weight_test=weight_test,
            score_test=score_test,
        )

    elif method == "BNLL_syst":

        def BNLL_syst_mu(
            theshold,
            nb_bins,
            mu_init,
            weight_exp_hold,
            label_exp_hold,
            score_exp_hold,
            score_test,
            weight_test,
        ):
            import os
            from systematic_analysis import (
                Polynomial_Reg_Model_forced_jes_tes,
                Polynomial_Reg_Model_forced_soft_met,
            )

            def BNLL_syst_mu_fitting_param(nb_bins):
                # # Load
                FitingData = [None] * 3
                SystName = ["TES", "JES", "SOFTMET"]
                from systematic_analysis import (
                    regression_tes,
                    regression_jes,
                    regression_soft_met,
                )

                regression_func_list = {
                    "TES": regression_tes,
                    "JES": regression_jes,
                    "SOFTMET": regression_soft_met,
                }

                current_dir = os.path.dirname(os.path.abspath(__file__))
                for i in range(3):  #######ATTENTION BESOIN DE METTRE A 3 POUR SOFT MET
                    if os.path.isfile(
                        "current_dir/Fitting_Parameters/one_bkg/%s_%sbins_2orderFittingParam.npz"
                        % (SystName[i], nb_bins)
                    ):
                        FitingData[i] = np.load(
                            "current_dir/Fitting_Parameters/one_bkg/%s_%sbins_2orderFittingParam.npz"
                            % (SystName[i], nb_bins)
                        )
                    else:
                        print("A file is missing")

                return FitingData

            FitingData = BNLL_syst_mu_fitting_param(nb_bins=nb_bins)

            # print(data['fit_param'])
            # print(data['fit_cov'])

            def BNLL_syst_mu_parameter(
                threshold,
                nb_bins,
                weight_exp_hold,
                label_exp_hold,
                score_exp_hold,
                weight_test,
                score_test,
            ):
                weight_ROIscore_exp_holdout = weight_exp_hold[
                    score_exp_hold > threshold
                ]
                label_Roiscore_exp_holdout = label_exp_hold[score_exp_hold > threshold]
                score_ROIscore_exp_holdout = score_exp_hold[score_exp_hold > threshold]
                score_ROIscore_test = score_test[score_test > threshold]
                weight_ROIscore_test = weight_test[score_test > threshold]

                Bins_edges = np.linspace(0, 1, nb_bins + 1)
                sig_exp_hold_unbiaised = np.histogram(
                    score_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 1],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        label_Roiscore_exp_holdout == 1
                    ],
                )[0]

                bkg_exp_hold_unbiaised = np.histogram(
                    score_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 0],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        label_Roiscore_exp_holdout == 0
                    ],
                )[0]

                n_obs_test_biaised = np.histogram(
                    score_ROIscore_test, bins=Bins_edges, weights=weight_ROIscore_test
                )[0]

                return (
                    n_obs_test_biaised,
                    sig_exp_hold_unbiaised,
                    bkg_exp_hold_unbiaised,
                )

            n_obs_test_biaised, sig_exp_hold_unbiaised, bkg_exp_hold_unbiaised = (
                BNLL_syst_mu_parameter(
                    threshold=threshold,
                    nb_bins=nb_bins,
                    weight_exp_hold=weight_exp_hold,
                    label_exp_hold=label_exp_hold,
                    score_exp_hold=score_exp_hold,
                    weight_test=weight_test,
                    score_test=score_test,
                )
            )

            def Cost_nll_syst(mu, tes, jes):  # tes, jes, soft_met):
                sig_exp_hold_biaised = (
                    sig_exp_hold_unbiaised
                    ###Below tes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                tes, *FitingData[0]["fit_param"][j][0]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    ###Below jes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                jes, *FitingData[1]["fit_param"][j][0]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    ###Below soft_met
                    # + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][0]) for j in range(nb_bins)
                    # ])
                )
                bkg_exp_hold_biaised = (
                    bkg_exp_hold_unbiaised
                    ###Below tes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                tes, *FitingData[0]["fit_param"][j][1]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    ###Below jes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                jes, *FitingData[1]["fit_param"][j][1]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    ###Below soft_met
                    # + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][1]) for j in range(nb_bins)
                    # ])
                )

                N_exp_hold = Model(
                    mu=mu, sig=sig_exp_hold_biaised, bkg=bkg_exp_hold_biaised
                )  # Clip also usefull for the BNLL because it's based on the UNLL

                return -np.sum(-N_exp_hold + n_obs_test_biaised * np.log(N_exp_hold))

            def BNLL_syst_mu_computation(
                mu_init, tes_init, jes_init
            ):  # ,tes_init ,jes_init ,soft_met_init):

                m_bnll_syst = Minuit(
                    Cost_nll_syst, mu=mu_init, tes=tes_init, jes=jes_init
                )  # tes=tes_init, jes=jes_init, soft_met=soft_met_init)
                m_bnll_syst.limits["mu"] = (0, None)
                m_bnll_syst.limits["tes"] = (0.9, 1.1)
                m_bnll_syst.limits["jes"] = (0.9, 1.1)
                # m_bnll_syst.limits["soft_met"] = (0, 5)
                m_bnll_syst.errordef = Minuit.LIKELIHOOD
                # m_bnll_syst.fixed["mu"] = True
                # m_bnll_syst.migrad()
                # m_bnll_syst.fixed["mu"] = False
                m_bnll_syst.migrad()
                m_bnll_syst.hesse()
                # m_bnll_syst.draw_mnmatrix(cl=[1, 2, 3])
                # m_bnll_syst.draw_mnprofile("mu")

                del_mu_stat = m_bnll_syst.errors["mu"]
                mu = m_bnll_syst.values["mu"]
                tes = m_bnll_syst.values["tes"]
                # tes=1
                jes = m_bnll_syst.values["jes"]
                # jes=1
                # soft_met = m_bnll_syst.values["soft_met"]
                soft_met = 0

                del_mu_stat_inf = -del_mu_stat
                del_mu_stat_sup = del_mu_stat

                # print("tes estimation =",tes," | jes estimation =",jes," | soft_met estimation =",soft_met)

                return (
                    mu,
                    del_mu_stat,
                    del_mu_stat_inf,
                    del_mu_stat_sup,
                    tes,
                    jes,
                    soft_met,
                )

            return BNLL_syst_mu_computation(
                mu_init=mu_init, tes_init=1, jes_init=1
            )  # tes_init=1 ,jes_init=1 ,soft_met_init=0)

        mu, del_mu_stat, del_mu_stat_inf, del_mu_stat_sup, tes, jes, soft_met = (
            BNLL_syst_mu(
                theshold=threshold,
                nb_bins=nb_bins,
                mu_init=mu_init,
                weight_exp_hold=weight_exp_hold,
                label_exp_hold=label_exp_hold,
                score_exp_hold=score_exp_hold,
                weight_test=weight_test,
                score_test=score_test,
            )
        )

    elif method == "BNLL_syst_normal_bkg":

        def BNLL_syst_norm_mu(
            theshold,
            nb_bins,
            mu_init,
            weight_exp_hold,
            label_exp_hold,
            score_exp_hold,
            score_test,
            weight_test,
        ):

            def BNLL_syst_norm_mu_parameter(
                threshold,
                nb_bins,
                weight_exp_hold,
                label_exp_hold,
                score_exp_hold,
                weight_test,
                score_test,
            ):
                weight_ROIscore_exp_holdout = weight_exp_hold[
                    score_exp_hold > threshold
                ]
                detailed_labels_Roiscore_exp_holdout = detailed_labels_exp_hold[
                    score_exp_hold > threshold
                ]
                score_ROIscore_exp_holdout = score_exp_hold[score_exp_hold > threshold]
                score_ROIscore_test = score_test[score_test > threshold]
                weight_ROIscore_test = weight_test[score_test > threshold]

                Bins_edges = np.linspace(0, 1, nb_bins + 1)
                sig_exp_hold = np.histogram(
                    score_ROIscore_exp_holdout[
                        detailed_labels_Roiscore_exp_holdout == "htautau"
                    ],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        detailed_labels_Roiscore_exp_holdout == "htautau"
                    ],
                )[0]

                bkg_scale_ztautau_exp_hold = np.histogram(
                    score_ROIscore_exp_holdout[detailed_labels_exp_hold == "ztautau"],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        detailed_labels_exp_hold == "ztautau"
                    ],
                )[0]

                bkg_ttbar_exp_hold = np.histogram(
                    score_ROIscore_exp_holdout[detailed_labels_exp_hold == "ttbar"],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        detailed_labels_exp_hold == "ttbar"
                    ],
                )[0]

                bkg_diboson_exp_hold = np.histogram(
                    score_ROIscore_exp_holdout[detailed_labels_exp_hold == "diboson"],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        detailed_labels_exp_hold == "diboson"
                    ],
                )[0]

                n_obs_test_biaised = np.histogram(
                    score_ROIscore_test, bins=Bins_edges, weights=weight_ROIscore_test
                )[0]

                return (
                    n_obs_test_biaised,
                    sig_exp_hold,
                    bkg_scale_ztautau_exp_hold,
                    bkg_ttbar_exp_hold,
                    bkg_diboson_exp_hold,
                )

            (
                n_obs_test_biaised,
                sig_exp_hold,
                bkg_scale_ztautau_exp_hold,
                bkg_ttbar_exp_hold,
                bkg_diboson_exp_hold,
            ) = BNLL_syst_norm_mu_parameter(
                threshold=threshold,
                nb_bins=nb_bins,
                weight_exp_hold=weight_exp_hold,
                label_exp_hold=label_exp_hold,
                score_exp_hold=score_exp_hold,
                weight_test=weight_test,
                score_test=score_test,
            )

            def Cost_nll_syst_norm(
                mu, bkg_scale, ttbar_scale, diboson_scale
            ):  # bkg_scale, ttbar_scale, diboson_scale tes, jes, soft_met):

                bkg_exp_hold_biaised = (
                    (1 + bkg_scale) * bkg_scale_ztautau_exp_hold
                    + (1 + bkg_scale) * (1 + ttbar_scale) * bkg_ttbar_exp_hold
                    + (1 + bkg_scale) * (1 + diboson_scale) * bkg_diboson_exp_hold
                )

                N_exp_hold = Model(
                    mu=mu, sig=sig_exp_hold, bkg=bkg_exp_hold_biaised
                )  # Clip also usefull for the BNLL because it's based on the UNLL

                return -np.sum(-N_exp_hold + n_obs_test_biaised * np.log(N_exp_hold))

            def BNLL_syst_norm_mu_computation(
                mu_init, bkg_scale_init, ttbar_scale_init, diboson_scale_init
            ):  # bkg_scale_init, ttbar_scale_init, diboson_scale_init ,tes_init ,jes_init ,soft_met_init):

                m_bnll_syst = Minuit(
                    Cost_nll_syst_norm,
                    mu=mu_init,
                    bkg_scale=bkg_scale_init,
                    ttbar_scale=ttbar_scale_init,
                    diboson_scale=diboson_scale_init,
                )  # bkg_scale=bkg_scale_init, ttbar_scale=ttbar_scale_init, diboson_scale=diboson_scale_init tes=tes_init, jes=jes_init, soft_met=soft_met_init)
                m_bnll_syst.limits["mu"] = (0, None)
                m_bnll_syst.limits["bkg_scale"] = (-0.01, 0.01)
                m_bnll_syst.limits["ttbar_scale"] = (-0.2, 0.2)
                m_bnll_syst.limits["diboson_scale"] = (-1, 1)
                m_bnll_syst.errordef = Minuit.LIKELIHOOD
                # m_bnll_syst.fixed["mu"] = True
                # m_bnll_syst.migrad()
                # m_bnll_syst.fixed["mu"] = False
                m_bnll_syst.migrad()
                m_bnll_syst.hesse()
                # m_bnll_syst.draw_mnmatrix(cl=[1, 2, 3,4])
                # m_bnll_syst.draw_mnprofile("mu")

                del_mu_stat = m_bnll_syst.errors["mu"]
                mu = m_bnll_syst.values["mu"]
                bkg_scale = m_bnll_syst.values["bkg_scale"]
                ttbar_scale = m_bnll_syst.values["ttbar_scale"]
                diboson_scale = m_bnll_syst.values["diboson_scale"]

                del_mu_stat_inf = -del_mu_stat
                del_mu_stat_sup = del_mu_stat

                # print("bkg_scale estimation =",bkg_scale," | ttbar_scale estimation =",ttbar_scale," | diboson_scale estimation =",diboson_scale)

                return (
                    mu,
                    del_mu_stat,
                    del_mu_stat_inf,
                    del_mu_stat_sup,
                    bkg_scale,
                    ttbar_scale,
                    diboson_scale,
                )

            return BNLL_syst_norm_mu_computation(
                mu_init=mu_init,
                bkg_scale_init=0,
                ttbar_scale_init=0,
                diboson_scale_init=0,
            )  # tes_init=1 ,jes_init=1 ,soft_met_init=0)

        (
            mu,
            del_mu_stat,
            del_mu_stat_inf,
            del_mu_stat_sup,
            bkg_scale,
            ttbar_scale,
            diboson_scale,
        ) = BNLL_syst_norm_mu(
            theshold=threshold,
            nb_bins=nb_bins,
            mu_init=mu_init,
            weight_exp_hold=weight_exp_hold,
            label_exp_hold=label_exp_hold,
            score_exp_hold=score_exp_hold,
            weight_test=weight_test,
            score_test=score_test,
        )
        tes = 1
        jes = 1
        soft_met = 0

    elif method == "BNLL_all_syst":

        def BNLL_all_syst_mu(
            theshold,
            nb_bins,
            mu_init,
            weight_exp_hold,
            label_exp_hold,
            score_exp_hold,
            score_test,
            weight_test,
        ):
            import os
            from systematic_analysis import (
                Polynomial_Reg_Model_forced_jes_tes,
                Polynomial_Reg_Model_forced_soft_met,
            )

            def BNLL_all_syst_mu_fitting_param(nb_bins):
                # # Load
                FitingData = [None] * 3
                SystName = ["TES", "JES", "SOFTMET"]
                from systematic_analysis import (
                    regression_tes,
                    regression_jes,
                    regression_soft_met,
                )

                regression_func_list = {
                    "TES": regression_tes,
                    "JES": regression_jes,
                    "SOFTMET": regression_soft_met,
                }

                current_dir = os.path.dirname(os.path.abspath(__file__))
                for i in range(3):  #######ATTENTION BESOIN DE METTRE A 3 for all
                    if os.path.isfile(
                        "current_dir/Fitting_Parameters/bkg_subchannel/%s_3bkg_%sbins_2orderFittingParam.npz"
                        % (SystName[i], nb_bins)
                    ):
                        FitingData[i] = np.load(
                            "current_dir/Fitting_Parameters/bkg_subchannel/%s_3bkg_%sbins_2orderFittingParam.npz"
                            % (SystName[i], nb_bins)
                        )
                    else:
                        print("A file is missing \n ~~~~ \n ~~~ \n ~~ \n ~")

                return FitingData

            FitingData = BNLL_all_syst_mu_fitting_param(nb_bins=nb_bins)

            # print(data['fit_param'])
            # print(data['fit_cov'])

            def BNLL_all_syst_mu_parameter(
                threshold,
                nb_bins,
                weight_exp_hold,
                label_exp_hold,
                score_exp_hold,
                weight_test,
                score_test,
            ):
                weight_ROIscore_exp_holdout = weight_exp_hold[
                    score_exp_hold > threshold
                ]
                detailed_labels_Roiscore_exp_holdout = detailed_labels_exp_hold[
                    score_exp_hold > threshold
                ]
                score_ROIscore_exp_holdout = score_exp_hold[score_exp_hold > threshold]
                score_ROIscore_test = score_test[score_test > threshold]
                weight_ROIscore_test = weight_test[score_test > threshold]

                Bins_edges = np.linspace(0, 1, nb_bins + 1)
                sig_exp_hold_unbiaised = np.histogram(
                    score_ROIscore_exp_holdout[
                        detailed_labels_Roiscore_exp_holdout == "htautau"
                    ],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        detailed_labels_Roiscore_exp_holdout == "htautau"
                    ],
                )[0]

                bkg_scale_ztautau_exp_hold = np.histogram(
                    score_ROIscore_exp_holdout[detailed_labels_exp_hold == "ztautau"],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        detailed_labels_exp_hold == "ztautau"
                    ],
                )[0]

                bkg_ttbar_exp_hold = np.histogram(
                    score_ROIscore_exp_holdout[detailed_labels_exp_hold == "ttbar"],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        detailed_labels_exp_hold == "ttbar"
                    ],
                )[0]

                bkg_diboson_exp_hold = np.histogram(
                    score_ROIscore_exp_holdout[detailed_labels_exp_hold == "diboson"],
                    bins=Bins_edges,
                    weights=weight_ROIscore_exp_holdout[
                        detailed_labels_exp_hold == "diboson"
                    ],
                )[0]

                n_obs_test_biaised = np.histogram(
                    score_ROIscore_test, bins=Bins_edges, weights=weight_ROIscore_test
                )[0]

                return (
                    n_obs_test_biaised,
                    sig_exp_hold_unbiaised,
                    bkg_scale_ztautau_exp_hold,
                    bkg_ttbar_exp_hold,
                    bkg_diboson_exp_hold,
                )

            (
                n_obs_test_biaised,
                sig_exp_hold_unbiaised,
                bkg_scale_ztautau_exp_hold,
                bkg_ttbar_exp_hold,
                bkg_diboson_exp_hold,
            ) = BNLL_all_syst_mu_parameter(
                threshold=threshold,
                nb_bins=nb_bins,
                weight_exp_hold=weight_exp_hold,
                label_exp_hold=label_exp_hold,
                score_exp_hold=score_exp_hold,
                weight_test=weight_test,
                score_test=score_test,
            )

            def Cost_nll_all_syst(
                mu, bkg_scale, ttbar_scale, diboson_scale, tes, jes, soft_met
            ):  # tes, jes, soft_met, bkg_scale, ttbar_scale, diboson_scale :
                sig_exp_hold_biaised = (
                    sig_exp_hold_unbiaised
                    # Below tes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                tes, *FitingData[0]["fit_param"][j][0]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    # Below jes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                jes, *FitingData[1]["fit_param"][j][0]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    # Below soft_met
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_soft_met(
                                soft_met, *FitingData[2]["fit_param"][j][0]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                )
                ###############################################################
                ######## First we define the tes, jes and soft met syst and then the scaling for ztautau,ttbar and diboson
                ###############################################################

                bkg_scale_ztautau_exp_hold_biaised = (
                    bkg_scale_ztautau_exp_hold
                    # Below tes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                tes, *FitingData[0]["fit_param"][j][1]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    # Below jes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                jes, *FitingData[1]["fit_param"][j][1]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    # Below soft_met
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_soft_met(
                                soft_met, *FitingData[2]["fit_param"][j][1]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                )

                bkg_ttbar_exp_hold_exp_hold_biaised = (
                    bkg_ttbar_exp_hold
                    # Below tes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                tes, *FitingData[0]["fit_param"][j][2]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    # Below jes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                jes, *FitingData[1]["fit_param"][j][2]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    # Below soft_met
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_soft_met(
                                soft_met, *FitingData[2]["fit_param"][j][2]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                )

                bkg_diboson_exp_hold_biaised = (
                    bkg_diboson_exp_hold
                    # Below tes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                tes, *FitingData[0]["fit_param"][j][3]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    # Below jes
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_jes_tes(
                                jes, *FitingData[1]["fit_param"][j][3]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                    # Below soft_met
                    + np.array(
                        [
                            Polynomial_Reg_Model_forced_soft_met(
                                soft_met, *FitingData[2]["fit_param"][j][3]
                            )
                            for j in range(nb_bins)
                        ]
                    )
                )

                bkg_exp_hold_biaised = (
                    (1 + bkg_scale) * bkg_scale_ztautau_exp_hold_biaised
                    + (1 + bkg_scale)
                    * (1 + ttbar_scale)
                    * bkg_ttbar_exp_hold_exp_hold_biaised
                    + (1 + bkg_scale)
                    * (1 + diboson_scale)
                    * bkg_diboson_exp_hold_biaised
                )

                N_exp_hold = Model(
                    mu=mu, sig=sig_exp_hold_biaised, bkg=bkg_exp_hold_biaised
                )  # Clip also usefull for the BNLL because it's based on the UNLL

                return -np.sum(-N_exp_hold + n_obs_test_biaised * np.log(N_exp_hold))

            def BNLL_all_syst_mu_computation(
                mu_init,
                tes_init,
                jes_init,
                soft_met_init,
                bkg_scale_init,
                ttbar_scale_init,
                diboson_scale_init,
            ):  # bkg_scale_init, ttbar_scale_init, diboson_scale_init ,tes_init ,jes_init ,soft_met_init):

                m_bnll_syst = Minuit(
                    Cost_nll_all_syst,
                    mu=mu_init,
                    tes=tes_init,
                    jes=jes_init,
                    soft_met=soft_met_init,
                    bkg_scale=bkg_scale_init,
                    ttbar_scale=ttbar_scale_init,
                    diboson_scale=diboson_scale_init,
                )  # bkg_scale=bkg_scale_init, ttbar_scale=ttbar_scale_init, diboson_scale=diboson_scale_init tes=tes_init, jes=jes_init, soft_met=soft_met_init)
                m_bnll_syst.limits["mu"] = (0, None)
                m_bnll_syst.limits["tes"] = (0.9, 1.1)
                m_bnll_syst.limits["jes"] = (0.9, 1.1)
                m_bnll_syst.limits["soft_met"] = (0, 5)
                m_bnll_syst.limits["bkg_scale"] = (-0.01, 0.01)
                m_bnll_syst.limits["ttbar_scale"] = (-0.2, 0.2)
                m_bnll_syst.limits["diboson_scale"] = (-1, 1)
                m_bnll_syst.errordef = Minuit.LIKELIHOOD
                # m_bnll_syst.fixed["soft_met"]=True
                # m_bnll_syst.fixed["mu"] = True
                # m_bnll_syst.migrad()
                # m_bnll_syst.fixed["mu"] = False
                m_bnll_syst.migrad()
                m_bnll_syst.hesse()
                # m_bnll_syst.draw_mnmatrix(cl=[1, 2, 3,4,5,6])
                # m_bnll_syst.draw_mnprofile("mu")

                del_mu_stat = m_bnll_syst.errors["mu"]
                mu = m_bnll_syst.values["mu"]
                bkg_scale = m_bnll_syst.values["bkg_scale"]
                ttbar_scale = m_bnll_syst.values["ttbar_scale"]
                diboson_scale = m_bnll_syst.values["diboson_scale"]
                tes = m_bnll_syst.values["tes"]
                jes = m_bnll_syst.values["jes"]
                soft_met = m_bnll_syst.values["soft_met"]
                # soft_met=0

                del_mu_stat_inf = -del_mu_stat
                del_mu_stat_sup = del_mu_stat

                # print("bkg_scale estimation =",bkg_scale," | ttbar_scale estimation =",ttbar_scale," | diboson_scale estimation =",diboson_scale)

                return (
                    mu,
                    del_mu_stat,
                    del_mu_stat_inf,
                    del_mu_stat_sup,
                    tes,
                    jes,
                    soft_met,
                    bkg_scale,
                    ttbar_scale,
                    diboson_scale,
                )

            return BNLL_all_syst_mu_computation(
                mu_init=mu_init,
                tes_init=1,
                jes_init=1,
                soft_met_init=0,
                bkg_scale_init=0,
                ttbar_scale_init=0,
                diboson_scale_init=0,
            )  # tes_init=1 ,jes_init=1 ,soft_met_init=0,bkg_scale_init=0, ttbar_scale_init=0, diboson_scale_init=0)

        (
            mu,
            del_mu_stat,
            del_mu_stat_inf,
            del_mu_stat_sup,
            tes,
            jes,
            soft_met,
            bkg_scale,
            ttbar_scale,
            diboson_scale,
        ) = BNLL_all_syst_mu(
            theshold=threshold,
            nb_bins=nb_bins,
            mu_init=mu_init,
            weight_exp_hold=weight_exp_hold,
            label_exp_hold=label_exp_hold,
            score_exp_hold=score_exp_hold,
            weight_test=weight_test,
            score_test=score_test,
        )

    else:
        print(
            "There is a problem in the computing of mu : The method label is invalid."
        )

    del_mu_sys = abs(0.0 * mu)
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    mu_axis_values = np.linspace(mu - 1, mu + 1, 1_000)

    if method == "UNLL" or method == "Direct":

        def UNLL_mu_parameter(saved_info_hold, weight_test, score_test, threshold):
            Parabola_Ploteur = False
            sig_exp_holdout_unll = saved_info_hold["gamma"]
            bkg_exp_holdout_unll = saved_info_hold["beta"]
            n_obs_test_unll = np.sum(weight_test[score_test > threshold])
            # print("Compute mu, sig= ",sig_exp_holdout_unll," bkg= ",bkg_exp_holdout_unll," N= ",n_obs_test_unll)

            return n_obs_test_unll, sig_exp_holdout_unll, bkg_exp_holdout_unll

        n_obs_test_unll, sig_exp_holdout_unll, bkg_exp_holdout_unll = UNLL_mu_parameter(
            saved_info_hold=saved_info_hold,
            weight_test=weight_test,
            score_test=score_test,
            threshold=threshold,
        )

        def Cost_unll(mu):
            N_exp_unll = Model(
                mu=mu, sig=sig_exp_holdout_unll, bkg=bkg_exp_holdout_unll
            )
            # print("sig_exp_holdout: ",sig_exp_holdout," bkg_exp_holdout: ",bkg_exp_holdout," mu: ",mu," N_exp: ",N_exp," n_obs: ",n_obs_test," N_exp mu=1: ",Model(mu=1, sig=sig_exp_holdout, bkg=bkg_exp_holdout)  )
            return -(-N_exp_unll + n_obs_test_unll * np.log(N_exp_unll))

        negloglike_values = np.array([Cost_unll(mub) for mub in mu_axis_values])
        negloglike_mu_hat = Cost_unll(mu)  # saved_info["gamma"]+saved_info["beta"])
        tes = 1
        jes = 1

    elif (
        method == "BNLL"
    ):  # or (method == "BNLL_syst")or (method == "BNLL_syst_normal_bkg") or (method == "BNLL_all_syst")
        # print("\n BNLL")
        def BNLL_mu_parameter(
            threshold,
            nb_bins,
            weight_exp_hold,
            label_exp_hold,
            score_exp_hold,
            weight_test,
            score_test,
        ):
            weight_ROIscore_exp_holdout = weight_exp_hold[score_exp_hold > threshold]
            label_Roiscore_exp_holdout = label_exp_hold[score_exp_hold > threshold]
            score_ROIscore_exp_holdout = score_exp_hold[score_exp_hold > threshold]
            score_ROIscore_test = score_test[score_test > threshold]
            weight_ROIscore_test = weight_test[score_test > threshold]
            # Bins_edges=np.linspace(np.min(score_ROIscore),np.max(score_ROIscore),nb_bins+1)
            Bins_edges = np.linspace(0, 1, nb_bins + 1)
            sig_exp_holdout_bnll = np.histogram(
                score_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 1],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 1],
            )[0]
            bkg_exp_holdout_bnll = np.histogram(
                score_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 0],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 0],
            )[0]
            n_obs_test_bnll = np.histogram(
                score_ROIscore_test, bins=Bins_edges, weights=weight_ROIscore_test
            )[0]

            return n_obs_test_bnll, sig_exp_holdout_bnll, bkg_exp_holdout_bnll

        n_obs_test_bnll, sig_exp_holdout_bnll, bkg_exp_holdout_bnll = BNLL_mu_parameter(
            threshold=threshold,
            nb_bins=nb_bins,
            weight_exp_hold=weight_exp_hold,
            label_exp_hold=label_exp_hold,
            score_exp_hold=score_exp_hold,
            score_test=score_test,
            weight_test=weight_test,
        )

        def Cost_bnll(mu):
            N_exp_bnll = Model(
                mu=mu, sig=sig_exp_holdout_bnll, bkg=bkg_exp_holdout_bnll
            )

            return -np.sum(-N_exp_bnll + n_obs_test_bnll * np.log(N_exp_bnll))

        negloglike_values = np.array([Cost_bnll(mub) for mub in mu_axis_values])
        negloglike_mu_hat = Cost_bnll(mu)  # saved_info["gamma"]+saved_info["beta"])
        tes = 1
        jes = 1

    elif method == "BNLL_syst":  ######  and method=="REf" Just BLOCK THE CALCULATION
        ####################
        ############
        ########
        #####
        # print("\n BNLL_syst")
        import os
        from systematic_analysis import (
            Polynomial_Reg_Model_forced_jes_tes,
            Polynomial_Reg_Model_forced_soft_met,
        )

        def BNLL_syst_mu_fitting_param(nb_bins):
            # # Load
            FitingData = [None] * 3
            SystName = ["TES", "JES", "SOFTMET"]
            from systematic_analysis import (
                regression_tes,
                regression_jes,
                regression_soft_met,
            )

            regression_func_list = {
                "TES": regression_tes,
                "JES": regression_jes,
                "SOFTMET": regression_soft_met,
            }

            current_dir = os.path.dirname(os.path.abspath(__file__))
            for i in range(3):  #######ATTENTION BESOIN DE METTRE A 3 POUR SOFT MET
                if os.path.isfile(
                    "current_dir/Fitting_Parameters/one_bkg/%s_%sbins_2orderFittingParam.npz"
                    % (SystName[i], nb_bins)
                ):
                    FitingData[i] = np.load(
                        "current_dir/Fitting_Parameters/one_bkg/%s_%sbins_2orderFittingParam.npz"
                        % (SystName[i], nb_bins)
                    )
                else:
                    print("A file is missing")

            return FitingData

        FitingData = BNLL_syst_mu_fitting_param(nb_bins=nb_bins)

        def BNLL_syst_mu_parameter(
            threshold,
            nb_bins,
            weight_exp_hold,
            label_exp_hold,
            score_exp_hold,
            weight_test,
            score_test,
        ):

            weight_ROIscore_exp_holdout = weight_exp_hold[score_exp_hold > threshold]
            label_Roiscore_exp_holdout = label_exp_hold[score_exp_hold > threshold]
            score_ROIscore_exp_holdout = score_exp_hold[score_exp_hold > threshold]
            score_ROIscore_test = score_test[score_test > threshold]
            weight_ROIscore_test = weight_test[score_test > threshold]

            Bins_edges = np.linspace(0, 1, nb_bins + 1)
            sig_exp_hold_unbiaised = np.histogram(
                score_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 1],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 1],
            )[0]

            bkg_exp_hold_unbiaised = np.histogram(
                score_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 0],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 0],
            )[0]

            n_obs_test_biaised = np.histogram(
                score_ROIscore_test, bins=Bins_edges, weights=weight_ROIscore_test
            )[0]

            return n_obs_test_biaised, sig_exp_hold_unbiaised, bkg_exp_hold_unbiaised

        n_obs_test_biaised, sig_exp_hold_unbiaised, bkg_exp_hold_unbiaised = (
            BNLL_syst_mu_parameter(
                threshold=threshold,
                nb_bins=nb_bins,
                weight_exp_hold=weight_exp_hold,
                label_exp_hold=label_exp_hold,
                score_exp_hold=score_exp_hold,
                weight_test=weight_test,
                score_test=score_test,
            )
        )

        def Cost_nll_syst(mu, tes):  # ,tes, jes, soft_met):
            sig_exp_hold_biaised = (
                sig_exp_hold_unbiaised
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            tes, *FitingData[0]["fit_param"][j][0]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # +np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][0]) for j in range(nb_bins)
                # ])
                # + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][0]) for j in range(nb_bins)
                # ])
            )
            bkg_exp_hold_biaised = (
                bkg_exp_hold_unbiaised
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            tes, *FitingData[0]["fit_param"][j][1]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # +np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][1]) for j in range(nb_bins)
                # ])
                # + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][1]) for j in range(nb_bins)
                # ])
            )

            # +np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][1]) for j in range(nb_bins)])  )
            # n_obs_test=(    n_obs_test_biaised-np.array([Polynomial_Reg_Model_forced_jes_tes(tes,*FitingData[0]["fit_param"][j][2]) for j in range(nb_bins)]) )
            # -np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][2]) for j in range(nb_bins)])   )

            N_exp_hold = Model(
                mu=mu, sig=sig_exp_hold_biaised, bkg=bkg_exp_hold_biaised
            )  # Clip also usefull for the BNLL because it's based on the UNLL

            return -np.sum(-N_exp_hold + n_obs_test_biaised * np.log(N_exp_hold))

        negloglike_values = np.array(
            [
                Cost_nll_syst(mu=mub, tes=tes) for mub in mu_axis_values
            ]  # tes=tes, jes=jes, soft_met=soft_met
        )
        negloglike_mu_hat = Cost_nll_syst(
            mu=mu, tes=tes
        )  # tes=tes,  jes=jes, soft_met=soft_met)

    elif method == "BNLL_syst_normal_bkg":

        def BNLL_syst_norm_mu_parameter(
            threshold,
            nb_bins,
            weight_exp_hold,
            label_exp_hold,
            score_exp_hold,
            weight_test,
            score_test,
        ):
            weight_ROIscore_exp_holdout = weight_exp_hold[score_exp_hold > threshold]
            label_Roiscore_exp_holdout = label_exp_hold[score_exp_hold > threshold]
            score_ROIscore_exp_holdout = score_exp_hold[score_exp_hold > threshold]
            score_ROIscore_test = score_test[score_test > threshold]
            weight_ROIscore_test = weight_test[score_test > threshold]

            Bins_edges = np.linspace(0, 1, nb_bins + 1)
            sig_exp_hold = np.histogram(
                score_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 1],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[label_Roiscore_exp_holdout == 1],
            )[0]

            bkg_scale_ztautau_exp_hold = np.histogram(
                score_ROIscore_exp_holdout[detailed_labels_exp_hold == "ztautau"],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[
                    detailed_labels_exp_hold == "ztautau"
                ],
            )[0]

            bkg_ttbar_exp_hold = np.histogram(
                score_ROIscore_exp_holdout[detailed_labels_exp_hold == "ttbar"],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[
                    detailed_labels_exp_hold == "ttbar"
                ],
            )[0]

            bkg_diboson_exp_hold = np.histogram(
                score_ROIscore_exp_holdout[detailed_labels_exp_hold == "diboson"],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[
                    detailed_labels_exp_hold == "diboson"
                ],
            )[0]

            n_obs_test_biaised = np.histogram(
                score_ROIscore_test, bins=Bins_edges, weights=weight_ROIscore_test
            )[0]

            return (
                n_obs_test_biaised,
                sig_exp_hold,
                bkg_scale_ztautau_exp_hold,
                bkg_ttbar_exp_hold,
                bkg_diboson_exp_hold,
            )

        (
            n_obs_test_biaised,
            sig_exp_hold,
            bkg_scale_ztautau_exp_hold,
            bkg_ttbar_exp_hold,
            bkg_diboson_exp_hold,
        ) = BNLL_syst_norm_mu_parameter(
            threshold=threshold,
            nb_bins=nb_bins,
            weight_exp_hold=weight_exp_hold,
            label_exp_hold=label_exp_hold,
            score_exp_hold=score_exp_hold,
            weight_test=weight_test,
            score_test=score_test,
        )

        def Cost_nll_syst_norm(
            mu, bkg_scale, ttbar_scale, diboson_scale
        ):  # bkg_scale, ttbar_scale, diboson_scale tes, jes, soft_met):

            bkg_exp_hold_biaised = (
                (1 + bkg_scale) * bkg_scale_ztautau_exp_hold
                + (1 + bkg_scale) * (1 + ttbar_scale) * bkg_ttbar_exp_hold
                + (1 + bkg_scale) * (1 + diboson_scale) * bkg_diboson_exp_hold
            )

            N_exp_hold = Model(
                mu=mu, sig=sig_exp_hold, bkg=bkg_exp_hold_biaised
            )  # Clip also usefull for the BNLL because it's based on the UNLL

            return -np.sum(-N_exp_hold + n_obs_test_biaised * np.log(N_exp_hold))

        negloglike_values = np.array(
            [
                Cost_nll_syst_norm(
                    mu=mub,
                    bkg_scale=bkg_scale,
                    ttbar_scale=ttbar_scale,
                    diboson_scale=diboson_scale,
                )
                for mub in mu_axis_values
            ]  # tes=tes, jes=jes, soft_met=soft_met
        )
        negloglike_mu_hat = Cost_nll_syst_norm(
            mu=mu,
            bkg_scale=bkg_scale,
            ttbar_scale=ttbar_scale,
            diboson_scale=diboson_scale,
        )  # tes=tes,  jes=jes, soft_met=soft_met)

    elif method == "BNLL_all_syst":
        import os
        from systematic_analysis import (
            Polynomial_Reg_Model_forced_jes_tes,
            Polynomial_Reg_Model_forced_soft_met,
        )

        # print("good negloglik BNLL_all_syst")
        def BNLL_all_syst_mu_fitting_param(nb_bins):
            # # Load
            FitingData = [None] * 3
            SystName = ["TES", "JES", "SOFTMET"]
            from systematic_analysis import (
                regression_tes,
                regression_jes,
                regression_soft_met,
            )

            regression_func_list = {
                "TES": regression_tes,
                "JES": regression_jes,
                "SOFTMET": regression_soft_met,
            }

            current_dir = os.path.dirname(os.path.abspath(__file__))
            for i in range(3):  #######ATTENTION BESOIN DE METTRE A 3 for all
                if os.path.isfile(
                    "current_dir/Fitting_Parameters/bkg_subchannel/%s_3bkg_%sbins_2orderFittingParam.npz"
                    % (SystName[i], nb_bins)
                ):
                    FitingData[i] = np.load(
                        "current_dir/Fitting_Parameters/bkg_subchannel/%s_3bkg_%sbins_2orderFittingParam.npz"
                        % (SystName[i], nb_bins)
                    )
                else:
                    print("A file is missing \n ~~~~ \n ~~~ \n ~~ \n ~")

            return FitingData

        FitingData = BNLL_all_syst_mu_fitting_param(nb_bins=nb_bins)

        # print(data['fit_param'])
        # print(data['fit_cov'])

        def BNLL_all_syst_mu_parameter(
            threshold,
            nb_bins,
            weight_exp_hold,
            label_exp_hold,
            score_exp_hold,
            weight_test,
            score_test,
        ):
            weight_ROIscore_exp_holdout = weight_exp_hold[score_exp_hold > threshold]
            detailed_labels_Roiscore_exp_holdout = detailed_labels_exp_hold[
                score_exp_hold > threshold
            ]
            score_ROIscore_exp_holdout = score_exp_hold[score_exp_hold > threshold]
            score_ROIscore_test = score_test[score_test > threshold]
            weight_ROIscore_test = weight_test[score_test > threshold]

            Bins_edges = np.linspace(0, 1, nb_bins + 1)
            sig_exp_hold_unbiaised = np.histogram(
                score_ROIscore_exp_holdout[
                    detailed_labels_Roiscore_exp_holdout == "htautau"
                ],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[
                    detailed_labels_Roiscore_exp_holdout == "htautau"
                ],
            )[0]

            bkg_scale_ztautau_exp_hold = np.histogram(
                score_ROIscore_exp_holdout[detailed_labels_exp_hold == "ztautau"],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[
                    detailed_labels_exp_hold == "ztautau"
                ],
            )[0]

            bkg_ttbar_exp_hold = np.histogram(
                score_ROIscore_exp_holdout[detailed_labels_exp_hold == "ttbar"],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[
                    detailed_labels_exp_hold == "ttbar"
                ],
            )[0]

            bkg_diboson_exp_hold = np.histogram(
                score_ROIscore_exp_holdout[detailed_labels_exp_hold == "diboson"],
                bins=Bins_edges,
                weights=weight_ROIscore_exp_holdout[
                    detailed_labels_exp_hold == "diboson"
                ],
            )[0]

            n_obs_test_biaised = np.histogram(
                score_ROIscore_test, bins=Bins_edges, weights=weight_ROIscore_test
            )[0]

            return (
                n_obs_test_biaised,
                sig_exp_hold_unbiaised,
                bkg_scale_ztautau_exp_hold,
                bkg_ttbar_exp_hold,
                bkg_diboson_exp_hold,
            )

        (
            n_obs_test_biaised,
            sig_exp_hold_unbiaised,
            bkg_scale_ztautau_exp_hold,
            bkg_ttbar_exp_hold,
            bkg_diboson_exp_hold,
        ) = BNLL_all_syst_mu_parameter(
            threshold=threshold,
            nb_bins=nb_bins,
            weight_exp_hold=weight_exp_hold,
            label_exp_hold=label_exp_hold,
            score_exp_hold=score_exp_hold,
            weight_test=weight_test,
            score_test=score_test,
        )

        def Cost_nll_all_syst(
            mu, bkg_scale, ttbar_scale, diboson_scale, tes, jes, soft_met
        ):  # tes, jes, soft_met, bkg_scale, ttbar_scale, diboson_scale :
            sig_exp_hold_biaised = (
                sig_exp_hold_unbiaised
                # Below tes
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            tes, *FitingData[0]["fit_param"][j][0]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # Below jes
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            jes, *FitingData[1]["fit_param"][j][0]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # Below soft_met
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_soft_met(
                            soft_met, *FitingData[2]["fit_param"][j][0]
                        )
                        for j in range(nb_bins)
                    ]
                )
            )
            ###############################################################
            ######## First we define the tes, jes and soft met syst and then the scaling for ztautau,ttbar and diboson
            ###############################################################

            bkg_scale_ztautau_exp_hold_biaised = (
                bkg_scale_ztautau_exp_hold
                # Below tes
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            tes, *FitingData[0]["fit_param"][j][1]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # Below jes
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            jes, *FitingData[1]["fit_param"][j][1]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # Below soft_met
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_soft_met(
                            soft_met, *FitingData[2]["fit_param"][j][1]
                        )
                        for j in range(nb_bins)
                    ]
                )
            )

            bkg_ttbar_exp_hold_exp_hold_biaised = (
                bkg_ttbar_exp_hold
                # Below tes
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            tes, *FitingData[0]["fit_param"][j][2]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # Below jes
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            jes, *FitingData[1]["fit_param"][j][2]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # Below soft_met
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_soft_met(
                            soft_met, *FitingData[2]["fit_param"][j][2]
                        )
                        for j in range(nb_bins)
                    ]
                )
            )

            bkg_diboson_exp_hold_biaised = (
                bkg_diboson_exp_hold
                # Below tes
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            tes, *FitingData[0]["fit_param"][j][3]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # Below jes
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_jes_tes(
                            jes, *FitingData[1]["fit_param"][j][3]
                        )
                        for j in range(nb_bins)
                    ]
                )
                # Below soft_met
                + np.array(
                    [
                        Polynomial_Reg_Model_forced_soft_met(
                            soft_met, *FitingData[2]["fit_param"][j][3]
                        )
                        for j in range(nb_bins)
                    ]
                )
            )

            bkg_exp_hold_biaised = (
                (1 + bkg_scale) * bkg_scale_ztautau_exp_hold_biaised
                + (1 + bkg_scale)
                * (1 + ttbar_scale)
                * bkg_ttbar_exp_hold_exp_hold_biaised
                + (1 + bkg_scale) * (1 + diboson_scale) * bkg_diboson_exp_hold_biaised
            )

            N_exp_hold = Model(
                mu=mu, sig=sig_exp_hold_biaised, bkg=bkg_exp_hold_biaised
            )  # Clip also usefull for the BNLL because it's based on the UNLL

            return -np.sum(-N_exp_hold + n_obs_test_biaised * np.log(N_exp_hold))

        negloglike_values = np.array(
            [
                Cost_nll_all_syst(
                    mu=mub,
                    tes=tes,
                    jes=jes,
                    soft_met=soft_met,
                    bkg_scale=bkg_scale,
                    ttbar_scale=ttbar_scale,
                    diboson_scale=diboson_scale,
                )
                for mub in mu_axis_values
            ]  # tes=tes, jes=jes, soft_met=soft_met
        )
        negloglike_mu_hat = Cost_nll_all_syst(
            mu=mu,
            tes=tes,
            jes=jes,
            soft_met=soft_met,
            bkg_scale=bkg_scale,
            ttbar_scale=ttbar_scale,
            diboson_scale=diboson_scale,
        )  # tes=tes,  jes=jes, soft_met=soft_met)

    print("mu", mu)
    # print("Returned mu=",mu, " Delta_mu_tot=",del_mu_tot)
    # print("method in statistical analysis",method)
    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_stat_inf": del_mu_stat_inf,
        "del_mu_stat_sup": del_mu_stat_sup,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
        "negloglike_mu_hat": negloglike_mu_hat,
        "negloglike_values": negloglike_values,
        "mu_axis_values": mu_axis_values,
        "tes": tes,
        "jes": jes,
    }


def calculate_best_threshold(
    valid_test_set,
    score_valid_test=0,
    score_hold_exp=0,
    holdout_exp_set=0,
    NbPoints_Prec_Thresh=10,
    del_mu_method="Direct",
    Plot=False,
):

    threshold_list = np.linspace(0, 1, NbPoints_Prec_Thresh, endpoint=False)
    AMS_list = np.zeros(NbPoints_Prec_Thresh)
    del_mu_tot_list = np.zeros(NbPoints_Prec_Thresh)
    mu_list = np.zeros(NbPoints_Prec_Thresh)
    compute_mu_tamp = np.zeros(NbPoints_Prec_Thresh)

    for i in range(NbPoints_Prec_Thresh):  # Ajouter barre de progressio
        # print("threshold[i] in best thresh calculate ",threshold_list[i])
        saved_info_tamp_hold = calculate_saved_info(
            score_hold_exp, holdout_exp_set, threshold_list[i]
        )
        saved_info_tamp_valid = calculate_saved_info(
            score_valid_test, valid_test_set, threshold_list[i]
        )

        # print("saved_info_tamp in best thresh calculate"," sig ",saved_info_tamp["gamma"]," bkg ",saved_info_tamp["beta"])
        AMS_list[i] = calculate_AMS(saved_info_tamp_valid, 10e-2)

        compute_mu_tamp = compute_mu(
            method=del_mu_method,
            score_test=score_valid_test,
            weight_test=valid_test_set["weights"],
            label_exp_hold=holdout_exp_set["labels"],
            score_exp_hold=score_hold_exp,
            weight_exp_hold=holdout_exp_set["weights"],
            saved_info_hold=saved_info_tamp_hold,
            threshold=threshold_list[i],
        )

        del_mu_tot_list[i] = compute_mu_tamp["del_mu_tot"]
        mu_list[i] = compute_mu_tamp["mu_hat"]

    if Plot == True:
        fig, (ax1, ax2) = plt.subplots(2, 1, layout="constrained")
        ax1bis = ax1.twinx()
        ax1.axvline(
            threshold_list[np.argmin(del_mu_tot_list)],
            color="blue",
            label="%s: Min uncertainty=%s for Threshold=%s"
            % (
                del_mu_method,
                round(min(del_mu_tot_list), 3),
                round(threshold_list[np.argmin(del_mu_tot_list)], 3),
            ),
        )
        ax1.fill_between(
            threshold_list,
            mu_list - del_mu_tot_list,
            mu_list + del_mu_tot_list,
            color="lightskyblue",
            alpha=0.5,
        )
        ax1.plot(threshold_list, mu_list, marker=None, color="dodgerblue")
        ax1bis.plot(threshold_list, del_mu_tot_list, marker=None, color="darkorange")
        ax1.set_ylabel(f"$\mu$", color="dodgerblue")
        ax1bis.set_ylabel(f"$\delta\mu$", color="darkorange")
        ax1bis.tick_params(axis="y", labelcolor="darkorange")
        ax1.legend(loc="upper right")
        ax1.set_xticklabels([])

        ax2.axvline(
            threshold_list[np.argmax(AMS_list)],
            color="red",
            label="Max AMS=%s for Threshold=%s"
            % (round(max(AMS_list), 3), round(threshold_list[np.argmax(AMS_list)], 3)),
        )
        ax2.plot(threshold_list, AMS_list, marker=None, color="dodgerblue")
        ax2.set_ylabel("AMS")
        ax2.legend(loc="lower right")
        ax2.set_xlabel("Threshold")

        fig.suptitle(
            "Best threshold for AMS and delta mu method for %s model\n Size of the subset : Train=%s Holdout=%s Validation=%s"
            % (ModelType, NbTrain, NbHoldout, NbValidation)
        )
        plt.grid(True)

        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists("%s/images" % (current_dir)):
            os.makedirs("%s/images" % (current_dir))
        plt.savefig(
            "%s/images/Best_Mu_AMS_Thresh_%s_Train=%s_Holdout=%s_Valid=%s.png"
            % (current_dir, ModelType, NbTrain, NbHoldout, NbValidation)
        )
        plt.show()

    best_optimisation = {
        "best_AMS": max(AMS_list),
        "best_del_mu": min(del_mu_tot_list),
        "best_mu_AMS": mu_list[np.argmax(AMS_list)],
        "best_mu_del_mu": mu_list[np.argmin(del_mu_tot_list)],
        "AMS_best_threshold": threshold_list[np.argmax(AMS_list)],
        "del_mu_best_thresh": threshold_list[np.argmin(del_mu_tot_list)],
        "mu_del_val_method": del_mu_method,
    }
    return best_optimisation


def calculate_saved_info(
    score,
    holdout_set,
    threshold=0,
    model=0,
    # find_threshold=False,
    # NbPoints_Prec_Thresh=50,
    # Del_Mu_method="None",
):
    # from systematic_analysis import tes_fitter
    # from systematic_analysis import jes_fitter

    # print("threshold=",threshold)
    # print("saved_info score shape before threshold", score.shape)
    # print("calculate_saved_info, score before flatten",score)
    score = score.flatten() > threshold
    score = score.astype(int)

    labels = holdout_set["labels"]
    # print("calculate_saved_info, labels",labels)
    weights = holdout_set["weights"]
    # print("calculate_saved_info, weights",weights)
    # print("saved_info score shape after threshold", score.shape)
    gamma = np.sum(weights * score * labels)
    beta = np.sum(weights * score * (1 - labels))
    N = np.sum(weights * score)
    # print("saved_info, sig= ",gamma," bkg= ",beta," N= ",N)

    """
    # Modified function
    # print("score shape before threshold", holdout_set["weights"].shape)
    weight_ROIscore = holdout_set["weights"][score > threshold]
    # print("score shape after threshold", weight_ROIscore.shape)
    label_Roiscore = holdout_set["labels"][score > threshold]

    gamma = np.sum(weight_ROIscore[label_Roiscore == 1])
    beta = np.sum(weight_ROIscore[label_Roiscore == 0])
    N=np.sum(weight_ROIscore)
    """

    # del_gamma=np.sqrt(np.sum(np.power(weight_ROIscore[label_Roiscore==1], 2)))
    # del_beta=np.sqrt(np.sum(np.power(weight_ROIscore[label_Roiscore==0], 2)))
    ###

    # print(gamma/gammabis,"gamma/gammabis")
    # print(beta/betabis,"beta/betabis")

    saved_info = {
        "N": N,  # Total nb of events in the ROI
        "beta": beta,
        "gamma": gamma,
        # "del_beta": del_beta,
        # "del_gamma": del_gamma,
        # "tes_fit": tes_fitter(score, holdout_set),
        # "jes_fit": jes_fitter(score, holdout_set),
    }

    # print("saved_info", saved_info)
    return saved_info


def calculate_saved_info_new(model, holdout_set, threshold=0, nb_bins=1):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """
    from systematic_analysis import tes_fitter
    from systematic_analysis import jes_fitter

    score = model.predict(holdout_set["data"])

    #    from systematic_analysis import tes_fitter
    #    from systematic_analysis import jes_fitter

    bins = np.linspace(0, 1, NUMBER_OF_BINS)

    # Calculate saved_info with this optimised cutoff
    score_flat = score.flatten() > threshold
    score_flat = score_flat.astype(int)

    label = holdout_set["labels"]

    gamma = np.sum(holdout_set["weights"] * score_flat * label)

    beta = np.sum(holdout_set["weights"] * score_flat * (1 - label))

    N = np.sum(weights * score)

    # Binned gamma and beta
    signal_mask = label == 1
    background_mask = label == 0

    gamma_hist, _ = np.histogram(
        score[signal_mask], bins=BINS, weights=holdout_set["weights"][signal_mask]
    )

    beta_hist, _ = np.histogram(
        score[background_mask],
        bins=BINS,
        weights=holdout_set["weights"][background_mask],
    )
    """
    # Modified function
    # print("score shape before threshold", holdout_set["weights"].shape)
    weight_ROIscore = holdout_set["weights"][score > threshold]
    # print("score shape after threshold", weight_ROIscore.shape)
    label_Roiscore = holdout_set["labels"][score > threshold]

    gamma = np.sum(weight_ROIscore[label_Roiscore == 1])
    beta = np.sum(weight_ROIscore[label_Roiscore == 0])
    N=np.sum(weight_ROIscore)
    """

    # del_gamma=np.sqrt(np.sum(np.power(weight_ROIscore[label_Roiscore==1], 2)))
    # del_beta=np.sqrt(np.sum(np.power(weight_ROIscore[label_Roiscore==0], 2)))
    ###

    saved_info = {
        "N": N,
        "beta": beta,
        "gamma": gamma,
        # "tes_fit": tes_fitter(model, holdout_set),
        # "jes_fit": jes_fitter(model, holdout_set),
        "best_threshold": best_threshold,
        "gamma_hist": gamma_hist,
        "beta_hist": beta_hist,
    }

    # print("saved_info", saved_info)

    return saved_info


def calculate_AMS(saved_info, beta_reg=10e-2):
    """
    Calculate AMS based on :
    """

    AMS = np.sqrt(
        2
        * (
            (  # Not needed
                (saved_info["gamma"] + saved_info["beta"] + beta_reg)
                * np.log(1 + (saved_info["gamma"] / (saved_info["beta"] + beta_reg)))
            )
            - saved_info["gamma"]
        )
    )
    return AMS
