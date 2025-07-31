# ------------------------------
# Dummy Sample Submission
# ------------------------------

###############################################################
###############################################################
######## BNLL_all_syst METHOD
###############################################################

# ///////////////////////////////////////////////////////////////////////////////
## Cost_nll_all_syst
# ///////////////////////////////////////////////////////////////////////////////

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##First we define the tes, jes and soft met syst and then the scaling for ztautau,ttbar and diboson
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from statistical_analysis import (
    calculate_saved_info,
    compute_mu,
    calculate_AMS,
    calculate_best_threshold,
)
import numpy as np
import matplotlib.pyplot as plt

from parameter_management_scan import Parameter_Distribution

Tamp_parameter = Parameter_Distribution.get_all()

SkipTHV_OnlyPredict = Tamp_parameter["SkipTHV_OnlyPredict"]

THV_size = Tamp_parameter["THV_size"]
Nb_bins_distrib = Tamp_parameter["Nb_bins_distrib"]
threshold_distrib = Tamp_parameter["threshold_distrib"]

ModelType = Tamp_parameter["ModelType"]
Load_Classifier = Tamp_parameter["Load_Classifier"]

random_seed = Tamp_parameter["random_seed"]

Force_3bkg_regression = Tamp_parameter["Force_3bkg_regression"]
Force_1bkg_regression = Tamp_parameter["Force_1bkg_regression"]
fitting_3bkg = Tamp_parameter["fitting_3bkg"]
fitting_1bkg = Tamp_parameter["fitting_1bkg"]

Predict_method = Tamp_parameter["Predict_method"][0]
Parabola_method = Tamp_parameter["Parabola_method"]
method_used = list(set(Tamp_parameter["Predict_method"] + Parabola_method))

First_plots_hist_roc = Tamp_parameter["First_plots_hist_roc"]
Bins_varia_plot = Tamp_parameter["Bins_varia_plot"]
if Bins_varia_plot == True:
    Bins_varia_Min_Max_Step = Tamp_parameter["Bins_varia_Min_Max_Step"]
Compute_Best_Opti = Tamp_parameter["Compute_Best_Opti"]
if Compute_Best_Opti == True:
    NbPoints_Prec_Thresh = Tamp_parameter["NbPoints_Prec_Thresh"]
Features_VS_syst = Tamp_parameter["Features_VS_syst"]
Print_NLL_score_biased_stacked = Tamp_parameter["Print_NLL_score_biased_stacked"]

NbTrain = THV_size[0]
NbHoldout = THV_size[1]
NbValidation = THV_size[2]


###############################################################
######## Class Model
###############################################################
class Model:

    ###############################################################
    ######## Init
    ###############################################################
    def __init__(self, get_train_set=None, systematics=None, model_type=ModelType):
        print("###########################################")
        print("Beginning of the initialisation of the model")
        print("###########################################")
        """
        Define the THV subset : 
        First the indices and the list of events associated
        Then the panda table associated
        """

        self.get_train_set = get_train_set
        self.systematics = systematics

        # ///////////////////////////////////////////////////////////////////////////////
        ## define classification method BDT / NN / other
        # ///////////////////////////////////////////////////////////////////////////////
        if model_type == "BDT":
            from boosted_decision_tree import BoostedDecisionTree

            self.model = BoostedDecisionTree(
                train_size=NbTrain, seed=random_seed, Load_Classifier=Load_Classifier
            )

        elif model_type == "NN":
            from neural_network import NeuralNetwork

            self.model = NeuralNetwork(
                train_size=NbTrain, seed=random_seed, Load_Classifier=Load_Classifier
            )

        elif model_type == "sample_model":
            from sample_model import SampleModel

            self.model = SampleModel()

        else:
            print(f"model_type {model_type} not found")
            raise ValueError(f"model_type {model_type} not found")

        self.name = model_type
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f" Model is { self.name}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    ###############################################################
    ######## FIT
    ###############################################################

    def fit(self):

        if SkipTHV_OnlyPredict:
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists("%s/Saved_Info" % (current_dir)):
                print("Problem : Saved info Not Found")
            else:
                from joblib import load

                self.saved_info = load(
                    "%s/Saved_Info/Saved_info_%s_%sbins_computed_wt_%s_events_seed%s.pkl"
                    % (
                        current_dir,
                        ModelType,
                        Nb_bins_distrib,
                        THV_size[1],
                        random_seed,
                    )
                )
                print("Saved info loaded")
                print("###########################################")

        else:
            print("###########################################")
            print("Beginning of the fitting of the model")
            print("###########################################")
            from utils import statistical_subset_info

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #### Define index of the events for train, holdout and validation set from the whole FairUniverse dataset
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            print(
                "///////////////////////////////////////////////////////////////////////////////"
            )
            print(f"Random seed used: {random_seed}")
            print(
                "///////////////////////////////////////////////////////////////////////////////"
            )
            indices = np.arange(THV_size.sum())
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            train_indices = indices[: THV_size[0]]
            holdout_indices = indices[THV_size[0] : THV_size[0] + THV_size[1]]
            valid_indices = indices[THV_size[0] + THV_size[1] :]

            # ///////////////////////////////////////////////////////////////////////////////
            #### Train the classifier (so create the train set) or load an already trained classifier
            # ///////////////////////////////////////////////////////////////////////////////
            if Load_Classifier != True or ModelType == "sample_model":
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #### Initialise train set
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                print("Train Subset : Created")
                training_df = self.get_train_set(selected_indices=train_indices)
                training_set = {
                    "labels": training_df.pop("labels"),
                    "weights": training_df.pop("weights"),
                    "detailed_labels": training_df.pop("detailed_labels"),
                    "data": training_df,
                }
                del training_df
                # statistical_subset_info(training_set,"Training subset Before poscut")

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #### Apply systematics on the training set
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                train_set_poscut = self.systematics(
                    training_set, tes=1, jes=1, soft_met=0
                )
                del training_set
                # statistical_subset_info(train_set_poscut,"Training subset After poscut")

                # ///////////////////////////////////////////////////////////////////////////////
                ## Normalisation Weight
                ####
                ################################
                ## TO CHECK #########################################################################
                ################################""
                ##
                # ///////////////////////////////////////////////////////////////////////////////
                balanced_set = train_set_poscut
                weights_train = train_set_poscut["weights"]
                train_labels = train_set_poscut["labels"]
                class_weights_train = (
                    weights_train[train_labels == 0].sum(),
                    weights_train[train_labels == 1].sum(),
                )

                for i in range(len(class_weights_train)):  # loop on B then S target
                    # training dataset: equalize number of background and signal
                    weights_train[train_labels == i] *= (
                        max(class_weights_train) / class_weights_train[i]
                    )
                    # test dataset : increase test weight to compensate for sampling
                balanced_set["weights"] = weights_train

                # ------------------------------
                #### Fit the classifier
                # ------------------------------
                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n Start of fitting of the classifier \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                )
                self.model.fit(
                    balanced_set["data"],
                    train_labels,
                    weights_train=balanced_set["weights"],
                    train_size=NbTrain,
                    seed=random_seed,
                )
                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n End of fitting of the classifier \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                )

                del weights_train, train_labels, class_weights_train
                del balanced_set
                del train_set_poscut
                print("Train subset : Deleted ")
                print(
                    "========================================================================================"
                )

            # ///////////////////////////////////////////////////////////////////////////////
            #### Create holdout subset and compute saved info
            # ///////////////////////////////////////////////////////////////////////////////
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #### Initialise holdout set
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            print("Holdout Subset : Created")
            holdout_df = self.get_train_set(selected_indices=holdout_indices)

            holdout_set = {
                "labels": holdout_df.pop("labels"),
                "weights": holdout_df.pop("weights"),
                "detailed_labels": holdout_df.pop("detailed_labels"),
                "data": holdout_df,
            }
            del holdout_df
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ## features_systematics_dependence plot
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if Features_VS_syst:

                from feature_analysis import features_systematics_dependence

                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n Start of Features VS syst big graph \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                )
                features_systematics_dependence(
                    dfall=holdout_set,
                    systematics=self.systematics,
                    nb_bins=20,
                    var_lenght=1000,
                    n_jobs=12,
                    columns=[
                        # "PRI_lep_phi",
                        # "PRI_met",
                        # "DER_pt_ratio_lep_had",
                        # "DER_deltaeta_jet_jet",
                        "PRI_lep_pt",
                        "PRI_lep_eta",
                        "PRI_lep_phi",
                        "PRI_had_pt",
                        "PRI_had_eta",
                        "PRI_had_phi",
                        "PRI_jet_leading_pt",
                        "PRI_jet_leading_eta",
                        "PRI_jet_leading_phi",
                        "PRI_jet_subleading_pt",
                        "PRI_jet_subleading_eta",
                        "PRI_jet_subleading_phi",
                        "PRI_n_jets",
                        "PRI_jet_all_pt",
                        "PRI_met",
                        "PRI_met_phi",
                        # "weights",###########################A retirer surement
                        "DER_mass_transverse_met_lep",
                        "DER_mass_vis",
                        "DER_pt_h",
                        "DER_deltaeta_jet_jet",
                        "DER_mass_jet_jet",
                        "DER_prodeta_jet_jet",
                        "DER_deltar_had_lep",
                        "DER_pt_tot",
                        "DER_sum_pt",
                        "DER_pt_ratio_lep_had",
                        "DER_met_phi_centrality",
                        "DER_lep_eta_centrality",
                    ],
                )
                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n End of Features VS syst big graph \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                )

            # statistical_subset_info(holdout_set,"Holdout Subset Before poscut")

            # ///////////////////////////////////////////////////////////////////////////////
            ## TES JES and Soft MET fitting  , there is 2 versions : one for syst and one for all syst (when we divide bkg in its differents channel)
            # ///////////////////////////////////////////////////////////////////////////////
            from systematic_analysis import (
                regression_tes,
                regression_jes,
                regression_soft_met,
                regression_tes_3bkg,
                regression_jes_3bkg,
                regression_soft_met_3bkg,
            )

            """
            the part below is used to create the file with the 
            regression for tes, jes and soft_met
            
            Since we use a fixed random seed for the creation of 
            the train, hold and validation subset
            We can just comment it out when the file is created
            """
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ####for BNLL_all_syst   (fit on Signal and Ztautau, ttbar and Diboson + Ntot)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (Force_3bkg_regression) or (
                ("BNLL_all_syst" in method_used) and fitting_3bkg
            ):
                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n Start of fitting of TES,JES,SoftMet for 3bkg \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                )
                from systematic_analysis import (
                    regression_tes_3bkg,
                    regression_jes_3bkg,
                    regression_soft_met_3bkg,
                )

                regression_tes_3bkg(
                    holdout_set, self.model, self.systematics, nb_bins=Nb_bins_distrib
                )
                regression_jes_3bkg(
                    holdout_set, self.model, self.systematics, nb_bins=Nb_bins_distrib
                )
                regression_soft_met_3bkg(
                    holdout_set, self.model, self.systematics, nb_bins=Nb_bins_distrib
                )
                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n End of fitting of TES,JES,SoftMet for 3bkg \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ##for BNLL_syst     (fit on Signal and Background + Ntot)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (Force_1bkg_regression) or (
                ("BNLL_syst" in method_used) and fitting_1bkg
            ):
                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n Start of fitting of TES,JES,SoftMet for 1bkg \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                )
                from systematic_analysis import (
                    regression_tes,
                    regression_jes,
                    regression_soft_met,
                )

                regression_tes(
                    holdout_set, self.model, self.systematics, nb_bins=Nb_bins_distrib
                )
                regression_jes(
                    holdout_set, self.model, self.systematics, nb_bins=Nb_bins_distrib
                )
                regression_soft_met(
                    holdout_set, self.model, self.systematics, nb_bins=Nb_bins_distrib
                )
                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n End of fitting of TES,JES,SoftMet for 1bkg \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #### Apply systematics on the holdout set
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            hold_set_tamp = holdout_set.copy()
            hold_set_poscut = self.systematics(hold_set_tamp, tes=1, jes=1, soft_met=0)
            del holdout_set
            # statistical_subset_info(hold_set_poscut,"Holdout Subset After poscut")
            holdout_score = self.model.predict(hold_set_poscut["data"])

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute score and saved info
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.saved_info = calculate_saved_info(
                model=self.model,
                score=holdout_score,
                holdout_set=hold_set_poscut,
                threshold=threshold_distrib,
                nb_bins=Nb_bins_distrib,
            )

            if First_plots_hist_roc or Parabola_method != []:
                holdout_weights = hold_set_poscut["weights"]

            else:
                del holdout_score
                del hold_set_poscut
                print(
                    "Holdout subset : deleted after saved info computation (1 of the 2 ways to delete it)"
                )
                print(
                    "========================================================================================"
                )

            # ///////////////////////////////////////////////////////////////////////////////
            ## Definition of the validation set
            # ///////////////////////////////////////////////////////////////////////////////
            print("Validation Subset : Created")
            valid_df = self.get_train_set(selected_indices=valid_indices)
            valid_set = {
                "labels": valid_df.pop("labels"),
                "weights": valid_df.pop("weights"),
                "detailed_labels": valid_df.pop("detailed_labels"),
                "data": valid_df,
            }
            del valid_df
            # statistical_subset_info(valid_set,"Validation Subset Before poscut")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #### Apply systematics on the validation set
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            valid_set_tamp = valid_set.copy()
            valid_set_poscut = self.systematics(
                valid_set_tamp, tes=1, jes=1, soft_met=0
            )
            del valid_set
            # statistical_subset_info(valid_set_poscut,"Validation Subset After poscut")

            if First_plots_hist_roc or Parabola_method != [] or Bins_varia_plot:
                valid_weights = valid_set_poscut["weights"]
                valid_score = self.model.predict(valid_set_poscut["data"])

                # ///////////////////////////////////////////////////////////////////////////////
                # Somes plots
                # ///////////////////////////////////////////////////////////////////////////////

                if First_plots_hist_roc:  # Be cautious it will maybe not run
                    holdout_results = compute_mu(
                        saved_info_hold=self.saved_info,
                        score_test=holdout_score,
                        weight_test=holdout_weights,
                    )

                    valid_results = compute_mu(
                        saved_info_hold=self.saved_info,
                        score_test=valid_score,
                        weight_test=valid_weights,
                    )

                    print("Holdout Results: ")
                    for key in holdout_results.keys():
                        print("\t", key, " : ", holdout_results[key])

                    print("Valid Results: ")
                    for key in valid_results.keys():
                        print("\t", key, " : ", valid_results[key])

                    print("saved info", self.saved_info)

                    valid_set_poscut["data"]["score"] = valid_score
                    from utils import roc_curve_wrapper, histogram_dataset

                    print("saved info", self.saved_info)

                    histogram_dataset(
                        valid_set_poscut["data"],
                        valid_set_poscut["labels"],
                        valid_set_poscut["weights"],
                        columns=["score"],
                    )

                    # from HiggsML.visualization import stacked_histogram  #Problem

                    # stacked_histogram(
                    #      self.valid_set["data"],
                    #      self.valid_set["labels"],
                    #     self.valid_set["weights"],
                    #     self.valid_set["detailed_labels"],
                    #      "score",
                    #  )

                    roc_curve_wrapper(
                        score=valid_score,
                        labels=valid_set_poscut["labels"],
                        weights=valid_set_poscut["weights"],
                        plot_label="valid_set" + self.name,
                    )
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # End of the 1st plots
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                if Parabola_method != [] or Bins_varia_plot:
                    score_test = [holdout_score, valid_score]
                    data_set_test_poscut = [hold_set_poscut, valid_set_poscut]
                    data_set_name = [" Holdout", " Validation"]

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Parabola curve
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if Parabola_method != []:
                        from Function_analysis import Parabola_Likelihood_plot

                        for i in range(2):
                            print("~~~~~~~~~\n Parabola plot for", data_set_name[i])

                            Parabola_Likelihood_plot(
                                saved_info_hold=self.saved_info,
                                score_test=score_test[i],
                                weight_test=data_set_test_poscut[i]["weights"],
                                nb_bins=Nb_bins_distrib,
                                threshold=threshold_distrib,
                                Methode_Mu_Compar=Parabola_method,  #  "UNLL", "BNLL", "Direct", "BNLL_syst","BNLL_syst_normal_bkg", "BNLL_all_syst"
                                mu_init=1.0,
                            )

                    if Bins_varia_plot:
                        from Function_analysis import Bins_BNLL_varia

                        for i in range(2):
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Numbers of bins VS Result
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            print("~~~~~~~~~\n Parabola plot for", data_set_name[i])
                            Bins_BNLL_varia(
                                saved_info_hold=self.saved_info,
                                score_test=score_test[i],
                                weight_test=data_set_test_poscut[i]["weights"],
                                bin_min=Bins_varia_Min_Max_Step[0],
                                bin_max=Bins_varia_Min_Max_Step[1],
                                value_bin_step=Bins_varia_Min_Max_Step[2],
                                mu_init=1.0,
                                threshold=0,
                            )

            else:
                del valid_set_poscut
                print(
                    "Validation subset : deleted after saved info computation (1 of the 2 ways to delete it)"
                )
                print(
                    "========================================================================================"
                )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Best result
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if Compute_Best_Opti:
            best_opti = calculate_best_threshold(
                score_valid_test=valid_score,
                valid_test_set=valid_set_poscut,
                score_hold_exp=holdout_score,
                holdout_exp_set=hold_set_poscut,
                NbPoints_Prec_Thresh=NbPoints_Prec_Thresh,
                del_mu_method="Direct",
                Plot=False,
            )
            self.best_opti = best_opti
            print(self.best_opti)
            return self.best_opti

        print("###########################################")
        print("End of the fitting of the model")
        print("###########################################")

    ###############################################################
    ######## Predict
    ###############################################################
    def predict(self, test_set):
        print("###########################################")
        print("Beginning of the prediction")
        print("###########################################")

        # ///////////////////////////////////////////////////////////////////////////////
        # dataset creation + variable (saved info / score)
        # ///////////////////////////////////////////////////////////////////////////////

        # import copy
        # hold_set_tamp=copy.deepcopy(self.holdout_set)
        # hold_set_poscut=self.systematics(hold_set_tamp,tes=1,jes=1,soft_met=0)
        # holdout_data=hold_set_poscut["data"]
        # holdout_score=self.model.predict(holdout_data)

        test_data = test_set["data"]
        test_weights = test_set["weights"]
        predictions = self.model.predict(test_data)

        result_mu_cal = compute_mu(
            saved_info_hold=self.saved_info,
            score_test=predictions,
            weight_test=test_weights,
            mu_init=1.0,
            method=Predict_method,
            threshold=threshold_distrib,
            nb_bins=Nb_bins_distrib,
        )

        if Print_NLL_score_biased_stacked:
            saved_biaised_info = calculate_saved_info(
                model=self.model,
                score=predictions,
                holdout_set=self.systematics(test_set, tes=0, jes=0, soft_met=1),
                threshold=threshold_distrib,
                nb_bins=Nb_bins_distrib,
            )

            def Bias_My_Set(
                saved_info_hold,
                nb_bins,
                mu,
                tes,
                jes,
                soft_met,
                bkg_scale,
                ttbar_scale,
                diboson_scale,
            ):
                import os

                def BNLL_all_syst_mu_fitting_param(nb_bins):
                    # # Load
                    FitingData = [None] * 3
                    SystName = ["TES", "JES", "SOFTMET"]

                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    for i in range(3):  #######ATTENTION BESOIN DE METTRE A 3 for all
                        if os.path.isfile(
                            "%s/Fitting_Parameters/bkg_subchannel/%s_3bkg_%sbins_2orderFittingParam.npz"
                            % (current_dir, SystName[i], nb_bins)
                        ):
                            FitingData[i] = np.load(
                                "%s/Fitting_Parameters/bkg_subchannel/%s_3bkg_%sbins_2orderFittingParam.npz"
                                % (current_dir, SystName[i], nb_bins)
                            )
                        else:
                            print("A file is missing \n ~~~~ \n ~~~ \n ~~ \n ~")

                    return FitingData

                FitingData = BNLL_all_syst_mu_fitting_param(nb_bins=nb_bins)

                # ///////////////////////////////////////////////////////////////////////////////
                ## Cost_nll_all_syst
                # ///////////////////////////////////////////////////////////////////////////////
                from systematic_analysis import (
                    Polynomial_Reg_Model_forced_jes_tes,
                    Polynomial_Reg_Model_forced_soft_met,
                )

                sig_exp_hold_biaised = (
                    saved_info_hold[
                        "signal_hist"
                    ]  # saved_info_hold["signal_hist"] = sig_exp_hold_unbiaised
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
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                ##First we define the tes, jes and soft met syst and then the scaling for ztautau,ttbar and diboson
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                bkg_scale_ztautau_exp_hold_biaised = (1 + bkg_scale) * (
                    saved_info_hold[
                        "ztautau_hist"
                    ]  # saved_info_hold["ztautau_hist"] = ztautau_hist_exp_hold_unbiaised
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
                    (1 + bkg_scale)
                    * (1 + ttbar_scale)
                    * (
                        saved_info_hold[
                            "ttbar_hist"
                        ]  # saved_info_hold["ttbar_hist"] = ttbar_hist_exp_hold_unbiaised
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
                )

                bkg_diboson_exp_hold_biaised = (
                    (1 + bkg_scale)
                    * (1 + diboson_scale)
                    * (
                        saved_info_hold[
                            "diboson_hist"
                        ]  # saved_info_hold["diboson_hist"] = diboson_hist_exp_hold_unbiaised
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
                )

                return (
                    sig_exp_hold_biaised,
                    bkg_scale_ztautau_exp_hold_biaised,
                    bkg_ttbar_exp_hold_exp_hold_biaised,
                    bkg_diboson_exp_hold_biaised,
                )

            (
                sig_exp_hold_biaised,
                bkg_scale_ztautau_exp_hold_biaised,
                bkg_ttbar_exp_hold_exp_hold_biaised,
                bkg_diboson_exp_hold_biaised,
            ) = Bias_My_Set(
                saved_info_hold=self.saved_info,
                nb_bins=Nb_bins_distrib,
                mu=result_mu_cal["mu_hat"],
                tes=result_mu_cal["tes"],
                jes=result_mu_cal["jes"],
                soft_met=result_mu_cal["soft_met"],
                bkg_scale=result_mu_cal["bkg_scale"],
                ttbar_scale=result_mu_cal["ttbar_scale"],
                diboson_scale=result_mu_cal["diboson_scale"],
            )
            Bins_edges = self.saved_info["Bins_edges"]
            plt.plot(1, 1, figsize=(8, 6))
            plt.hist(
                bkg_scale_ztautau_exp_hold_biaised,
                bins=Bins_edges,
                alpha=0.4,
                color="blue",
                label="Predicted Ztautau",
            )
            plt.hist(
                bkg_ttbar_exp_hold_exp_hold_biaised,
                bins=Bins_edges,
                alpha=0.4,
                color="orange",
                label="Predicted ttbar",
            )
            plt.hist(
                bkg_diboson_exp_hold_biaised,
                bins=Bins_edges,
                alpha=0.4,
                color="green",
                label="Predicted diboson",
            )
            plt.hist(
                sig_exp_hold_biaised,
                bins=Bins_edges,
                alpha=0.4,
                color="red",
                label="Predicted Htautau",
            )

            half_bins = np.linspace(0, 1, Nb_bins_distrib)

            plt.scatter(
                half_bins,
                saved_biaised_info["signal_hist"] + saved_biaised_info["bkg_hist"],
                label="Ground Truth",
            )
            plt.xlabel("BDT Score")
            plt.ylabel("Weighted number of events")
            plt.title(
                "%s_%s_bins_%s Scores" % (ModelType, Nb_bins_distrib, Predict_method)
            )
            plt.legend()
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists("%s/Images/ScoresVSBDT_biased" % (current_dir)):
                os.makedirs("%s/Images/ScoresVSBDT_biased" % (current_dir))
            plt.savefig(
                "%s/Images/ScoresVSBDT_biased/%s_%s_bins_%s Scores"
                % (current_dir, ModelType, Nb_bins_distrib, Predict_method)
            )
            plt.close()

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result
