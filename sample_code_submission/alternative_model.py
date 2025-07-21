# ------------------------------
# Dummy Sample Submission
# ------------------------------

###############################################################
###############################################################
######## BNLL_all_syst METHOD
###############################################################

    #///////////////////////////////////////////////////////////////////////////////
    ## Cost_nll_all_syst
    #/////////////////////////////////////////////////////////////////////////////// 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ##First we define the tes, jes and soft met syst and then the scaling for ztautau,ttbar and diboson
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
THV_size = Tamp_parameter["THV_size"]
ModelType = Tamp_parameter["ModelType"]

NbTrain = THV_size[0]
NbHoldout = THV_size[1]
NbValidation = THV_size[2]

Nb_bins_distrib=20



""
    #####################        class Model
""
class Model:

###############################################################
###############################################################
######## Init
###############################################################
    def __init__(self, get_train_set=None, systematics=None, model_type="sample_model"):

    #///////////////////////////////////////////////////////////////////////////////
    ## Define the THV subset : 
    ## First the indices and the list of events associated
    ## Then the panda table associated
    #/////////////////////////////////////////////////////////////////////////////// 
        #import time
        #seed = int(time.time_ns() % (2**32))  # or use os.urandom() if needed
        #print(f"Random seed used: {seed}")

        indices = np.arange(THV_size.sum())
        
        np.random.seed(10912983)    #10912983
        np.random.shuffle(indices)
        
        train_indices = indices[: THV_size[0]]
        holdout_indices = indices[THV_size[0] : THV_size[0] + THV_size[1]]
        valid_indices = indices[THV_size[0] + THV_size[1] :]

        self.systematics = systematics

        training_df = get_train_set(selected_indices=train_indices)
        self.training_set = {
            "labels": training_df.pop("labels"),
            "weights": training_df.pop("weights"),
            "detailed_labels": training_df.pop("detailed_labels"),
            "data": training_df,
        }
        del training_df


        valid_df = get_train_set(selected_indices=valid_indices)
        self.valid_set = {
            "labels": valid_df.pop("labels"),
            "weights": valid_df.pop("weights"),
            "detailed_labels": valid_df.pop("detailed_labels"),
            "data": valid_df,
        }
        del valid_df

        holdout_df = get_train_set(selected_indices=holdout_indices)
        self.holdout_set = {
            "labels": holdout_df.pop("labels"),
            "weights": holdout_df.pop("weights"),
            "detailed_labels": holdout_df.pop("detailed_labels"),
            "data": holdout_df,
        }
        del holdout_df

        
    #///////////////////////////////////////////////////////////////////////////////
    ## Subset information data (we will print later the poscut subset information)
    #/////////////////////////////////////////////////////////////////////////////// 
        from utils import statistical_subset_info
        print("We have just created the subset")
        statistical_subset_info(self.training_set,"train init")
        statistical_subset_info(self.holdout_set,"holdout init")
        statistical_subset_info(self.valid_set,"valid init")

        
    #///////////////////////////////////////////////////////////////////////////////
    ## computation methode BDT / NN / other
    #/////////////////////////////////////////////////////////////////////////////// 
        if model_type == "BDT":
            from boosted_decision_tree import BoostedDecisionTree
            self.model = BoostedDecisionTree(train_data=self.training_set["data"])

        elif model_type == "NN":
            from neural_network import NeuralNetwork
            self.model = NeuralNetwork(train_data=self.training_set["data"])

        elif model_type == "sample_model":
            from sample_model import SampleModel
            self.model = SampleModel()

        else:
            print(f"model_type {model_type} not found")
            raise ValueError(f"model_type {model_type} not found")
        
        self.name = model_type
        print(f" Model is { self.name}")


    
###############################################################
###############################################################
######## FIT
###############################################################

    def fit(self, Add_info_plot=False):
     
        Add_info_plot = "True"
        NbPoints_Prec_Thresh = 50  # NbPoints_Precision_Threshold  #75

        
    #///////////////////////////////////////////////////////////////////////////////
        ## Normalisation Weight
        ####
        ################################
        ## TO CHECK #########################################################################
        ################################""
        ##
    #/////////////////////////////////////////////////////////////////////////////// 
        balanced_set = self.training_set.copy()
        weights_train = self.training_set["weights"].copy()
        train_labels = self.training_set["labels"].copy()
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

        self.model.fit(
            balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
        )

        
    #///////////////////////////////////////////////////////////////////////////////
    ## TES JES and Soft MET fitting  , there is 2 versions : one for syst and one for all syst (when we divide bkg in its differents channel)
    #/////////////////////////////////////////////////////////////////////////////// 
        from systematic_analysis import regression_tes, regression_jes,regression_soft_met,regression_tes_3bkg,regression_jes_3bkg,regression_soft_met_3bkg
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #####################       the part below is used to create the file with the 
        #####################       regression for tes, jes and soft_met
        #####################       
        #####################       Since we use a fixed random seed for the creation of 
        #####################       the train, hold and validation subset
        #####################       We can just comment it out when the file is created
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ####for BNLL_all_syst   (fit on Signal and Ztautau, ttbar and Diboson + Ntot)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
        regression_tes_3bkg( self.holdout_set, self.model ,nb_bins=Nb_bins_distrib)
        regression_jes_3bkg( self.holdout_set, self.model ,nb_bins=Nb_bins_distrib)
        regression_soft_met_3bkg( self.holdout_set, self.model ,nb_bins=Nb_bins_distrib)
        """

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ##for BNLL_syst     (fit on Signal and Background + Ntot)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
        regression_tes ( self.holdout_set, self.model ,nb_bins=Nb_bins_distrib)  
        regression_jes( self.holdout_set, self.model ,nb_bins=Nb_bins_distrib)
        regression_soft_met ( self.holdout_set, self.model,nb_bins=Nb_bins_distrib )
        """
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Applying poscut thanks to unbiaised systematics  + new subset information
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        import copy
        train_set_tamp=copy.deepcopy(self.training_set)
        train_set_poscut=self.systematics(train_set_tamp,tes=1,jes=1,soft_met=0)
        hold_set_tamp=copy.deepcopy(self.holdout_set)
        hold_set_poscut=self.systematics(hold_set_tamp,tes=1,jes=1,soft_met=0)
        valid_set_tamp=copy.deepcopy(self.valid_set)
        valid_set_poscut=self.systematics(valid_set_tamp,tes=1,jes=1,soft_met=0)

        
        from utils import statistical_subset_info
        statistical_subset_info(train_set_poscut,"train fit poscut")
        statistical_subset_info(hold_set_poscut,"holdout fit poscut")
        statistical_subset_info(valid_set_poscut,"valid fit poscut")

        
        
       
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute score and saved info
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.saved_info = calculate_saved_info(model=self.model,
             score=self.model.predict(self.holdout_set["data"]),
             holdout_set=self.holdout_set,)
        train_score = self.model.predict(train_set_poscut["data"])
        """
        train_results = compute_mu(
              score_test=train_score, 
            #self.training_set["weights"], self.saved_info
             saved_info_hold=self.saved_info,
             weight_test=self.valid_set["weights"],)
        """
        #print("hold_set_poscute[weight] ",hold_set_poscut["weights"].shape)
        holdout_score = self.model.predict(hold_set_poscut["data"])

        """
        holdout_results = compute_mu(
               score_test= holdout_score, 
            #self.holdout_set["weights"], self.saved_info
             saved_info_hold=self.saved_info,
             weight_test=self.holdout_set["weights"],)
        """
        #print("valid_set_poscute[weight] ",valid_set_poscut["weights"].shape)
        valid_score = self.model.predict(valid_set_poscut["data"])
        #print("valid_score ",valid_score.shape )
        """
        valid_results = compute_mu(
            score_test= valid_score, 
            #self.valid_set["weights"], self.saved_info
             saved_info_hold=self.saved_info,
             weight_test=self.valid_set["weights"],)
        """
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Somes plots
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        from Function_analysis import Parabola_Likelihood_plot
        Add_info_plot="True"
        if Add_info_plot == "True":
            """
            print("Train Results: ")
            for key in train_results.keys():
                print("\t", key, " : ", train_results[key])

            print("Holdout Results: ")
            for key in holdout_results.keys():
                print("\t", key, " : ", holdout_results[key])

            print("Valid Results: ")
            for key in valid_results.keys():
                print("\t", key, " : ", valid_results[key])

            print("saved info", self.saved_info)
            """
            """
            self.valid_set["data"]["score"] = valid_score
            from utils import roc_curve_wrapper, histogram_dataset
            
            print("saved info", self.saved_info)
            
            histogram_dataset(
                 self.valid_set["data"],
                 self.valid_set["labels"],
                 self.valid_set["weights"],
                 columns=["score"],
             )

            from HiggsML.visualization import stacked_histogram

            stacked_histogram(
                 self.valid_set["data"],
                 self.valid_set["labels"],
                self.valid_set["weights"],
                self.valid_set["detailed_labels"],
                 "score",
             )

            roc_curve_wrapper(
                 score=valid_score,
                 labels=self.valid_set["labels"],
                 weights=self.valid_set["weights"],
                 plot_label="valid_set" + self.name,
             )

            """
           

        #print("~~~~~~~~\n Saved_info pour hold poscut set")
        saved_info_tamp_hold = calculate_saved_info(
            holdout_score, hold_set_poscut , threshold=0,model=self.model
        )
        #print("~~~~~~~~\n Saved_info pour valid poscut set")
        saved_info_tamp_valid = calculate_saved_info(
            valid_score, valid_set_poscut , threshold=0,model=self.model
        )
        #print("~~~~~~~~\n Saved_info pour train poscut set")
        saved_info_tamp_train = calculate_saved_info(
            train_score, train_set_poscut , threshold=0,model=self.model
        )

        
        score_test=[train_score,holdout_score,valid_score]
        data_set_test_poscut=[train_set_poscut,hold_set_poscut,valid_set_poscut]
        data_set_name=[" Train"," Holdout"," Validation"]
 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Parabola curve
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i in range (3):
            print("~~~~~~~~~\n Parabola plot for",data_set_name[i])
            
            Parabola_Likelihood_plot(
                nb_bins=Nb_bins_distrib,
                threshold=0,
                saved_info_hold=saved_info_tamp_hold,
                score_test=score_test[i],
                weight_test= data_set_test_poscut[i] ["weights"],
                score_exp_hold=holdout_score,
                weight_exp_hold= hold_set_poscut ["weights"],
                label_exp_hold= hold_set_poscut ["labels"],
                detailed_labels_exp_hold=hold_set_poscut["detailed_labels"],
                Methode_Mu_Compar=["UNLL", "BNLL"],   #  "UNLL", "BNLL", "Direct", "BNLL_syst","BNLL_syst_normal_bkg", "BNLL_all_syst"
                mu_init=1.0,
            )
        
        
        """
        from Function_analysis import Bins_BNLL_varia
        for i in range (3):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Numbers of bins VS Result
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            print("~~~~~~~~~\n Parabola plot for",data_set_name[i])
            Bins_BNLL_varia(
            score_test=score_test[i],
            weight_test= data_set_test_poscut[i] ["weights"],
            score_exp_hold=holdout_score,
            label_exp_hold= hold_set_poscut ["labels"],
            weight_exp_hold= hold_set_poscut ["weights"],
            bin_min=1,
            bin_max=51,
            value_bin_step=3,
            )

        """
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Best result 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        best_opti = calculate_best_threshold(
            score_valid_test=valid_score,
            valid_test_set= valid_set_poscut ,
            score_hold_exp=holdout_score,
            holdout_exp_set= hold_set_poscut ,
            NbPoints_Prec_Thresh=NbPoints_Prec_Thresh,
            del_mu_method="Direct",
            Plot=False,
        )
        self.best_opti = best_opti
        print(self.best_opti)
        return self.best_opti
        """


    

###############################################################
###############################################################
######## Predict
###############################################################
    def predict(self, test_set):

        from statistical_analysis import calculate_saved_info

        #/////////////////////////////////////////////////////////////////////////////// 
        # Parameter
        #/////////////////////////////////////////////////////////////////////////////// 
        threshold=0
        mu_init=1

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #####################        The method define the model used
        #####################  "BNLL_all_syst" = 5 syst (all except soft_met which is commented out)
        #####################  "BNLL_syst_normal_bkg"  = bkg_scale, ttbar_scale and diboson_scale but without tes or jes
        #####################  "BNLL_syst" = tes and jes (soft_met is commented out)
        #####################  "BNLL" = only binned 
        method="BNLL_all_syst"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #/////////////////////////////////////////////////////////////////////////////// 
        # dataset creation + variable (saved info / score)
        #/////////////////////////////////////////////////////////////////////////////// 
        test_data = test_set["data"]
        test_weights = test_set["weights"]
    
        import copy
        hold_set_tamp=copy.deepcopy(self.holdout_set)
        hold_set_poscut=self.systematics(hold_set_tamp,tes=1,jes=1,soft_met=0)
        holdout_data=hold_set_poscut["data"]
        
        predictions = self.model.predict(test_data)
        holdout_score=self.model.predict(holdout_data)

        saved_info_hold=calculate_saved_info(
                    holdout_score, hold_set_poscut, threshold,model=self.model
                    )

        #test_weights=np.ones(len(test_weights))*(saved_info_hold["N"])/len(hold_set_poscut["weights"])
            
        result_mu_cal = compute_mu(saved_info_hold=saved_info_hold,
                                   score_test=predictions, 
                                   weight_test=test_weights,
                                   score_exp_hold=holdout_score,
                                   weight_exp_hold=hold_set_poscut["weights"],    #CAUTION ABOUT THE ROI (threshold)
                                   label_exp_hold=hold_set_poscut["labels"],
                                   detailed_labels_exp_hold=hold_set_poscut["detailed_labels"],
                                   mu_init=mu_init,
                                   method=method,
                                   threshold=threshold,
                                   nb_bins=Nb_bins_distrib)
        """
        #print("result_mu_cal[mu_hat]: ", result_mu_cal["mu_hat"])
        from Function_analysis import Parabola_Likelihood_plot


        Parabola_Likelihood_plot(
            nb_bins=Nb_bins_distrib,
            threshold=threshold,
            saved_info_hold=saved_info_hold,
            score_test=predictions,
            weight_test= test_weights,
            score_exp_hold=holdout_score,
            weight_exp_hold= hold_set_poscut["weights"],
            label_exp_hold= hold_set_poscut ["labels"],
            detailed_labels_exp_hold=hold_set_poscut["detailed_labels"],
            Methode_Mu_Compar=["BNLL_syst", "BNLL_all_syst"],   #  "UNLL", "BNLL", "Direct", "BNLL_syst","BNLL_syst_normal_bkg", "BNLL_all_syst"
            mu_init=mu_init,

        )
        """
        
        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result
