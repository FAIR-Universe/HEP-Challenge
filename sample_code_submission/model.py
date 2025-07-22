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
Nb_bins_distrib=Tamp_parameter["Nb_bins_distrib"]
threshold_distrib=Tamp_parameter["threshold_distrib"]

random_seed=Tamp_parameter["random_seed"]

Force_3bkg_regression =Tamp_parameter["Force_3bkg_regression"]
Force_1bkg_regression =Tamp_parameter["Force_1bkg_regression"]
fitting_3bkg =Tamp_parameter["fitting_3bkg"]
fitting_1bkg =Tamp_parameter["fitting_1bkg"]

Predict_method= Tamp_parameter["Predict_method"][0]
Parabola_method= Tamp_parameter["Parabola_method"]
method_used= list(set(Tamp_parameter["Predict_method"]+Parabola_method))

First_plots_hist_roc=Tamp_parameter["First_plots_hist_roc"]
Bins_varia_plot = Tamp_parameter["Bins_varia_plot"]
if Bins_varia_plot == True:
    Bins_varia_Min_Max_Step=Tamp_parameter["Bins_varia_Min_Max_Step"]
Compute_Best_Opti = Tamp_parameter["Compute_Best_Opti"]
if Compute_Best_Opti == True:
    NbPoints_Prec_Thresh = Tamp_parameter["NbPoints_Prec_Thresh"]


NbTrain = THV_size[0]
NbHoldout = THV_size[1]
NbValidation = THV_size[2]


""
    #####################        class Model
""
class Model:

###############################################################
###############################################################
######## Init
###############################################################
    def __init__(self, get_train_set=None, systematics=None, model_type="sample_model"):

        """
        Define the THV subset : 
        First the indices and the list of events associated
        Then the panda table associated
        """
        print("///////////////////////////////////////////////////////////////////////////////")
        print(f"Random seed used: {random_seed}")
        print("///////////////////////////////////////////////////////////////////////////////")

        indices = np.arange(THV_size.sum())
        
        np.random.seed(random_seed)    
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

    def fit(self):
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
        
        """
        the part below is used to create the file with the 
        regression for tes, jes and soft_met
        
        Since we use a fixed random seed for the creation of 
        the train, hold and validation subset
        We can just comment it out when the file is created
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ####for BNLL_all_syst   (fit on Signal and Ztautau, ttbar and Diboson + Ntot)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if ( Force_3bkg_regression == True or (method_used == "BNLL_all_syst" and fitting_3bkg ==True) ) :
            from systematic_analysis import regression_tes_3bkg,regression_jes_3bkg,regression_soft_met_3bkg
            regression_tes_3bkg( self.holdout_set, self.model ,self.systematics, nb_bins=Nb_bins_distrib)
            regression_jes_3bkg( self.holdout_set, self.model ,self.systematics,nb_bins=Nb_bins_distrib)
            regression_soft_met_3bkg( self.holdout_set, self.model ,self.systematics,nb_bins=Nb_bins_distrib)
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ##for BNLL_syst     (fit on Signal and Background + Ntot)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if ( Force_1bkg_regression == True or (method_used == "BNLL_syst" and fitting_1bkg ==True) ) :
            from systematic_analysis import regression_tes, regression_jes,regression_soft_met
            regression_tes ( self.holdout_set, self.model ,self.systematics,nb_bins=Nb_bins_distrib)  
            regression_jes( self.holdout_set, self.model ,self.systematics,nb_bins=Nb_bins_distrib)
            regression_soft_met ( self.holdout_set, self.model,self.systematics,nb_bins=Nb_bins_distrib )
        
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Applying poscut thanks to unbiaised systematics  + new subset information
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        import copy
        train_set_tamp=copy.deepcopy(self.training_set)
        train_set_poscut=self.systematics(train_set_tamp,tes=1,jes=1,soft_met=0)
        hold_set_tamp=copy.deepcopy(self.holdout_set)
        self.hold_set_poscut=self.systematics(hold_set_tamp,tes=1,jes=1,soft_met=0)
        valid_set_tamp=copy.deepcopy(self.valid_set)
        valid_set_poscut=self.systematics(valid_set_tamp,tes=1,jes=1,soft_met=0)

        
        from utils import statistical_subset_info
        statistical_subset_info(train_set_poscut,"train fit poscut")
        statistical_subset_info(self.hold_set_poscut,"holdout fit poscut")
        statistical_subset_info(valid_set_poscut,"valid fit poscut")
        
       
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute score and saved info
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.saved_info = calculate_saved_info(model=self.model,
             score=self.model.predict(self.holdout_set["data"]),
             holdout_set=self.holdout_set,
             threshold=threshold_distrib,
             nb_bins=Nb_bins_distrib,)
        
        train_score = self.model.predict(train_set_poscut["data"])

        self.holdout_score = self.model.predict(self.hold_set_poscut["data"])

        valid_score = self.model.predict(valid_set_poscut["data"])
      
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Somes plots
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        from Function_analysis import Parabola_Likelihood_plot

        if First_plots_hist_roc == "True":   #Be cautious it will maybe not run
            train_results = compute_mu(
                saved_info_hold=self.saved_info,
                weight_test=self.valid_set["weights"],
                score_test=train_score, )

            holdout_results = compute_mu(
                saved_info_hold=self.saved_info,
                score_test= self.holdout_score, 
                weight_test=self.holdout_set["weights"],)

            valid_results = compute_mu(
                saved_info_hold=self.saved_info,
                score_test= valid_score, 
                weight_test=self.valid_set["weights"],)


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

            
            self.valid_set["data"]["score"] = valid_score
            from utils import roc_curve_wrapper, histogram_dataset
            
            print("saved info", self.saved_info)
            
            histogram_dataset(
                 self.valid_set["data"],
                 self.valid_set["labels"],
                 self.valid_set["weights"],
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
                 labels=self.valid_set["labels"],
                 weights=self.valid_set["weights"],
                 plot_label="valid_set" + self.name,
             )
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # End of the plots
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            

        score_test=[train_score,self.holdout_score,valid_score]
        data_set_test_poscut=[train_set_poscut,self.hold_set_poscut,valid_set_poscut]
        data_set_name=[" Train"," Holdout"," Validation"]
 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Parabola curve
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if Parabola_method != [] :
            for i in range (3):
                print("~~~~~~~~~\n Parabola plot for",data_set_name[i])
                
                Parabola_Likelihood_plot(
                    saved_info_hold=self.saved_info,

                    score_test=score_test[i],
                    weight_test= data_set_test_poscut[i] ["weights"],

                    nb_bins=Nb_bins_distrib,
                    threshold=threshold_distrib,
                    Methode_Mu_Compar=Parabola_method,   #  "UNLL", "BNLL", "Direct", "BNLL_syst","BNLL_syst_normal_bkg", "BNLL_all_syst"
                    mu_init=1.0,
                )
        
        
        if Bins_varia_plot == True :
            from Function_analysis import Bins_BNLL_varia
            for i in range (3):
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Numbers of bins VS Result
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                print("~~~~~~~~~\n Parabola plot for",data_set_name[i])
                Bins_BNLL_varia(
                    saved_info_hold=self.saved_info,

                    score_test=score_test[i],
                    weight_test= data_set_test_poscut[i] ["weights"],

                    bin_min=Bins_varia_Min_Max_Step[0],
                    bin_max=Bins_varia_Min_Max_Step[1],
                    value_bin_step=Bins_varia_Min_Max_Step[2],

                    mu_init=1.0,
                    threshold=0,
                )

        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Best result 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if Compute_Best_Opti == True :
            best_opti = calculate_best_threshold(
                score_valid_test=valid_score,
                valid_test_set= valid_set_poscut ,

                score_hold_exp=self.holdout_score,
                holdout_exp_set= self.hold_set_poscut ,

                NbPoints_Prec_Thresh=NbPoints_Prec_Thresh,
                del_mu_method="Direct",
                Plot=False,
            )
            self.best_opti = best_opti
            print(self.best_opti)
            return self.best_opti
        


    

###############################################################
###############################################################
######## Predict
###############################################################
    def predict(self, test_set):

        from statistical_analysis import calculate_saved_info

        #/////////////////////////////////////////////////////////////////////////////// 
        # dataset creation + variable (saved info / score)
        #/////////////////////////////////////////////////////////////////////////////// 

        #import copy
        #hold_set_tamp=copy.deepcopy(self.holdout_set)
        #hold_set_poscut=self.systematics(hold_set_tamp,tes=1,jes=1,soft_met=0)
        #holdout_data=hold_set_poscut["data"]
        #holdout_score=self.model.predict(holdout_data)
        
        test_data = test_set["data"]
        test_weights = test_set["weights"]
        predictions = self.model.predict(test_data)
        
            
        result_mu_cal = compute_mu(saved_info_hold=self.saved_info,
                                   
                                   score_test=predictions, 
                                   weight_test=test_weights,

                                   mu_init=1.0,
                                   method= Predict_method,
                                   threshold=threshold_distrib,
                                   nb_bins=Nb_bins_distrib)
        
        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result
