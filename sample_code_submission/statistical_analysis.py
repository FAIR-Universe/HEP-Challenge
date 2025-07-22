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

Dont_compute_tes = Tamp_parameter["Dont_compute_tes"]
Dont_compute_jes = Tamp_parameter["Dont_compute_jes"]
Dont_compute_soft_met = Tamp_parameter["Dont_compute_soft_met"]


def compute_mu(
    saved_info_hold,

    weight_test,
    score_test,

    method="Direct",
    threshold=0,
    nb_bins=10,
    mu_init=1.0,

    score_exp_hold=0,
    weight_exp_hold=0,
    label_exp_hold=0,
    detailed_labels_exp_hold=0
):  

    if saved_info_hold["threshold"]!=threshold or saved_info_hold["nb_bins"]!=nb_bins:
        if saved_info_hold["threshold"]!=threshold :
            print ("/////////////////////////////////////////")
            print("THERE IS A PROBLEM WITH SAVED INFO THRESHOLD")
            print("////////////////////////////////////////////")
        if saved_info_hold["nb_bins"]!=nb_bins :
            print ("/////////////////////////////////////////")
            print("THERE IS A PROBLEM WITH SAVED INFO NB OF BINS")
            print("////////////////////////////////////////////")
            
    tes_init_ref=1
    jes_init_ref=1
    soft_met_init_ref=0

    bkg_scale_init_ref=0
    ttbar_scale_init_ref=0
    diboson_scale_init_ref=0

    Bins_edges = np.linspace(0, 1, nb_bins + 1)

    #print ("signal :",saved_info_hold["signal"]," bkg: ",saved_info_hold["bkg"], " n_obs: ", np.sum(weight_test[score_test>threshold]) )

    def Model(mu, sig, bkg):
        return mu * sig + bkg
    
    def n_obs () :
        return np.sum(weight_test[score_test>threshold])
    
    def n_obs_hist():
        score_ROIscore_test = score_test[score_test > threshold]
        weight_ROIscore_test = weight_test[score_test > threshold]
        return  np.histogram(score_ROIscore_test, bins=Bins_edges, weights=weight_ROIscore_test)[0]


    mu, del_mu_stat, del_mu_tot, del_mu_sys = (0, 0, 0, 0)

    ###############################################################
    ###############################################################
    ######## DIRECT METHOD
    ###############################################################
    if method == "Direct":  # Based on N=mu*S+b

        def counting_mu(score, weight, saved_info):
            mu = (n_obs() - saved_info["bkg"]) / saved_info["signal"]
            del_mu_stat = (
                np.sqrt(saved_info["bkg"] + saved_info["signal"]) / saved_info["signal"]
            )
            return mu, del_mu_stat
        

        mu, del_mu_stat = counting_mu(score_test, weight_test, saved_info_hold)
        
        del_mu_stat_inf=-del_mu_stat
        del_mu_stat_sup=del_mu_stat

        tes=tes_init_ref
        jes=jes_init_ref
        soft_met=soft_met_init_ref

    ###############################################################
    ###############################################################
    ######## UNLL METHOD
    ###############################################################    
    elif method == "UNLL":
    
        # def UNLL_mu_parameter(saved_info_hold,weight_test,score_test,threshold) :
        #     sig_exp_holdout_unll = saved_info_hold["signal"]
        #     bkg_exp_holdout_unll = saved_info_hold["bkg"]
        #     n_obs_test_unll = np.sum(weight_test[score_test>threshold])

        #     return n_obs_test_unll,sig_exp_holdout_unll,bkg_exp_holdout_unll

        # n_obs_test_unll, sig_exp_holdout_unll, bkg_exp_holdout_unll = UNLL_mu_parameter(saved_info_hold=saved_info_hold,weight_test=weight_test,score_test=score_test,threshold=threshold)


        #///////////////////////////////////////////////////////////////////////////////
        ## Cost_unll
        #///////////////////////////////////////////////////////////////////////////////   
        def Cost_unll(mu):
            N_exp_unll = Model(
            mu=mu, sig=saved_info_hold["signal"], bkg=saved_info_hold["bkg"])  
      
            return -(-N_exp_unll + n_obs() * np.log(N_exp_unll) )
    
        def UNLL_mu_computation (mu_init) :
            m_unll = Minuit(Cost_unll, mu=mu_init)
            m_unll.limits["mu"] = (0, 4)
            m_unll.errordef = Minuit.LIKELIHOOD

            m_unll.migrad()
            m_unll.hesse()
            #m_unll.draw_mnprofile("mu")
            mu = m_unll.values["mu"]
            del_mu_stat = m_unll.errors["mu"]
            del_mu_stat_inf=-del_mu_stat
            del_mu_stat_sup=del_mu_stat
       
            return mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup
                
        mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup=UNLL_mu_computation(mu_init=mu_init)
        tes=tes_init_ref
        jes=jes_init_ref
        soft_met=soft_met_init_ref

    ###############################################################
    ###############################################################
    ######## BNLL METHOD
    ###############################################################
    elif method == "BNLL":

        #///////////////////////////////////////////////////////////////////////////////
        ## Cost_bnll
        #///////////////////////////////////////////////////////////////////////////////   
        def Cost_bnll(mu):
            N_exp_bnll = Model(
            mu=mu, sig=saved_info_hold["signal_hist"], bkg=saved_info_hold["bkg_hist"])  
            
            return -np.sum(-N_exp_bnll + n_obs_hist() * np.log(N_exp_bnll) )
    
        def BNLL_mu_computation (mu_init) :
            m_bnll = Minuit(Cost_bnll, mu=mu_init)
            m_bnll.limits["mu"] = (0, 4)
            m_bnll.errordef = Minuit.LIKELIHOOD

            m_bnll.migrad()
            m_bnll.hesse()
            #m_bnll.draw_mnprofile("mu")
            mu = m_bnll.values["mu"]
            del_mu_stat = m_bnll.errors["mu"]
            del_mu_stat_inf=-del_mu_stat
            del_mu_stat_sup=del_mu_stat

            return mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup

        mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup=BNLL_mu_computation(mu_init=mu_init)                  
        tes=tes_init_ref
        jes=jes_init_ref
        soft_met=soft_met_init_ref


    ###############################################################
    ######## BNLL_syst METHOD
    ###############################################################
    elif method == "BNLL_syst":

        import os
        def BNLL_syst_mu_fitting_param(nb_bins) :
            # # Load
            FitingData = [None] * 3
            SystName = ["TES", "JES", "SOFTMET"]
    
            current_dir = os.path.dirname(os.path.abspath(__file__))
            for i in range(3):   #######ATTENTION BESOIN DE METTRE A 3 POUR SOFT MET
                if os.path.isfile(
                    "%s/Fitting_Parameters/one_bkg/%s_%sbins_2orderFittingParam.npz" % (current_dir,SystName[i], nb_bins)
                ):
                    FitingData[i] = np.load(
                        "%s/Fitting_Parameters/one_bkg/%s_%sbins_2orderFittingParam.npz" % (current_dir,SystName[i], nb_bins)
                    )
                else:
                    print("A file is missing")

            return FitingData
        
        FitingData=BNLL_syst_mu_fitting_param(nb_bins=nb_bins)
            
        #///////////////////////////////////////////////////////////////////////////////
        ## Cost_bnll_syst
        #///////////////////////////////////////////////////////////////////////////////   
        def Cost_bnll_syst(mu ,tes , jes,soft_met): # tes, jes, soft_met):
            from systematic_analysis import Polynomial_Reg_Model_forced_jes_tes,Polynomial_Reg_Model_forced_soft_met
            sig_exp_hold_biaised = ( saved_info_hold["signal_hist"]  #saved_info_hold["signal_hist"] = sig_exp_hold_unbiaised
                                    ###Below tes
                    + np.array([Polynomial_Reg_Model_forced_jes_tes(tes, *FitingData[0]["fit_param"][j][0])for j in range(nb_bins) 
                    ])
                                    ###Below jes
                    +np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][0]) for j in range(nb_bins)
                    ])
                                    ###Below soft_met
                    + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][0]) for j in range(nb_bins)
                    ])   
                                    )
            bkg_exp_hold_biaised = ( saved_info_hold["bkg_hist"]  #saved_info_hold["bkg_hist"] = bkg_exp_hold_unbiaised
                                     ###Below tes
                    + np.array([Polynomial_Reg_Model_forced_jes_tes(tes, *FitingData[0]["fit_param"][j][1]) for j in range(nb_bins) 
                    ]) 
                                     ###Below jes
                    +np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][1]) for j in range(nb_bins)
                    ]) 
                                    ###Below soft_met
                    + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][1]) for j in range(nb_bins)
                    ])
                                   )
            
            N_exp_hold = Model(
                mu=mu, sig=sig_exp_hold_biaised, bkg=bkg_exp_hold_biaised
            )  
            return -np.sum(-N_exp_hold + n_obs_hist() * np.log(N_exp_hold))
            

        def BNLL_syst_mu_computation (mu_init ,tes_init,jes_init): # ,tes_init ,jes_init ,soft_met_init):
    
            m_bnll_syst = Minuit(Cost_bnll_syst, mu=mu_init, tes=tes_init,jes=jes_init )# tes=tes_init, jes=jes_init, soft_met=soft_met_init)
            m_bnll_syst.limits["mu"] = (0, 4)
            m_bnll_syst.limits["tes"] = (0.9, 1.1)
            m_bnll_syst.limits["jes"] = (0.9, 1.1)
            m_bnll_syst.limits["soft_met"] = (0, 5)
            m_bnll_syst.errordef = Minuit.LIKELIHOOD
            
            m_bnll_syst.fixed["tes"]=Dont_compute_tes
            m_bnll_syst.fixed["jes"]=Dont_compute_jes
            m_bnll_syst.fixed["soft_met"]=Dont_compute_soft_met

            m_bnll_syst.migrad()
            m_bnll_syst.hesse()
            #m_bnll_syst.draw_mnmatrix(cl=[1, 2, 3])
            #m_bnll_syst.draw_mnprofile("mu")
    
            del_mu_stat = m_bnll_syst.errors["mu"]
            mu = m_bnll_syst.values["mu"]
            tes = m_bnll_syst.values["tes"]
            jes = m_bnll_syst.values["jes"]
            soft_met = m_bnll_syst.values["soft_met"]
        

            del_mu_stat_inf = -del_mu_stat
            del_mu_stat_sup = del_mu_stat

            return mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup, tes ,jes,  soft_met
    

        mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup, tes, jes, soft_met =BNLL_syst_mu_computation(mu_init=mu_init ,tes_init=tes_init_ref ,jes_init=jes_init_ref,soft_met_init=soft_met_init_ref ) #tes_init=1 ,jes_init=1 ,soft_met_init=soft_met_init_ref)
                                   

    ###############################################################
    ###############################################################
    ######## BNLL_syst_normal_bkg METHOD
    ###############################################################
    elif method == "BNLL_syst_normal_bkg":
            
        #///////////////////////////////////////////////////////////////////////////////
        ## Cost_nll_syst_norm
        #///////////////////////////////////////////////////////////////////////////////         
        def Cost_nll_syst_norm(mu ,bkg_scale, ttbar_scale, diboson_scale ): # bkg_scale, ttbar_scale, diboson_scale tes, jes, soft_met):
          
            bkg_exp_hold_biaised = ( 
                    (1+bkg_scale)*saved_info_hold["ztautau_hist"]
                    +(1+bkg_scale)*(1+ttbar_scale)*saved_info_hold["ttbar_hist"]
                    +(1+bkg_scale)*(1+diboson_scale)*saved_info_hold["diboson_hist"]
                                   )

            N_exp_hold = Model(
                mu=mu, sig=saved_info_hold["signal_hist"], bkg=bkg_exp_hold_biaised
            )  
            
            return -np.sum(-N_exp_hold + n_obs_hist() * np.log(N_exp_hold))

        def BNLL_syst_norm_mu_computation (mu_init ,bkg_scale_init, ttbar_scale_init, diboson_scale_init): # bkg_scale_init, ttbar_scale_init, diboson_scale_init 

            m_bnll_syst_norm = Minuit(Cost_nll_syst_norm, mu=mu_init, bkg_scale=bkg_scale_init, ttbar_scale=ttbar_scale_init, diboson_scale=diboson_scale_init )# bkg_scale=bkg_scale_init, ttbar_scale=ttbar_scale_init, diboson_scale=diboson_scale_init 
            m_bnll_syst_norm.limits["mu"] = (0, 4)
            m_bnll_syst_norm.limits["bkg_scale"] = (-0.01,0.01)
            m_bnll_syst_norm.limits["ttbar_scale"] = (-0.2, 0.2)
            m_bnll_syst_norm.limits["diboson_scale"] = (-1, 1)
            m_bnll_syst_norm.errordef = Minuit.LIKELIHOOD

            m_bnll_syst_norm.migrad()
            m_bnll_syst_norm.hesse()
            #m_bnll_syst_norm.draw_mnmatrix(cl=[1, 2, 3,4])
            #m_bnll_syst_norm.draw_mnprofile("mu")
    
            del_mu_stat = m_bnll_syst_norm.errors["mu"]
            mu = m_bnll_syst_norm.values["mu"]
            bkg_scale = m_bnll_syst_norm.values["bkg_scale"]
            ttbar_scale = m_bnll_syst_norm.values["ttbar_scale"]
            diboson_scale = m_bnll_syst_norm.values["diboson_scale"]

            del_mu_stat_inf = -del_mu_stat
            del_mu_stat_sup = del_mu_stat
    

            return mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup,bkg_scale ,ttbar_scale,  diboson_scale
    
        mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup, bkg_scale ,ttbar_scale,  diboson_scale =BNLL_syst_norm_mu_computation(mu_init=mu_init ,
                                                                                                    bkg_scale_init=bkg_scale_init_ref, 
                                                                                                    ttbar_scale_init=ttbar_scale_init_ref, 
                                                                                                    diboson_scale_init=diboson_scale_init_ref )
        tes=tes_init_ref
        jes=jes_init_ref
        soft_met=soft_met_init_ref
                                   

    ###############################################################
    ###############################################################
    ######## BNLL_all_syst METHOD
    ###############################################################
    elif method == "BNLL_all_syst":
        
        import os
        def BNLL_all_syst_mu_fitting_param(nb_bins) :
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
            for i in range(3):   #######ATTENTION BESOIN DE METTRE A 3 for all
                if os.path.isfile(
                    "current_dir/Fitting_Parameters/bkg_subchannel/%s_3bkg_%sbins_2orderFittingParam.npz" % (SystName[i], nb_bins)
                ):
                    FitingData[i] = np.load(
                        "current_dir/Fitting_Parameters/bkg_subchannel/%s_3bkg_%sbins_2orderFittingParam.npz" % (SystName[i], nb_bins)
                    )
                else:
                    print("A file is missing \n ~~~~ \n ~~~ \n ~~ \n ~")

            return FitingData
        
        FitingData=BNLL_all_syst_mu_fitting_param(nb_bins=nb_bins)

        
        #///////////////////////////////////////////////////////////////////////////////
        ## Cost_nll_all_syst
        #/////////////////////////////////////////////////////////////////////////////// 
        def Cost_nll_all_syst(mu ,bkg_scale, ttbar_scale, diboson_scale,tes, jes, soft_met): # tes, jes, soft_met, bkg_scale, ttbar_scale, diboson_scale :
            from systematic_analysis import Polynomial_Reg_Model_forced_jes_tes,Polynomial_Reg_Model_forced_soft_met
            sig_exp_hold_biaised = ( saved_info_hold["signal_hist"]  #saved_info_hold["signal_hist"] = sig_exp_hold_unbiaised  
                                    # Below tes
                    + np.array([Polynomial_Reg_Model_forced_jes_tes(tes, *FitingData[0]["fit_param"][j][0])for j in range(nb_bins) 
                    ])
                                    # Below jes
                    +np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][0]) for j in range(nb_bins)
                    ])
                                    # Below soft_met
                    + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][0]) for j in range(nb_bins)
                    ]) 
                                   )
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ##First we define the tes, jes and soft met syst and then the scaling for ztautau,ttbar and diboson
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            bkg_scale_ztautau_exp_hold_biaised = ( saved_info_hold["ztautau_hist"]  #saved_info_hold["ztautau_hist"] = ztautau_hist_exp_hold_unbiaised  
                                                  # Below tes
                        + np.array([Polynomial_Reg_Model_forced_jes_tes(tes, *FitingData[0]["fit_param"][j][1])for j in range(nb_bins) 
                        ])
                                                  # Below jes
                        +np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][1]) for j in range(nb_bins)
                        ])
                                                   # Below soft_met
                        + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][1]) for j in range(nb_bins)
                        ])   
                                                    )
            
            bkg_ttbar_exp_hold_exp_hold_biaised = ( saved_info_hold["ttbar_hist"]  #saved_info_hold["ttbar_hist"] = ttbar_hist_exp_hold_unbiaised  
                                                    # Below tes
                        + np.array([Polynomial_Reg_Model_forced_jes_tes(tes, *FitingData[0]["fit_param"][j][2])for j in range(nb_bins) 
                        ])
                                                    # Below jes
                        +np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][2]) for j in range(nb_bins)
                        ])
                                                   # Below soft_met
                        + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][2]) for j in range(nb_bins)
                        ])   
                                                    )
            
            bkg_diboson_exp_hold_biaised =( saved_info_hold["diboson_hist"]  #saved_info_hold["diboson_hist"] = diboson_hist_exp_hold_unbiaised  
                                            # Below tes
                        + np.array([Polynomial_Reg_Model_forced_jes_tes(tes, *FitingData[0]["fit_param"][j][3])for j in range(nb_bins) 
                        ])
                                            # Below jes
                        +np.array([Polynomial_Reg_Model_forced_jes_tes(jes,*FitingData[1]["fit_param"][j][3]) for j in range(nb_bins)
                        ])
                                           # Below soft_met
                        + np.array([Polynomial_Reg_Model_forced_soft_met(soft_met,*FitingData[2]["fit_param"][j][3]) for j in range(nb_bins)
                        ])   
                                            )
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ##Then the normalisation weight uncertainty
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            bkg_exp_hold_biaised = ( 
                                    (1+bkg_scale)*bkg_scale_ztautau_exp_hold_biaised
                                    +(1+bkg_scale)*(1+ttbar_scale)*bkg_ttbar_exp_hold_exp_hold_biaised
                                    +(1+bkg_scale)*(1+diboson_scale)*bkg_diboson_exp_hold_biaised
                                   )

            N_exp_hold = Model(
                mu=mu, sig=sig_exp_hold_biaised, bkg=bkg_exp_hold_biaised
            ) 
            return -np.sum(-N_exp_hold + n_obs_hist() * np.log(N_exp_hold))
        

        def BNLL_all_syst_mu_computation (mu_init ,tes_init ,jes_init ,soft_met_init, bkg_scale_init, ttbar_scale_init, diboson_scale_init): # bkg_scale_init, ttbar_scale_init, diboson_scale_init ,tes_init ,jes_init ,soft_met_init):

            m_bnll_all_syst = Minuit(Cost_nll_all_syst, mu=mu_init,tes=tes_init ,jes=jes_init ,soft_met=soft_met_init, bkg_scale=bkg_scale_init, ttbar_scale=ttbar_scale_init, diboson_scale=diboson_scale_init )# bkg_scale=bkg_scale_init, ttbar_scale=ttbar_scale_init, diboson_scale=diboson_scale_init tes=tes_init, jes=jes_init, soft_met=soft_met_init)
            m_bnll_all_syst.limits["mu"] = (0, 4)
            m_bnll_all_syst.limits["tes"] = (0.9, 1.1)
            m_bnll_all_syst.limits["jes"] = (0.9, 1.1)
            m_bnll_all_syst.limits["soft_met"] = (0, 5)
            m_bnll_all_syst.limits["bkg_scale"] = (-0.01,0.01)
            m_bnll_all_syst.limits["ttbar_scale"] = (-0.2, 0.2)
            m_bnll_all_syst.limits["diboson_scale"] = (-1, 1)
            m_bnll_all_syst.errordef = Minuit.LIKELIHOOD
            
            m_bnll_all_syst.fixed["tes"]=Dont_compute_tes
            m_bnll_all_syst.fixed["jes"]=Dont_compute_jes
            m_bnll_all_syst.fixed["soft_met"]=Dont_compute_soft_met

            m_bnll_all_syst.migrad()
            m_bnll_all_syst.hesse()
            #m_bnll_all_syst.draw_mnmatrix(cl=[1, 2, 3,4,5,6])
            #m_bnll_all_syst.draw_mnprofile("mu")
    
            del_mu_stat = m_bnll_all_syst.errors["mu"]
            mu = m_bnll_all_syst.values["mu"]
            bkg_scale = m_bnll_all_syst.values["bkg_scale"]
            ttbar_scale = m_bnll_all_syst.values["ttbar_scale"]
            diboson_scale = m_bnll_all_syst.values["diboson_scale"]
            tes = m_bnll_all_syst.values["tes"]
            jes = m_bnll_all_syst.values["jes"]
            soft_met = m_bnll_all_syst.values["soft_met"]


            del_mu_stat_inf = -del_mu_stat
            del_mu_stat_sup = del_mu_stat

            return mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup,tes, jes, soft_met,bkg_scale ,ttbar_scale,  diboson_scale
    

        mu, del_mu_stat,del_mu_stat_inf,del_mu_stat_sup,tes, jes, soft_met, bkg_scale ,ttbar_scale,  diboson_scale =BNLL_all_syst_mu_computation(
                                                                                        mu_init=mu_init ,
                                                                                        tes_init=tes_init_ref ,jes_init=jes_init_ref ,soft_met_init=soft_met_init_ref, 
                                                                                        bkg_scale_init=bkg_scale_init_ref, ttbar_scale_init=ttbar_scale_init_ref, diboson_scale_init=diboson_scale_init_ref 
                                    ) #tes_init=1 ,jes_init=1 ,soft_met_init=0,bkg_scale_init=0, ttbar_scale_init=0, diboson_scale_init=0)
                                  
        
    else:
        print(
            "There is a problem in the computing of mu : The method label is invalid."
        )

    del_mu_sys = abs(0.0 * mu)
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    mu_axis_values = np.linspace(mu-1, mu+1, 1_000)

    if method =="Direct" :
        negloglike_values = [0]*len(mu_axis_values)
        negloglike_mu_hat = 0
        
    if method =="UNLL" :
        negloglike_values = np.array([Cost_unll(mub) for mub in mu_axis_values])
        negloglike_mu_hat = Cost_unll(mu)  
       
        
    elif (method == "BNLL"):  #or (method == "BNLL_syst")or (method == "BNLL_syst_normal_bkg") or (method == "BNLL_all_syst")
        negloglike_values = np.array([Cost_bnll(mub) for mub in mu_axis_values])
        negloglike_mu_hat = Cost_bnll(mu)  # saved_info["signal"]+saved_info["bkg"])


    elif method=="BNLL_syst":   ######  and method=="REf" Just BLOCK THE CALCULATION
        negloglike_values = np.array(
            [Cost_bnll_syst(mu=mub ,tes=tes, jes=jes,soft_met=soft_met) for mub in mu_axis_values] #tes=tes, jes=jes, soft_met=soft_met
        )
        negloglike_mu_hat = Cost_bnll_syst(mu=mu ,tes=tes, jes=jes,soft_met=soft_met )# tes=tes,  jes=jes, soft_met=soft_met)


    elif method == "BNLL_syst_normal_bkg":
        negloglike_values = np.array(
            [Cost_nll_syst_norm(mu=mub ,bkg_scale=bkg_scale, ttbar_scale=ttbar_scale, diboson_scale=diboson_scale) for mub in mu_axis_values]
            )
        negloglike_mu_hat = Cost_nll_syst_norm(mu=mu ,bkg_scale=bkg_scale, ttbar_scale=ttbar_scale, diboson_scale=diboson_scale)


    elif method=="BNLL_all_syst" :
        negloglike_values = np.array(
        [Cost_nll_all_syst(mu=mub ,tes=tes,  jes=jes, soft_met=soft_met, bkg_scale=bkg_scale, ttbar_scale=ttbar_scale, diboson_scale=diboson_scale) for mub in mu_axis_values] #tes=tes, jes=jes, soft_met=soft_met
        )
        negloglike_mu_hat = Cost_nll_all_syst(mu=mu ,tes=tes,  jes=jes, soft_met=soft_met, bkg_scale=bkg_scale, ttbar_scale=ttbar_scale, diboson_scale=diboson_scale)# tes=tes,  jes=jes, soft_met=soft_met)

    print("mu",mu)
    #print("Returned mu=",mu, " Delta_mu_tot=",del_mu_tot)
    #print("method in statistical analysis",method)
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
        "soft_met":soft_met
    }




def calculate_saved_info(
    score,
    holdout_set,
    threshold=0,
    nb_bins=1,
    model=0,
    #find_threshold=False,
    #NbPoints_Prec_Thresh=50,
    #Del_Mu_method="None",
):

    # del_signal=np.sqrt(np.sum(np.power(weight_ROIscore[label_Roiscore==1], 2)))
    # del_bkg=np.sqrt(np.sum(np.power(weight_ROIscore[label_Roiscore==0], 2)))


    weight_ROIscore_exp_holdout = holdout_set["weights"][score > threshold]
    detailed_labels_Roiscore_exp_holdout = holdout_set["detailed_labels"][score > threshold]
    score_ROIscore_exp_holdout = score[score > threshold]

    signal=np.sum(weight_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout == "htautau"])
    bkg=np.sum(weight_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout != "htautau"])
    N=signal + bkg
    #N=np.sum(weight_ROIscore_exp_holdout)
    #bkg=N-signal


    Bins_edges = np.linspace(0, 1, nb_bins + 1)

    signal_hist = np.histogram(
        score_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout == "htautau"],
        bins=Bins_edges,
        weights=weight_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout == "htautau"],)[0]
    
    ztautau_hist = np.histogram(
        score_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout == "ztautau"],
        bins=Bins_edges,
        weights=weight_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout == "ztautau"],)[0]

    ttbar_hist = np.histogram(
        score_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout == "ttbar"],
        bins=Bins_edges,
        weights=weight_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout == "ttbar"],)[0]

    diboson_hist = np.histogram(
        score_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout == "diboson"],
        bins=Bins_edges,
        weights=weight_ROIscore_exp_holdout[detailed_labels_Roiscore_exp_holdout == "diboson"],)[0]
    


    saved_info = {
        "N": N,  # Total nb of events in the ROI
        "bkg": bkg,
        "signal": signal,
        
        "ztautau_hist": ztautau_hist,
        "ttbar_hist": ttbar_hist,
        "diboson_hist": diboson_hist,
        "bkg_hist": ztautau_hist +ttbar_hist+diboson_hist ,
        "signal_hist":signal_hist,

        "threshold":threshold,
        "nb_bins":nb_bins,
        # "del_beta": del_beta,
        # "del_gamma": del_gamma,
    }

    # print("saved_info", saved_info)
    return saved_info



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
        saved_info_tamp_valid= calculate_saved_info(
            score_valid_test, valid_test_set, threshold_list[i]
        )

        # print("saved_info_tamp in best thresh calculate"," sig ",saved_info_tamp["signal"]," bkg ",saved_info_tamp["bkg"])
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
        if not os.path.exists("images"):
            os.makedirs("current_dir/images")
        plt.savefig(
            "current_dir/images/Best_Mu_AMS_Thresh_%s_Train=%s_Holdout=%s_Valid=%s.png"
            % (ModelType, NbTrain, NbHoldout, NbValidation)
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



def calculate_AMS(saved_info, beta_reg=10e-2):
    """
    Calculate AMS based on :
    """

    AMS = np.sqrt(
        2
        * (
            (  # Not needed
                (saved_info["signal"] + saved_info["bkg"] + beta_reg)
                * np.log(1 + (saved_info["signal"] / (saved_info["bkg"] + beta_reg)))
            )
            - saved_info["signal"]
        )
    )
    return AMS
