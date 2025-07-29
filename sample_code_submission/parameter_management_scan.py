class Parameter_Distribution:

    import numpy as np
    
    import time
    seed = int(time.time_ns() % (2**32))  # or use os.urandom() if needed
    

    parameter = {
        "ModelType": "BDT",
        "Load_Classifier":True,

        "THV_size": np.array([5_000_000, 50_000_000, 20_000_000]),  #5M for BDT 40 for NN   2_500_000 for little BDT    50_000_000, 20_000_000
        "Nb_bins_distrib": 20,
        "threshold_distrib": 0,

        "random_seed":10912983, #10912983 or seed for random seed

        "First_plots_hist_roc": False ,

        "Bins_varia_plot": False,
        "Bins_varia_Min_Max_Step": [1,52,5],

        "Features_VS_syst": True,

        "Compute_Best_Opti": False,
        "NbPoints_Prec_Thresh":50,

        "Parabola_method":[],

#        The method define the model used
#  "BNLL_all_syst" = all syst 
#  "BNLL_syst_normal_bkg"  = bkg_scale, ttbar_scale and diboson_scale but without tes or jes
#  "BNLL_syst" = tes and jes (soft_met is commented out)
#  "BNLL" = only binned 
#  "Direct"
#  "UNLL"
        "Predict_method":[ "BNLL" ],  #l'autre c'etait all syst

        "Dont_compute_tes" : False,
        "Dont_compute_jes" : False,
        "Dont_compute_soft_met" : False,

        "Force_3bkg_regression": False ,
        "Force_1bkg_regression": False ,

        "fitting_3bkg": False ,
        "fitting_1bkg": False ,

    }



    @classmethod
    def get_variable(cls, Param_Name):
        return cls.parameter[Param_Name]
    
    @classmethod
    def get_all(cls):
        return cls.parameter

    @classmethod
    def get_THV_ModelType (cls):
        keys = ["ModelType", "THV_size"]
        return {k: cls.parameter[k] for k in keys}
    
    @classmethod
    def overwrite_size(cls, Algebraic_diff_Pop_Main_Sub, Main_Study_Subset, Pop_modified ):
        import numpy as np
        
        #Modification_Size_Method= nbr it means I give all the difference to the other subset (0:Train, 1:Holdout, 2:Validation)
        #The main study subset is defined thanks to number (0:Train, 1:Holdout, 2:Validation)
        New_Pop=np.array(cls.parameter["THV_size"])

        Security_Sum=np.sum(New_Pop)

        New_Pop[Main_Study_Subset]+=Algebraic_diff_Pop_Main_Sub  #algebraic addition of the subset

        Algebraic_diff_Pop_Main_Sub/=len(Pop_modified)
        New_Pop[Pop_modified]=New_Pop[Pop_modified]-Algebraic_diff_Pop_Main_Sub


        #######To debug 
        # if type(Algebraic_diff_Pop_Main_Sub)!=int :
        #     print("The algebric diff pop was odd and Pop modified 2 : Nb of Pop modified",Pop_modified," Diff ",Algebraic_diff_Pop_Main_Sub)
        #     # Algebraic_diff_Pop_Main_Sub=np.ceil(Algebraic_diff_Pop_Main_Sub)
        #     # New_Pop[Pop_modified[0]]-=Algebraic_diff_Pop_Main_Sub
        #     # New_Pop[Pop_modified[-1]]-=Algebraic_diff_Pop_Main_Sub+1
        #     New_Pop[Pop_modified]=np.ceil(New_Pop[Pop_modified])
        #     New_Pop[Pop_modified[-1]]-=1




        # else :
        #      New_Pop[Pop_modified]-=Algebraic_diff_Pop_Main_Sub


        if Security_Sum!=np.sum(New_Pop) :
            print("There is a problem in the population modificiation, Ini total pop = ",Security_Sum," ,New = ",np.sum(New_Pop))



        cls.parameter["THV_size"] = New_Pop
    

    @classmethod
    def overwrite_ModelType (cls, New_Model):
        cls.parameter["ModelType"] = New_Model
