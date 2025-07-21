class Parameter_Distribution:

    import numpy as np

    parameter = {
        "ModelType": "BDT",
        "THV_size": np.array([600_000, 5_000_000, 4_000_000]),
    }



    @classmethod
    def get_variable(cls, Param_Name):
        return cls.parameter[Param_Name]
    
    @classmethod
    def get_all(cls):
        return cls.parameter

    
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
