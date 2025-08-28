def Test_Variable_SubSize(
    dataset,
    Main_Study_Subset=0,
    Pop_modified=[1],
    Average_EndPop_MainSubset=25_000,
    Nb_Step=3,
):

    import numpy as np
    import matplotlib.pyplot as plt
    from parameter_management_scan import (
        Parameter_Distribution,
    )  # To change the pop size
    from HiggsML.ingestion import Ingestion
    from model import Model

    Tamp_parameter = Parameter_Distribution.get_all()
    ModelType = Tamp_parameter["ModelType"]
    THV_size = Tamp_parameter["THV_size"]
    NbTrain = THV_size[0]
    NbHoldout = THV_size[1]
    NbValidation = THV_size[2]
    Float_Step = (Average_EndPop_MainSubset - THV_size[Main_Study_Subset]) / Nb_Step
    AlgebraicPopStep = int(Float_Step)  # Sous-estime toujours
    if AlgebraicPopStep == 0:
        AlgebraicPopStep == 1 * np.sign(Float_Step)

    Pop_modified = np.array(Pop_modified)
    Parameter_Distribution.overwrite_size(
        int(AlgebraicPopStep), Main_Study_Subset, Pop_modified
    )
    Best_AMS = np.zeros(Nb_Step)
    Best_mu = np.zeros(Nb_Step)
    Best_del_mu = np.zeros(Nb_Step)
    Best_mu_AMS = np.zeros(Nb_Step)
    Best_AMS_thresh = np.zeros(Nb_Step)
    Main_Pop_Size = np.zeros(Nb_Step)

    for i in range(Nb_Step):
        print("Step nbr ", i)
        PopTamp = Parameter_Distribution.get_variable("THV_size")
        if PopTamp.any() <= 0:  # In fact mainly test the modified or the main
            break

        Main_Pop_Size[i] = PopTamp[Main_Study_Subset]

        ingestion = Ingestion(dataset)
        ingestion.init_submission(Model, ModelType)
        ingestion.fit_submission()
        Fitting_tamp = ingestion.model

        Best_AMS[i] = Fitting_tamp.best_opti["best_AMS"]
        Best_del_mu[i] = Fitting_tamp.best_opti["best_del_mu"]
        Best_mu_AMS[i] = Fitting_tamp.best_opti["best_mu_AMS"]
        Best_AMS_thresh[i] = Fitting_tamp.best_opti["AMS_best_threshold"]

        Parameter_Distribution.overwrite_size(
            int(AlgebraicPopStep), Main_Study_Subset, Pop_modified
        )

    fig, ax = fig, axes = plt.subplot_mosaic("AA;BC", layout="constrained")
    fig.patch.set_facecolor("white")
    axa = ax["A"].twinx()
    axb = ax["B"].twinx()
    axc = ax["C"].twinx()
    ax["A"].plot(Main_Pop_Size, Best_AMS, label="Best_AMS", color="black")
    axa.plot(Main_Pop_Size, Best_del_mu, label="Best_del_mu", color="dodgerblue")
    ax["B"].plot(Main_Pop_Size, Best_AMS, label="Best_AMS", color="black")
    axb.plot(Main_Pop_Size, Best_mu_AMS, label="Best_mu_AMS", color="cyan")
    ax["C"].plot(Main_Pop_Size, Best_AMS, label="Best_AMS", color="black")
    axc.plot(Main_Pop_Size, Best_AMS_thresh, label="Best_AMS_thresh", color="green")
    ax["A"].tick_params(axis="y", labelcolor="black")
    ax["B"].tick_params(axis="y", labelcolor="black")
    ax["C"].tick_params(axis="y", labelcolor="black")
    axa.tick_params(axis="y", labelcolor="cyan")
    axb.tick_params(axis="y", labelcolor="dodgerblue")
    axc.tick_params(axis="y", labelcolor="green")
    ax["A"].grid()
    ax["B"].grid()
    ax["C"].grid()
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # ax[3].legend()
    ax["A"].set_xlabel("Train Population Size")
    ax["B"].set_xlabel("Train Population Size")
    ax["C"].set_xlabel("Train Population Size")
    ax["A"].set_ylabel("Best AMS")
    ax["B"].set_ylabel("Best AMS")
    ax["C"].set_ylabel("Best AMS")
    axa.set_ylabel("Best Delta mu", color="cyan")
    axb.set_ylabel("Best mu (AMS ref)", color="dodgerblue")
    axc.set_ylabel("Best threshold (AMS ref)", color="green")
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("current_dir/images"):
        os.makedirs("current_dir/images")
    plt.savefig(
        "current_dir/images/Better_Best_Mu_AMS_Thresh_Vs_TrainSize_%s_MuMthde_%s_PopStep=%s_Train_ini=%s_Holdout_ini=%s_Valid=%s.png"
        % (
            ModelType,
            Fitting_tamp.best_opti["mu_del_val_method"],
            AlgebraicPopStep,
            NbTrain,
            NbHoldout,
            NbValidation,
        ),
        facecolor="white",
    )
    plt.show()

    # # if AlgebraicPopStep>0 :
    # #     Sign=-1
    # # while (Sign*Parameter_Distribution.get_variable("THV_size")>Sign*Average_EndPop_MainSubset) :
    # if AlgebraicPopStep<0 : #Neg Step
    #     while (Parameter_Distribution.get_variable("THV_size")>Average_EndPop_MainSubset) :
    #         ingestion = Ingestion(dataset)
    #         ingestion.init_submission(Model,ModelType)
    #         ingestion.fit_submission()
    #         Parameter_Distribution.overwrite_size(AlgebraicPopStep, Main_Study_Subset, Pop_modified )

    # else :    #Pos Step
    #     while (Parameter_Distribution.get_variable("THV_size")<Average_EndPop_MainSubset) :
    #         ingestion = Ingestion(dataset)
    #         ingestion.init_submission(Model,ModelType)
    #         ingestion.fit_submission()
    #         Parameter_Distribution.overwrite_size(AlgebraicPopStep, Main_Study_Subset, Pop_modified )
