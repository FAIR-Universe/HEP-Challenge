import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# --------------------------------------  Ploteur

def Correlation_big_graph(dfplot,data,caption,Detailled_label,Detailled_label_type,Nb_feature) :
    from parameter_management_scan import Parameter_Distribution
    Tamp_parameter=Parameter_Distribution.get_all()
    THV_size=Tamp_parameter["THV_size"]
    ModelType=Tamp_parameter["ModelType"]

    Ax_nbr=0
    Correlation_matrix_list=[0,0,0,0]

    sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")
    fig,ax=plt.subplots(2,3, layout='constrained',figsize=(24,17),sharex=True)
    #fig,ax=plt.subplots(3,2, layout='constrained',figsize=(19,24),sharex=True)
    ax = ax.flatten()

    for j in range(4):
        for i in range(2):

            Bkg_Sig_dfplot=dfplot[data["labels"] == i]   #Sert surement a rien ###############################################
            #############################
            #############
            ########
            ####
            ##
            #
            Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[data["detailed_labels"] == Detailled_label[j]]
            if  Detail_Sig_Bkg_dfplot.empty == False :
                Correlation_matrix_list[Ax_nbr]=Detail_Sig_Bkg_dfplot.corr()
                sns.heatmap(Correlation_matrix_list[Ax_nbr], vmin=-1,vmax=1,annot=False, linewidth=.5,linecolor='black',cmap="Spectral_r",ax=ax[Ax_nbr])    #rainbow  #Spectral_r #gnuplot
                ax[Ax_nbr].set_title("%s events considered as %s"%(Detailled_label[j],caption[Detailled_label_type[Ax_nbr]]),fontsize=17)
                Ax_nbr+=1
    Bkg_Sig_dfplot=dfplot[data["labels"] == 0]
    Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[data["detailed_labels"] != "htautau"]
    Weighted_Correlation_matrix_mean_bkg=Detail_Sig_Bkg_dfplot.corr()
    #Correlation_matrix_mean_bkg=(Correlation_matrix_list[1]+Correlation_matrix_list[2]+Correlation_matrix_list[3])/3
    sns.heatmap( Weighted_Correlation_matrix_mean_bkg , vmin=-1,vmax=1, annot=False, linewidth=.5,cmap="Spectral_r",linecolor='black',ax=ax[4])
    ax[4].set_title("Weighted mean Background Correlation matrix",fontsize=17)
    sns.heatmap( np.abs((Correlation_matrix_list[0]-Weighted_Correlation_matrix_mean_bkg)/2) , annot=False, linewidth=.5,linecolor='black',cmap="afmhot_r",ax=ax[5])
    ax[5].set_title("Absolute difference between the correlation matrix of signal and weighted mean background",fontsize=17)

    plt.suptitle("Correlation matrix of features",fontsize=27)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("current_dir/images/FeaturesAnalysis_Corr_Distri"):
        os.makedirs("current_dir/images/FeaturesAnalysis_Corr_Distri")
    plt.savefig("current_dir/images/FeaturesAnalysis_Corr_Distri/BlackSwan_CorrelationMat_DataSet_tot_size%s.png"%(np.sum(THV_size)))

    plt.show()
    del Bkg_Sig_dfplot,Detail_Sig_Bkg_dfplot,Weighted_Correlation_matrix_mean_bkg


def Correlation_diff_graph(dfplot,data,caption,Detailled_label,Detailled_label_type,detailled_label_num_ref,Nb_feature) :
    from parameter_management_scan import Parameter_Distribution
    Tamp_parameter = Parameter_Distribution.get_all()
    THV_size = Tamp_parameter["THV_size"]
    ModelType = Tamp_parameter["ModelType"]


    Correlation_matrix_list=[0,0,0,0]
    sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")
    fig,ax=plt.subplots(2,3, layout='constrained',figsize=(24,17),sharex=True)
    #fig,ax=plt.subplots(3,2, layout='constrained',figsize=(19,24),sharex=True)
    ax = ax.flatten()

    Bkg_Sig_dfplot=dfplot[data["labels"] == Detailled_label_type[detailled_label_num_ref] ]
    Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[data["detailed_labels"] == Detailled_label[detailled_label_num_ref] ]
    Correlation_matrix_list[detailled_label_num_ref]=Detail_Sig_Bkg_dfplot.corr()   
    sns.heatmap(Correlation_matrix_list[detailled_label_num_ref], vmin=-1,vmax=1,annot=False, linewidth=.5,linecolor='black',cmap="Spectral_r",ax=ax[0])    #rainbow  #Spectral_r #gnuplot
    ax[1].set_title("Reference: %s events considered as %s"%(Detailled_label[detailled_label_num_ref],caption[Detailled_label_type[detailled_label_num_ref]]),fontsize=17)

    Ax_nbr=1
    for j in range (0,4) :
        if j!=detailled_label_num_ref :

            Bkg_Sig_dfplot=dfplot[data["labels"] == Detailled_label_type[j]]
            Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[data["detailed_labels"] == Detailled_label[j]]
            Correlation_matrix_list[j]=Detail_Sig_Bkg_dfplot.corr()
            sns.heatmap( np.abs(Correlation_matrix_list[j]-Correlation_matrix_list[detailled_label_num_ref])/2,vmin=0,vmax=0.5, annot=False, linewidth=.5,linecolor='black',cmap="afmhot_r",ax=ax[Ax_nbr])    #rainbow  #Spectral_r #gnuplot
            ax[Ax_nbr].set_title("Absolute difference between the correlation matrix of %s and and %s "%(Detailled_label[detailled_label_num_ref],Detailled_label[j]),fontsize=17)
            Ax_nbr+=1

    Bkg_Sig_dfplot=dfplot[data["labels"] == 0]
    Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[data["detailed_labels"] != Detailled_label[detailled_label_num_ref]]
    Weighted_Correlation_mean_other_bkg=Detail_Sig_Bkg_dfplot.corr()
    #Correlation_matrix_mean_bkg=(Correlation_matrix_list[1]+Correlation_matrix_list[2]+Correlation_matrix_list[3])/3
    sns.heatmap( np.abs(Weighted_Correlation_mean_other_bkg-Correlation_matrix_list[detailled_label_num_ref])/2 ,vmin=0,vmax=0.5, annot=False, linewidth=.5,cmap="afmhot_r",linecolor='black',ax=ax[4])
    ax[4].set_title("Absolute difference between the correlation matrix\ of\n %s and and weighted mean of all the other background "%(Detailled_label[detailled_label_num_ref]),fontsize=17)

    Detail_Sig_Bkg_dfplot =dfplot[data["detailed_labels"] != Detailled_label[detailled_label_num_ref]]
    Weighted_Correlation_mean_other_tot=Detail_Sig_Bkg_dfplot.corr()
    sns.heatmap( np.abs(Weighted_Correlation_mean_other_tot-Correlation_matrix_list[detailled_label_num_ref])/2 ,vmin=0,vmax=0.5, annot=False, linewidth=.5,cmap="afmhot_r",linecolor='black',ax=ax[5])
    ax[5].set_title("Absolute difference between the correlation matrix of\n %s and and weighted mean of all the other data "%(Detailled_label[detailled_label_num_ref]),fontsize=17)

    plt.suptitle("Difference between the correlation matrix of features considering as reference %s"%(Detailled_label[detailled_label_num_ref]),fontsize=27)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("current_dir/images/FeaturesAnalysis_Corr_Distri"):
        os.makedirs("current_dir/images/FeaturesAnalysis_Corr_Distri")
    plt.savefig("current_dir/images/FeaturesAnalysis_Corr_Distri/BlackSwan_DiffCorrMat_ref_%s_DataSet_tot_size%s.png"%(Detailled_label[detailled_label_num_ref],np.sum(THV_size)))

    plt.show()
    del Bkg_Sig_dfplot,Detail_Sig_Bkg_dfplot, Weighted_Correlation_mean_other_tot,Weighted_Correlation_mean_other_bkg


# --------------------------------------  Fct mise en forme panda table and appel fct ploteur

def feature_correlations(data,field_name):

    dfplot = pd.DataFrame(
                data,
                # columns=[
                #     # "PRI_lep_phi",
                #     # "PRI_met",
                #     # "DER_pt_ratio_lep_had",
                #     # "DER_deltaeta_jet_jet",

                #     "PRI_lep_pt",
                #     "PRI_lep_eta",
                #     "PRI_lep_phi",
                #     "PRI_had_pt",
                #     "PRI_had_eta",
                #     "PRI_had_phi",
                #     "PRI_jet_leading_pt",
                #     "PRI_jet_leading_eta",
                #     "PRI_jet_leading_phi",
                #     "PRI_jet_subleading_pt",
                #     "PRI_jet_subleading_eta",
                #     "PRI_jet_subleading_phi",
                #     "PRI_n_jets",
                #     "PRI_jet_all_pt",
                #     "PRI_met",
                #     "PRI_met_phi",
                #     #"weights",###########################A retirer surement
                #     "DER_mass_transverse_met_lep",
                #     "DER_mass_vis",
                #     "DER_pt_h",
                #     "DER_deltaeta_jet_jet",
                #     "DER_mass_jet_jet",
                #     "DER_prodeta_jet_jet",
                #     "DER_deltar_had_lep",
                #     "DER_pt_tot",
                #     "DER_sum_pt",
                #     "DER_pt_ratio_lep_had",
                #     "DER_met_phi_centrality",
                #     "DER_lep_eta_centrality",
                # ],
                columns=field_name,
            )
    Nb_feature=len(field_name)
    caption = ["Background_feature","Signal_feature"]
    Detailled_label_list=["htautau","diboson","ttbar","ztautau"]
    Detailled_label_type_list=[1,0,0,0]

    Correlation_big_graph(dfplot=dfplot,data=data,caption=caption,Detailled_label=Detailled_label_list,Detailled_label_type=Detailled_label_type_list,Nb_feature=Nb_feature)
    for j in range(0,4) :
        Correlation_diff_graph(
            dfplot=dfplot,data=data,caption=caption,Detailled_label=Detailled_label_list,
            Detailled_label_type=Detailled_label_type_list,detailled_label_num_ref=j,Nb_feature=Nb_feature
        )

    del dfplot





# ##Based on the code for the stacked_histogram in HiggsML.visualisation

def stacked_histogram_modified(
    dfall,
    target,
    weights,
    detailed_label,
    field_name,
    mu_hat=1.0,
    nbins=30,
    y_scale="linear",
    shape="compact",
):
    """
    Plots a stacked histogram of a specific field in the dataset.

    Args:
        * dfall : Pandas Dataframe
        * target : numpy array with labels
        * weights : numpy array with event weights
        * weights : numpy array with detailed labels of the events
        * detailed_label : The name of the field to plot.
        * mu_hat : The value of mu (default: 1.0).
        * bins (int): The number of bins for the histogram (default: 30).

    .. Image:: images/stacked_histogram.png
    """
    import math
    from parameter_management_scan import Parameter_Distribution
    Tamp_parameter = Parameter_Distribution.get_all()
    THV_size = Tamp_parameter["THV_size"]
    ModelType = Tamp_parameter["ModelType"]
    
    if shape=="compact" :
        nb_col=5
    elif shape=="long":
        nb_col=3
    elif shape=="verylong" :   #Better tu use number
        nb_col=2
    
    Nbr_field_name=len(field_name)
    fig,ax=plt.subplots(math.ceil(Nbr_field_name/nb_col),nb_col, layout='constrained',figsize=(20,(20/nb_col)*math.ceil(Nbr_field_name/nb_col)))
    ax = ax.flatten()
    for i in range (Nbr_field_name) :
        field = dfall[field_name[i]]

        weight_keys = {}
        keys = np.unique(detailed_label)

        for key in keys:
            weight_keys[key] = weights[detailed_label == key]

        #print("keys", keys)
        #print("keys 2", weight_keys.keys())

        sns.set_theme(rc={"figure.figsize": (8, 7)}, style="whitegrid")



        """
        field_clipped = field[(field >= lower_bound) & (field <= upper_bound)]
        weights_clipped = weights[(field >= lower_bound) & (field <= upper_bound)]
        target_clipped = target[(field >= lower_bound) & (field <= upper_bound)]
        detailed_labels_clipped = detailed_label[
            (field >= lower_bound) & (field <= upper_bound)
        ]
        """
        
        field_clipped = field
        weights_clipped = weights
        target_clipped = target
        detailed_labels_clipped = detailed_label
        

        min_value = field_clipped.min()
        max_value = field_clipped.max()

        # Define the bin edges
        bins = np.linspace(min_value, max_value, nbins + 1)

        hist_s, bins = np.histogram(
            field_clipped[target_clipped == 1],
            bins=bins,
            weights=weights_clipped[target_clipped == 1],
        )

        hist_b, bins = np.histogram(
            field_clipped[target_clipped == 0],
            bins=bins,
            weights=weights_clipped[target_clipped == 0],
        )

        hist_bkg = (hist_b).copy()  #hist_b+hist_s

        color=["purple","orange","cornflowerblue","forestgreen"]
        ax[i].stairs(hist_s+hist_b, bins, fill=True, label="htautau sig", color=color[0])
        for j, key in enumerate(keys[keys!="htautau"]):
            print("key=",key)
            hist, bins = np.histogram(
                field_clipped[detailed_labels_clipped == key],
                bins=bins,
                weights=weights_clipped[detailed_labels_clipped == key],
            )
            ax[i].stairs(hist_b, bins, fill=True, label=f"{key} bkg", color=color[j+1])  #hist_b
            hist_b -= hist
           
                
        ax[i].stairs(
            hist_s * mu_hat + hist_bkg,
            bins,
            fill=False,
            color="yellow",
            linewidth=1.5,
            label=f"$H \\rightarrow \\tau \\tau (\\mu = {mu_hat:.3f})$",
        )

        ax[i].stairs(
            hist_s + hist_bkg,
            bins,
            fill=False,
            color="red",
            linewidth=0.5,
            label=f"$H \\rightarrow \\tau \\tau (\\mu = {1.0:.3f})$",
        )

        ax[i].set_title(f"Stacked histogram of {field_name[i]}")
        ax[i].set_yscale(y_scale) 
        ax[i].set_xlabel(f"{field_name[i]}")
        ax[i].set_ylabel("Weighted count")
        ax[i].legend()

    plt.legend()
    plt.suptitle("Stacked Histogram for differents features",fontsize=27)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("current_dir/images/FeaturesAnalysis_Corr_Distri"):
        os.makedirs("current_dir/images/FeaturesAnalysis_Corr_Distri")
    plt.savefig("current_dir/images/FeaturesAnalysis_Corr_Distri/BlackSwan_StackedHist_%s_DataSet_tot_size%s.png"%(shape,THV_size[0]+THV_size[1]+THV_size[2]))

        
        # plt.xlabel(f"{field_name}")
        # plt.ylabel("Weighted count")
        # plt.yscale(y_scale)
    plt.show()


def features_systematics_dependence (dfall,systematics,columns,nb_bins=20):
    
    # histogram_dataset(
    #     dfall=data_set,
    #     target=target,
    #     weights=weights,
    #     columns=["PRI_lep_phi", "PRI_met", "DER_mass_vis", "DER_deltaeta_jet_jet"],
    #     nbin=25
    # )
    """
    Plots histograms of the dataset features.

    Args:
        * columns (list): The list of column names to consider (default: None, which includes all columns).
        * nbin (int): The number of bins for the histogram (default: 25).

    .. Image:: images/histogram_datasets.png
    """

    from parameter_management_scan import Parameter_Distribution
    Tamp_parameter = Parameter_Distribution.get_all()
    THV_size = Tamp_parameter["THV_size"]
    ModelType = Tamp_parameter["ModelType"]

    weights = dfall["weights"]
    target = dfall["labels"]


    import os
    import matplotlib.pyplot as plt
    import logging
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()


    logging.basicConfig(
        level=getattr(
            logging, log_level, logging.INFO
        ),  # Fallback to INFO if the level is invalid
        format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    if columns is None:
        columns = columns
    else:
        for col in columns:
            if col not in dfall.columns:   #maybee just columns
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")

    sns.set_theme(style="whitegrid")

    dfall_tamp=dfall.copy()
    dfall_poscut=systematics(dfall_tamp)
    del dfall_tamp
    df_ref = pd.DataFrame(dfall_poscut, columns=columns)
    del dfall_poscut


    # Number of rows and columns in the subplot grid
    n_cols = 2  # Number of columns in the subplot grid
    n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 6 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

    for i, column in enumerate(columns):
        # Determine the combined range for the current column

        print(f"[*] --- {column} histogram")

        min_value = df_ref[column].min()
        max_value = df_ref[column].max()

        # Define the bin edges
        bin_edges = np.linspace(min_value, max_value, nb_bins + 1)  #Serait bien de définir min et max par rapport à la valeur moyenne

        signal_field = df_ref[target == 1][column]
        background_field = df_ref[target == 0][column]
        del df_ref
        signal_weights = weights[target == 1]
        background_weights = weights[target == 0]
        
        signal_hist_ref=np.histogram(signal_field,bins=bin_edges,weights=signal_weights,density=True)[0]
        bkg_hist_ref=np.histogram(background_field,bins=bin_edges,weights=background_weights,density=True)[0]
        del signal_weights, background_weights

        var_lenght=3
        tes=np.linspace(0.9,1.1,var_lenght)
        signal_hist=[None]*var_lenght
        bkg_hist=[None]*var_lenght
        for i in range(var_lenght) :
            dfall_tamp=dfall.copy()
            dfall_poscut=systematics(dfall_tamp,tes[i])
            del dfall_tamp
            df_syst = pd.DataFrame(dfall_poscut, columns=columns)
            del dfall_poscut
            
            #########################################################
            ##########Can be removed, not sure if it's nice to have this
            bin_edges[0]=df_syst[column].min()
            bin_edges[-1]=df_syst[column].max()

            signal_field = df_syst[target == 1][column]
            background_field = df_syst[target == 0][column]
            del df_syst
            signal_weights = weights[target == 1]
            background_weights = weights[target == 0]

            signal_hist[i]=np.histogram(signal_field,bins=bin_edges,weights=signal_weights,density=True
                                        )[0]-signal_hist_ref
            
            bkg_hist[i]=np.histogram(background_field,bins=bin_edges,weights=background_weights,density=True
                                     )[0] - bkg_hist_ref
            del signal_weights, background_weights
        
        signal_hist_order = np.array( [
        [signal_hist[j][i] for j in range(var_lenght)] for i in range(nb_bins)
        ])
        del signal_hist
        bkg_hist_order = np.array( [
        [bkg_hist[j][i] for j in range(var_lenght)] for i in range(nb_bins)
        ])
        del bkg_hist


        from mpl_toolkits.mplot3d import Axes3D
        y_obs=[signal_hist_order,bkg_hist_order]
        y_obs_name=["signal","bkg"]
        for k in range(len(y_obs)) :
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot surface with facecolors set by normalized Z values
            surf_signal = ax.plot_surface(tes, bin_edges, y_obs[k], facecolors=plt.cm.viridis((y_obs[k] - y_obs[k].min()) / (y_obs[k].max() - y_obs[k].min())), shade=False)

            ax.set_xlabel('tes')
            ax.set_ylabel('Bins')
            ax.set_zlabel('Density shift')

            current_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists("%s/Images/FeaturesAnalysis_VS_syst"%(current_dir)):
                os.makedirs("%s/Images/FeaturesAnalysis_VS_syst"%(current_dir))
            plt.savefig("%s/Images/FeaturesAnalysis_VS_syst/FAIRuniverse_StackedHist_%s_tot_size%s.png"%(current_dir,y_obs_name[k],THV_size[0]+THV_size[1]+THV_size[2]))

            plt.close()


















def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns
