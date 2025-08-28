import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --------------------------------------  Ploteur


def Correlation_big_graph(
    dfplot, data, caption, Detailled_label, Detailled_label_type, Nb_feature
):
    from parameter_management_scan import Parameter_Distribution

    Tamp_parameter = Parameter_Distribution.get_all()
    THV_size = Tamp_parameter["THV_size"]
    ModelType = Tamp_parameter["ModelType"]

    Ax_nbr = 0
    Correlation_matrix_list = [0, 0, 0, 0]

    sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")
    fig, ax = plt.subplots(2, 3, layout="constrained", figsize=(24, 17), sharex=True)
    # fig,ax=plt.subplots(3,2, layout='constrained',figsize=(19,24),sharex=True)
    ax = ax.flatten()

    for j in range(4):
        for i in range(2):

            Bkg_Sig_dfplot = dfplot[
                data["labels"] == i
            ]  

            Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[
                data["detailed_labels"] == Detailled_label[j]
            ]
            if Detail_Sig_Bkg_dfplot.empty == False:
                Correlation_matrix_list[Ax_nbr] = Detail_Sig_Bkg_dfplot.corr()
                sns.heatmap(
                    Correlation_matrix_list[Ax_nbr],
                    vmin=-1,
                    vmax=1,
                    annot=False,
                    linewidth=0.5,
                    linecolor="black",
                    cmap="Spectral_r",
                    ax=ax[Ax_nbr],
                )  # rainbow  #Spectral_r #gnuplot
                ax[Ax_nbr].set_title(
                    "%s events considered as %s"
                    % (Detailled_label[j], caption[Detailled_label_type[Ax_nbr]]),
                    fontsize=17,
                )
                Ax_nbr += 1
    Bkg_Sig_dfplot = dfplot[data["labels"] == 0]
    Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[data["detailed_labels"] != "htautau"]
    Weighted_Correlation_matrix_mean_bkg = Detail_Sig_Bkg_dfplot.corr()
    
    sns.heatmap(
        Weighted_Correlation_matrix_mean_bkg,
        vmin=-1,
        vmax=1,
        annot=False,
        linewidth=0.5,
        cmap="Spectral_r",
        linecolor="black",
        ax=ax[4],
    )
    ax[4].set_title("Weighted mean Background Correlation matrix", fontsize=17)
    sns.heatmap(
        np.abs((Correlation_matrix_list[0] - Weighted_Correlation_matrix_mean_bkg) / 2),
        annot=False,
        linewidth=0.5,
        linecolor="black",
        cmap="afmhot_r",
        ax=ax[5],
    )
    ax[5].set_title(
        "Absolute difference between the correlation matrix of signal and weighted mean background",
        fontsize=17,
    )

    plt.suptitle("Correlation matrix of features", fontsize=27)
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("current_dir/images/FeaturesAnalysis_Corr_Distri"):
        os.makedirs("current_dir/images/FeaturesAnalysis_Corr_Distri")
    plt.savefig(
        "current_dir/images/FeaturesAnalysis_Corr_Distri/BlackSwan_CorrelationMat_DataSet_tot_size%s.png"
        % (np.sum(THV_size))
    )

    plt.show()
    del Bkg_Sig_dfplot, Detail_Sig_Bkg_dfplot, Weighted_Correlation_matrix_mean_bkg


def Correlation_diff_graph(
    dfplot,
    data,
    caption,
    Detailled_label,
    Detailled_label_type,
    detailled_label_num_ref,
    Nb_feature,
):
    from parameter_management_scan import Parameter_Distribution

    Tamp_parameter = Parameter_Distribution.get_all()
    THV_size = Tamp_parameter["THV_size"]
    ModelType = Tamp_parameter["ModelType"]

    Correlation_matrix_list = [0, 0, 0, 0]
    sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")
    fig, ax = plt.subplots(2, 3, layout="constrained", figsize=(24, 17), sharex=True)
    # fig,ax=plt.subplots(3,2, layout='constrained',figsize=(19,24),sharex=True)
    ax = ax.flatten()

    Bkg_Sig_dfplot = dfplot[
        data["labels"] == Detailled_label_type[detailled_label_num_ref]
    ]
    Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[
        data["detailed_labels"] == Detailled_label[detailled_label_num_ref]
    ]
    Correlation_matrix_list[detailled_label_num_ref] = Detail_Sig_Bkg_dfplot.corr()
    sns.heatmap(
        Correlation_matrix_list[detailled_label_num_ref],
        vmin=-1,
        vmax=1,
        annot=False,
        linewidth=0.5,
        linecolor="black",
        cmap="Spectral_r",
        ax=ax[0],
    )  # rainbow  #Spectral_r #gnuplot
    ax[1].set_title(
        "Reference: %s events considered as %s"
        % (
            Detailled_label[detailled_label_num_ref],
            caption[Detailled_label_type[detailled_label_num_ref]],
        ),
        fontsize=17,
    )

    Ax_nbr = 1
    for j in range(0, 4):
        if j != detailled_label_num_ref:

            Bkg_Sig_dfplot = dfplot[data["labels"] == Detailled_label_type[j]]
            Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[
                data["detailed_labels"] == Detailled_label[j]
            ]
            Correlation_matrix_list[j] = Detail_Sig_Bkg_dfplot.corr()
            sns.heatmap(
                np.abs(
                    Correlation_matrix_list[j]
                    - Correlation_matrix_list[detailled_label_num_ref]
                )
                / 2,
                vmin=0,
                vmax=0.5,
                annot=False,
                linewidth=0.5,
                linecolor="black",
                cmap="afmhot_r",
                ax=ax[Ax_nbr],
            )  # rainbow  #Spectral_r #gnuplot
            ax[Ax_nbr].set_title(
                "Absolute difference between the correlation matrix of %s and and %s "
                % (Detailled_label[detailled_label_num_ref], Detailled_label[j]),
                fontsize=17,
            )
            Ax_nbr += 1

    Bkg_Sig_dfplot = dfplot[data["labels"] == 0]
    Detail_Sig_Bkg_dfplot = Bkg_Sig_dfplot[
        data["detailed_labels"] != Detailled_label[detailled_label_num_ref]
    ]
    Weighted_Correlation_mean_other_bkg = Detail_Sig_Bkg_dfplot.corr()
    # Correlation_matrix_mean_bkg=(Correlation_matrix_list[1]+Correlation_matrix_list[2]+Correlation_matrix_list[3])/3
    sns.heatmap(
        np.abs(
            Weighted_Correlation_mean_other_bkg
            - Correlation_matrix_list[detailled_label_num_ref]
        )
        / 2,
        vmin=0,
        vmax=0.5,
        annot=False,
        linewidth=0.5,
        cmap="afmhot_r",
        linecolor="black",
        ax=ax[4],
    )
    ax[4].set_title(
        "Absolute difference between the correlation matrix\ of\n %s and and weighted mean of all the other background "
        % (Detailled_label[detailled_label_num_ref]),
        fontsize=17,
    )

    Detail_Sig_Bkg_dfplot = dfplot[
        data["detailed_labels"] != Detailled_label[detailled_label_num_ref]
    ]
    Weighted_Correlation_mean_other_tot = Detail_Sig_Bkg_dfplot.corr()
    sns.heatmap(
        np.abs(
            Weighted_Correlation_mean_other_tot
            - Correlation_matrix_list[detailled_label_num_ref]
        )
        / 2,
        vmin=0,
        vmax=0.5,
        annot=False,
        linewidth=0.5,
        cmap="afmhot_r",
        linecolor="black",
        ax=ax[5],
    )
    ax[5].set_title(
        "Absolute difference between the correlation matrix of\n %s and and weighted mean of all the other data "
        % (Detailled_label[detailled_label_num_ref]),
        fontsize=17,
    )

    plt.suptitle(
        "Difference between the correlation matrix of features considering as reference %s"
        % (Detailled_label[detailled_label_num_ref]),
        fontsize=27,
    )
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("current_dir/images/FeaturesAnalysis_Corr_Distri"):
        os.makedirs("current_dir/images/FeaturesAnalysis_Corr_Distri")
    plt.savefig(
        "current_dir/images/FeaturesAnalysis_Corr_Distri/BlackSwan_DiffCorrMat_ref_%s_DataSet_tot_size%s.png"
        % (Detailled_label[detailled_label_num_ref], np.sum(THV_size))
    )

    plt.show()
    del (
        Bkg_Sig_dfplot,
        Detail_Sig_Bkg_dfplot,
        Weighted_Correlation_mean_other_tot,
        Weighted_Correlation_mean_other_bkg,
    )


""" Fct for panda table and then call plotting function 
"""

def feature_correlations(data, field_name):

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
    Nb_feature = len(field_name)
    caption = ["Background_feature", "Signal_feature"]
    Detailled_label_list = ["htautau", "diboson", "ttbar", "ztautau"]
    Detailled_label_type_list = [1, 0, 0, 0]

    Correlation_big_graph(
        dfplot=dfplot,
        data=data,
        caption=caption,
        Detailled_label=Detailled_label_list,
        Detailled_label_type=Detailled_label_type_list,
        Nb_feature=Nb_feature,
    )
    for j in range(0, 4):
        Correlation_diff_graph(
            dfplot=dfplot,
            data=data,
            caption=caption,
            Detailled_label=Detailled_label_list,
            Detailled_label_type=Detailled_label_type_list,
            detailled_label_num_ref=j,
            Nb_feature=Nb_feature,
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

    if shape == "compact":
        nb_col = 5
    elif shape == "long":
        nb_col = 3
    elif shape == "verylong":  # Better tu use number
        nb_col = 2

    Nbr_field_name = len(field_name)
    fig, ax = plt.subplots(
        math.ceil(Nbr_field_name / nb_col),
        nb_col,
        layout="constrained",
        figsize=(20, (20 / nb_col) * math.ceil(Nbr_field_name / nb_col)),
    )
    ax = ax.flatten()
    for i in range(Nbr_field_name):
        field = dfall[field_name[i]]

        weight_keys = {}
        keys = np.unique(detailed_label)

        for key in keys:
            weight_keys[key] = weights[detailed_label == key]

        # print("keys", keys)
        # print("keys 2", weight_keys.keys())

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

        hist_bkg = (hist_b).copy()  # hist_b+hist_s

        color = ["purple", "orange", "cornflowerblue", "forestgreen"]
        ax[i].stairs(
            hist_s + hist_b, bins, fill=True, label="htautau sig", color=color[0]
        )
        for j, key in enumerate(keys[keys != "htautau"]):
            print("key=", key)
            hist, bins = np.histogram(
                field_clipped[detailed_labels_clipped == key],
                bins=bins,
                weights=weights_clipped[detailed_labels_clipped == key],
            )
            ax[i].stairs(
                hist_b, bins, fill=True, label=f"{key} bkg", color=color[j + 1]
            )  # hist_b
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
    plt.suptitle("Stacked Histogram for differents features", fontsize=27)
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("current_dir/images/FeaturesAnalysis_Corr_Distri"):
        os.makedirs("current_dir/images/FeaturesAnalysis_Corr_Distri")
    plt.savefig(
        "current_dir/images/FeaturesAnalysis_Corr_Distri/BlackSwan_StackedHist_%s_DataSet_tot_size%s.png"
        % (shape, THV_size[0] + THV_size[1] + THV_size[2])
    )

    # plt.xlabel(f"{field_name}")
    # plt.ylabel("Weighted count")
    # plt.yscale(y_scale)
    plt.show()


def features_systematics_dependence(
    dfall, systematics, columns, nb_bins=20, var_lenght=100, n_jobs=5
):

    import os
    import matplotlib.pyplot as plt
    import logging
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from joblib import Parallel, delayed

    tes = np.linspace(0.9, 1.1, var_lenght)
    jes = tes
    soft_met = np.linspace(0, 5, var_lenght)
    systematics_list_TJS_type = [tes, jes, soft_met]
    systematics_TJS_list_name = ["TES", "JES", "SOFTMET"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/Images/FeaturesAnalysis_VS_syst" % (current_dir)):
        os.makedirs("%s/Images/FeaturesAnalysis_VS_syst" % (current_dir))

    """
    Plots heatmaps of the dataset features vs systematics (parallelized version).

    Args:
        * columns (list): The list of column names to consider (default: None, which includes all columns).
        * nb_bins (int): The number of bins for the histogram (default: 20).
        * var_lenght (int): The number of systematic variation points (default: 100).
        * n_jobs (int): Number of parallel jobs (-1 uses all available cores).

    .. Image:: images/heatmap_datasets.png
    """

    from parameter_management_scan import Parameter_Distribution

    Tamp_parameter = Parameter_Distribution.get_all()
    THV_size = Tamp_parameter["THV_size"]
    ModelType = Tamp_parameter["ModelType"]

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
            if col not in columns:  # maybe just columns
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for heatmap plotting.")

    # Set seaborn theme for better plots
    sns.set_theme(rc={"figure.figsize": (12, 8)}, style="whitegrid")

    # Get reference data
    dfall_tamp = dfall.copy()
    dfall_poscut_ref = systematics(dfall_tamp)
    del dfall_tamp

    labels_ref = dfall_poscut_ref["labels"]
    weights_ref = dfall_poscut_ref["weights"]
    data_ref = dfall_poscut_ref["data"]
    del dfall_poscut_ref
    middle = False

    def process_single_systematic_variation(
        i,
        syst_value,
        syst_idx,
        column,
        bin_edges,
        signal_hist_ref,
        bkg_hist_ref,
        dfall_copy,
    ):
        """
        Process a single systematic variation point.

        Args:
            i: Index of the systematic variation
            syst_value: Value of the systematic parameter
            syst_idx: Index of systematic type (0=TES, 1=JES, 2=SOFTMET)
            column: Column name to process
            bin_edges: Histogram bin edges
            signal_hist_ref: Reference signal histogram
            bkg_hist_ref: Reference background histogram
            dfall_copy: Copy of the original dataframe

        Returns:
            tuple: (i, signal_hist_diff, bkg_hist_diff)
        """
        try:
            print(f"systematics: {syst_value} pour {column}")
            middle = False
            # Apply systematic variations
            if syst_idx == 0:  # TES
                dfall_poscut = systematics(dfall_copy, tes=syst_value)
                # if i==(np.ceil(nb_bins/2)-1):
                #     middle=False
                #     print("Middle tes")

            elif syst_idx == 1:  # JES
                dfall_poscut = systematics(dfall_copy, jes=syst_value)
                # if i==(np.ceil(nb_bins/2)-1):
                #     middle=False
                #     print("Middle jes")

            else:  # SOFTMET
                dfall_poscut = systematics(dfall_copy, soft_met=syst_value)
                # if i==0:
                #     middle=False
                #     print("Middle soft met")

            labels = dfall_poscut["labels"]
            weights = dfall_poscut["weights"]
            data = dfall_poscut["data"]
            del dfall_poscut

            # Update bin edges if necessary (optional)
            colum_min = data[column].min()
            colum_max = data[column].max()
            updated_bin_edges = bin_edges.copy()
            if colum_min < bin_edges[0]:
                updated_bin_edges[0] = colum_min
            if colum_max > bin_edges[-1]:
                updated_bin_edges[-1] = colum_max

            # Calculate histograms for this systematic variation
            signal_field = data[column][labels == 1]
            background_field = data[column][labels == 0]
            signal_weights = weights[labels == 1]
            background_weights = weights[labels == 0]

            signal_hist = np.histogram(
                signal_field,
                bins=updated_bin_edges,
                weights=signal_weights,
                density=True,
            )[0]
            bkg_hist = np.histogram(
                background_field,
                bins=updated_bin_edges,
                weights=background_weights,
                density=True,
            )[0]

            # Calculate differences from reference
            # if middle :
            #     signal_hist_diff =  signal_hist_ref
            #     bkg_hist_diff =  bkg_hist_ref
            # else :
            signal_hist_diff = signal_hist - signal_hist_ref
            bkg_hist_diff = bkg_hist - bkg_hist_ref

            return i, signal_hist_diff, bkg_hist_diff

        except Exception as e:
            logger.error(f"Error processing systematic variation {i}: {e}")
            return i, np.zeros_like(signal_hist_ref), np.zeros_like(bkg_hist_ref)

    for h, column in enumerate(columns):
        print(f"[*] --- {column} heatmap (parallelized)")

        # Determine the combined range for the current column
        min_value = data_ref[column].min()
        max_value = data_ref[column].max()
        print("min value:", min_value, "  max value:", max_value)

        # Define the bin edges
        bin_edges = np.linspace(min_value, max_value, nb_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate reference histograms
        signal_field_ref = data_ref[column][labels_ref == 1]
        background_field_ref = data_ref[column][labels_ref == 0]
        signal_weights = weights_ref[labels_ref == 1]
        background_weights = weights_ref[labels_ref == 0]

        signal_hist_ref = np.histogram(
            signal_field_ref, bins=bin_edges, weights=signal_weights, density=True
        )[0]
        bkg_hist_ref = np.histogram(
            background_field_ref,
            bins=bin_edges,
            weights=background_weights,
            density=True,
        )[0]
        del signal_field_ref, background_field_ref, signal_weights, background_weights

        # Create subplots for heatmaps
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Track global min/max for consistent color scaling
        global_min = float("inf")
        global_max = float("-inf")
        # global_min = min(signal_hist_ref.min(),bkg_hist_ref.min())
        # global_max = max(signal_hist_ref.max(),bkg_hist_ref.max())
        # Track global min/max for consistent color scaling
        # local_min=min(signal_hist_ref.min(),bkg_hist_ref.min())
        # local_max=max(signal_hist_ref.max(),bkg_hist_ref.max())
        # global_min = -max(abs(local_min),local_max)
        # global_max=max(abs(local_min),local_max)

        # Store all heatmap data for consistent scaling
        all_heatmaps = []

        for j, syst_type in enumerate(systematics_list_TJS_type):
            print(
                f"Processing systematic: {systematics_TJS_list_name[j]} (parallelized)"
            )

            # Prepare arguments for parallel processing
            parallel_args = []
            for i in range(var_lenght):
                dfall_copy = dfall.copy()  # Each worker gets its own copy
                parallel_args.append(
                    (
                        i,
                        syst_type[i],
                        j,
                        column,
                        bin_edges,
                        signal_hist_ref,
                        bkg_hist_ref,
                        dfall_copy,
                    )
                )

            # Process systematic variations in parallel
            results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(process_single_systematic_variation)(*args)
                for args in parallel_args
            )

            # Sort results by index and extract histogram differences
            results.sort(key=lambda x: x[0])

            # Initialize arrays to store histogram differences
            # Shape: (nb_bins, var_lenght) where rows=bins, columns=systematic_values
            signal_hist_diff = np.zeros((nb_bins, var_lenght))
            bkg_hist_diff = np.zeros((nb_bins, var_lenght))

            # Fill arrays with results
            for i, sig_diff, bkg_diff in results:
                # if j == 0 & i==(np.ceil(nb_bins/2)-1):  # TES
                #     signal_hist_diff[:, i] = signal_hist_ref
                #     bkg_hist_diff[:, i] = bkg_hist_ref
                #     print("Middle tes")

                # elif j == 1 & i==(np.ceil(nb_bins/2)-1):  # JES
                #     signal_hist_diff[:, i] = signal_hist_ref
                #     bkg_hist_diff[:, i] = bkg_hist_ref
                #     print("Middle jes")

                # elif j==2 & i==0:  # SOFTMET
                #     signal_hist_diff[:, i] = signal_hist_ref
                #     bkg_hist_diff[:, i] = bkg_hist_ref
                #     print("Middle soft met")

                # else :
                signal_hist_diff[:, i] = sig_diff
                bkg_hist_diff[:, i] = bkg_diff

            # Update global min/max
            global_min = min(global_min, signal_hist_diff.min(), bkg_hist_diff.min())
            global_max = max(global_max, signal_hist_diff.max(), bkg_hist_diff.max())

            # Store heatmap data
            all_heatmaps.extend([signal_hist_diff, bkg_hist_diff])

        # Create heatmaps with consistent color scale
        vmin, vmax = global_min, global_max
        from matplotlib.colors import TwoSlopeNorm

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        for j, syst_type in enumerate(systematics_list_TJS_type):
            signal_hist_diff = all_heatmaps[2 * j]
            bkg_hist_diff = all_heatmaps[2 * j + 1]

            # Create individual heatmaps
            for k, (hist_data, label) in enumerate(
                [(signal_hist_diff, "signal"), (bkg_hist_diff, "bkg")]
            ):
                # Individual heatmap (commented out as in your code)
                # fig_ind, ax_ind = plt.subplots(figsize=(12, 8))

                # # Plot heatmap with correct orientation
                # # hist_data shape is (nb_bins, var_lenght)
                # # imshow plots with (0,0) at top-left, so we use origin='lower' to flip y-axis
                #   im = ax_ind.imshow(hist_data, aspect='auto', cmap='RdBu_r',
                #                   vmin=vmin, vmax=vmax, origin='lower')
                # #im = ax_ind.imshow(hist_data, aspect='auto', cmap='RdBu_r',
                # #                  norm=norm, origin='lower')

                # # Set labels and ticks
                # ax_ind.set_xlabel(f'{systematics_TJS_list_name[j]} Variation')
                # ax_ind.set_ylabel('Bin Number')
                # ax_ind.set_title(f'{systematics_TJS_list_name[j]} - {label} - {column}')

                # # Set x-axis ticks to show systematic values
                # x_ticks = np.linspace(0, var_lenght-1, 5).astype(int)
                # x_labels = [f'{syst_type[idx]:.3f}' for idx in x_ticks]
                # ax_ind.set_xticks(x_ticks)
                # ax_ind.set_xticklabels(x_labels)

                # # Set y-axis ticks to show bin numbers (1 to nb_bins)
                # y_ticks = np.linspace(0, nb_bins-1, min(nb_bins, 10)).astype(int)
                # y_labels = [f'{idx+1}' for idx in y_ticks]  # Bin numbers start from 1
                # ax_ind.set_yticks(y_ticks)
                # ax_ind.set_yticklabels(y_labels)

                # # Add colorbar
                # cbar = plt.colorbar(im, ax=ax_ind)
                # cbar.set_label('Density Shift')

                # plt.tight_layout()
                # plt.savefig(f"{current_dir}/Images/FeaturesAnalysis_VS_syst/{column}_{systematics_TJS_list_name[j]}_{label}_heatmap_size{THV_size[0]+THV_size[1]+THV_size[2]}.png",
                #            dpi=150, bbox_inches='tight')
                # plt.close(fig_ind)

                # Add to combined plot

                # im_combined = axes[2*j + k].imshow(hist_data, aspect='auto', cmap='RdBu_r',
                #                                   vmin=vmin, vmax=vmax, origin='lower')
                im_combined = axes[2 * j + k].imshow(
                    hist_data, aspect="auto", cmap="RdBu_r", norm=norm, origin="lower"
                )

                axes[2 * j + k].set_xlabel(f"{systematics_TJS_list_name[j]} Variation")
                axes[2 * j + k].set_ylabel("Bin Number")
                axes[2 * j + k].set_title(f"{systematics_TJS_list_name[j]} - {label}")

                # Set x-axis ticks to show systematic values
                x_ticks = np.linspace(0, var_lenght - 1, 5).astype(int)
                x_labels = [f"{syst_type[idx]:.3f}" for idx in x_ticks]
                axes[2 * j + k].set_xticks(x_ticks)
                axes[2 * j + k].set_xticklabels(x_labels, rotation=45)

                # Set y-axis ticks to show bin numbers (1 to nb_bins)
                y_ticks = np.linspace(0, nb_bins - 1, min(nb_bins, 10)).astype(int)
                y_labels = [f"{idx+1}" for idx in y_ticks]  # Bin numbers start from 1
                axes[2 * j + k].set_yticks(y_ticks)
                axes[2 * j + k].set_yticklabels(y_labels)

        # Add colorbars to combined plot
        for i in range(6):
            cbar = fig.colorbar(im_combined, ax=axes[i], shrink=0.8)
            cbar.set_label("Relative Density Shift over the unbiaised value")

        fig.suptitle(f"{column} - Systematic Variations Heatmaps", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            f"{current_dir}/Images/FeaturesAnalysis_VS_syst/{column}_combined_heatmaps_size{THV_size[0]+THV_size[1]+THV_size[2]}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"[*] Completed heatmaps for {column} (parallelized)")


def features_systematics_dependence_new_old(
    dfall, systematics, columns, nb_bins=20, var_lenght=100
):

    import os
    import matplotlib.pyplot as plt
    import logging
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    tes = np.linspace(0.9, 1.1, var_lenght)
    jes = tes
    soft_met = np.linspace(0, 5, var_lenght)
    systematics_list_TJS_type = [tes, jes, soft_met]
    systematics_TJS_list_name = ["TES", "JES", "SOFTMET"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/Images/FeaturesAnalysis_VS_syst" % (current_dir)):
        os.makedirs("%s/Images/FeaturesAnalysis_VS_syst" % (current_dir))

    """
    Plots heatmaps of the dataset features vs systematics.

    Args:
        * columns (list): The list of column names to consider (default: None, which includes all columns).
        * nb_bins (int): The number of bins for the histogram (default: 20).
        * var_lenght (int): The number of systematic variation points (default: 100).

    .. Image:: images/heatmap_datasets.png
    """

    from parameter_management_scan import Parameter_Distribution

    Tamp_parameter = Parameter_Distribution.get_all()
    THV_size = Tamp_parameter["THV_size"]
    ModelType = Tamp_parameter["ModelType"]

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
            if col not in columns:  # maybe just columns
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for heatmap plotting.")

    # Set seaborn theme for better plots
    sns.set_theme(rc={"figure.figsize": (12, 8)}, style="whitegrid")

    # Get reference data
    dfall_tamp = dfall.copy()
    dfall_poscut_ref = systematics(dfall_tamp)
    del dfall_tamp

    labels_ref = dfall_poscut_ref["labels"]
    weights_ref = dfall_poscut_ref["weights"]
    data_ref = dfall_poscut_ref["data"]
    del dfall_poscut_ref

    for h, column in enumerate(columns):
        print(f"[*] --- {column} heatmap")

        # Determine the combined range for the current column
        min_value = data_ref[column].min()
        max_value = data_ref[column].max()
        print("min value:", min_value, "  max value:", max_value)

        # Define the bin edges
        bin_edges = np.linspace(min_value, max_value, nb_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate reference histograms
        signal_field_ref = data_ref[column][labels_ref == 1]
        background_field_ref = data_ref[column][labels_ref == 0]
        signal_weights = weights_ref[labels_ref == 1]
        background_weights = weights_ref[labels_ref == 0]

        signal_hist_ref = np.histogram(
            signal_field_ref, bins=bin_edges, weights=signal_weights, density=True
        )[0]
        bkg_hist_ref = np.histogram(
            background_field_ref,
            bins=bin_edges,
            weights=background_weights,
            density=True,
        )[0]
        del signal_field_ref, background_field_ref, signal_weights, background_weights

        # Create subplots for heatmaps
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Track global min/max for consistent color scaling
        global_min = float("inf")
        global_max = float("-inf")

        # Store all heatmap data for consistent scaling
        all_heatmaps = []

        for j, syst_type in enumerate(systematics_list_TJS_type):
            print(f"Processing systematic: {systematics_TJS_list_name[j]}")

            """
            Initialize arrays to store histogram differences
            Shape: (nb_bins, var_lenght) where rows=bins, columns=systematic_values
            """
            signal_hist_diff = np.zeros((nb_bins, var_lenght))
            bkg_hist_diff = np.zeros((nb_bins, var_lenght))

            for i in range(var_lenght):
                dfall_tamp = dfall.copy()

                """
                Apply systematic variations
                """
                if j == 0:  # TES
                    print(f"systematics: {syst_type[i]} pour {column}")
                    dfall_poscut = systematics(dfall_tamp, tes=syst_type[i])
                elif j == 1:  # JES
                    print(f"systematics: {syst_type[i]} pour {column}")
                    dfall_poscut = systematics(dfall_tamp, jes=syst_type[i])
                else:  # SOFTMET
                    print(f"systematics: {syst_type[i]} pour {column}")
                    dfall_poscut = systematics(dfall_tamp, soft_met=syst_type[i])

                del dfall_tamp

                labels = dfall_poscut["labels"]
                weights = dfall_poscut["weights"]
                data = dfall_poscut["data"]
                del dfall_poscut

                """
                Update bin edges if necessary (optional)
                """
                colum_min = data[column].min()
                colum_max = data[column].max()
                if colum_min < bin_edges[0]:
                    bin_edges[0] = colum_min
                if colum_max > bin_edges[-1]:
                    bin_edges[-1] = colum_max

                """
                Calculate histograms for this systematic variation
                """
                signal_field = data[column][labels == 1]
                background_field = data[column][labels == 0]
                signal_weights = weights[labels == 1]
                background_weights = weights[labels == 0]

                signal_hist = np.histogram(
                    signal_field, bins=bin_edges, weights=signal_weights, density=True
                )[0]
                bkg_hist = np.histogram(
                    background_field,
                    bins=bin_edges,
                    weights=background_weights,
                    density=True,
                )[0]

                """
                Store differences from reference
                """
                signal_hist_diff[:, i] = signal_hist - signal_hist_ref
                bkg_hist_diff[:, i] = bkg_hist - bkg_hist_ref

                del signal_weights, background_weights

            """
            Update global min/max
            """
            global_min = min(global_min, signal_hist_diff.min(), bkg_hist_diff.min())
            global_max = max(global_max, signal_hist_diff.max(), bkg_hist_diff.max())

            """
            Store heatmap data
            """
            all_heatmaps.extend([signal_hist_diff, bkg_hist_diff])

        """
        Create heatmaps with consistent color scale
        """
        vmin, vmax = global_min, global_max

        for j, syst_type in enumerate(systematics_list_TJS_type):
            signal_hist_diff = all_heatmaps[2 * j]
            bkg_hist_diff = all_heatmaps[2 * j + 1]

            """
            Create individual heatmaps
            """
            for k, (hist_data, label) in enumerate(
                [(signal_hist_diff, "signal"), (bkg_hist_diff, "bkg")]
            ):
                # Individual heatmap
                # fig_ind, ax_ind = plt.subplots(figsize=(12, 8))

                # # Plot heatmap with correct orientation
                # # hist_data shape is (nb_bins, var_lenght)
                # # imshow plots with (0,0) at top-left, so we use origin='lower' to flip y-axis
                # im = ax_ind.imshow(hist_data, aspect='auto', cmap='RdBu_r',
                #                   vmin=vmin, vmax=vmax, origin='lower')

                # # Set labels and ticks
                # ax_ind.set_xlabel(f'{systematics_TJS_list_name[j]} Variation')
                # ax_ind.set_ylabel('Bin Number')
                # ax_ind.set_title(f'{systematics_TJS_list_name[j]} - {label} - {column}')

                # # Set x-axis ticks to show systematic values
                # x_ticks = np.linspace(0, var_lenght-1, 5).astype(int)
                # x_labels = [f'{syst_type[idx]:.3f}' for idx in x_ticks]
                # ax_ind.set_xticks(x_ticks)
                # ax_ind.set_xticklabels(x_labels)

                # # Set y-axis ticks to show bin numbers (1 to nb_bins)
                # y_ticks = np.linspace(0, nb_bins-1, min(nb_bins, 10)).astype(int)
                # y_labels = [f'{idx+1}' for idx in y_ticks]  # Bin numbers start from 1
                # ax_ind.set_yticks(y_ticks)
                # ax_ind.set_yticklabels(y_labels)

                # # Add colorbar
                # cbar = plt.colorbar(im, ax=ax_ind)
                # cbar.set_label('Density Shift')

                # plt.tight_layout()
                # plt.savefig(f"{current_dir}/Images/FeaturesAnalysis_VS_syst/{column}_{systematics_TJS_list_name[j]}_{label}_heatmap_size{THV_size[0]+THV_size[1]+THV_size[2]}.png",
                #            dpi=150, bbox_inches='tight')
                # plt.close(fig_ind)

                # Add to combined plot
                im_combined = axes[2 * j + k].imshow(
                    hist_data,
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                )

                axes[2 * j + k].set_xlabel(f"{systematics_TJS_list_name[j]} Variation")
                axes[2 * j + k].set_ylabel("Bin Number")
                axes[2 * j + k].set_title(f"{systematics_TJS_list_name[j]} - {label}")


                axes[2 * j + k].set_xticks(x_ticks)
                axes[2 * j + k].set_xticklabels(x_labels, rotation=45)
                axes[2 * j + k].set_yticks(y_ticks)
                axes[2 * j + k].set_yticklabels(y_labels)

        """
        Add colorbars to combined plot
        """
        for i in range(6):
            cbar = fig.colorbar(im_combined, ax=axes[i], shrink=0.8)
            cbar.set_label("Density Shift")

        fig.suptitle(f"{column} - Systematic Variations Heatmaps", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            f"{current_dir}/Images/FeaturesAnalysis_VS_syst/{column}_combined_heatmaps_size{THV_size[0]+THV_size[1]+THV_size[2]}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"[*] Completed heatmaps for {column}")


def features_systematics_dependence_old(
    dfall, systematics, columns, nb_bins=20, var_lenght=100
):

    import os
    import matplotlib.pyplot as plt
    import logging
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    tes = np.linspace(0.9, 1.1, var_lenght)
    jes = tes
    soft_met = np.linspace(0, 5, var_lenght)
    systematics_list_TJS_type = [tes, jes, soft_met]
    systematics_TJS_list_name = ["TES", "JES", "SOFTMET"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists("%s/Images/FeaturesAnalysis_VS_syst" % (current_dir)):
        os.makedirs("%s/Images/FeaturesAnalysis_VS_syst" % (current_dir))

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
            if col not in columns:  # maybee just columns
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")

    # sns.set_theme(style="whitegrid")
    sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")

    dfall_tamp = dfall.copy()
    dfall_poscut_ref = systematics(dfall_tamp)
    del dfall_tamp

    labels_ref = dfall_poscut_ref["labels"]
    weights_ref = dfall_poscut_ref["weights"]
    data_ref = dfall_poscut_ref["data"]
    del dfall_poscut_ref

    """
    # Number of rows and columns in the subplot grid
    n_cols = 2  # Number of columns in the subplot grid
    n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 6 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing
    """

    for h, column in enumerate(columns):
        # Determine the combined range for the current column

        print(f"[*] --- {column} histogram")
        min_value = data_ref[column].min()
        max_value = data_ref[column].max()

        print("min value :", min_value, "  max value:", max_value)

        """
        Define the bin edges
        """
        bin_edges = np.linspace(
            min_value, max_value, nb_bins + 1
        )  

        ### Add list of min max to remove ref
        signal_field_ref = data_ref[column][labels_ref == 1]
        background_field_ref = data_ref[column][labels_ref == 0]
        signal_weights = weights_ref[labels_ref == 1]
        background_weights = weights_ref[labels_ref == 0]

        signal_hist_ref = np.histogram(
            signal_field_ref, bins=bin_edges, weights=signal_weights, density=True
        )[0]
        bkg_hist_ref = np.histogram(
            background_field_ref,
            bins=bin_edges,
            weights=background_weights,
            density=True,
        )[0]
        del signal_field_ref, background_field_ref, signal_weights, background_weights

        figu, axi = plt.subplots(
            2, 3, figsize=(17, 20), subplot_kw={"projection": "3d"}
        )
        axi = axi.flatten()
        y_obs_absmin = 1000
        y_obs_absmax = 1000
        for j, syst_type in enumerate(systematics_list_TJS_type):
            signal_hist = [None] * var_lenght
            bkg_hist = [None] * var_lenght
            for i in range(var_lenght):
                dfall_tamp = dfall.copy()
                if j == 0:
                    print("systematics :", syst_type[i], " pour ", column)
                    dfall_poscut = systematics(dfall_tamp, tes=syst_type[i])
                elif j == 1:
                    print("systematics :", syst_type[i], " pour ", column)
                    dfall_poscut = systematics(dfall_tamp, jes=syst_type[i])
                else:
                    print("systematics :", syst_type[i], " pour ", column)
                    dfall_poscut = systematics(dfall_tamp, soft_met=syst_type[i])
                del dfall_tamp
                labels = dfall_poscut["labels"]
                weights = dfall_poscut["weights"]
                data = dfall_poscut["data"]
                del dfall_poscut


                # Can be removed, not sure if it's nice to have this
                colum_min = data[column].min()
                colum_max = data[column].max()
                if colum_min < bin_edges[0]:
                    bin_edges[0] = data[column].min()
                if colum_max > bin_edges[-1]:
                    bin_edges[-1] = data[column].max()

                signal_field = data[column][labels == 1]
                background_field = data[column][labels == 0]
                signal_weights = weights[labels == 1]
                background_weights = weights[labels == 0]

                signal_hist[i] = (
                    np.histogram(
                        signal_field,
                        bins=bin_edges,
                        weights=signal_weights,
                        density=True,
                    )[0]
                    - signal_hist_ref
                )

                bkg_hist[i] = (
                    np.histogram(
                        background_field,
                        bins=bin_edges,
                        weights=background_weights,
                        density=True,
                    )[0]
                    - bkg_hist_ref
                )
                del signal_weights, background_weights

            signal_hist_order = np.array(
                [[signal_hist[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
            )
            del signal_hist
            bkg_hist_order = np.array(
                [[bkg_hist[j][i] for j in range(var_lenght)] for i in range(nb_bins)]
            )
            del bkg_hist

            hist_min = min(np.min(signal_hist_order), np.min(bkg_hist_order))
            hist_max = max(np.max(signal_hist_order), np.max(bkg_hist_order))

            if j == 0 or hist_min < y_obs_absmin:
                y_obs_absmin = hist_min
            if j == 0 or hist_max > y_obs_absmax:
                y_obs_absmax = hist_max

            from mpl_toolkits.mplot3d import Axes3D

            y_obs = [signal_hist_order, bkg_hist_order]
            y_obs_name = ["signal", "bkg"]
            for k in range(len(y_obs)):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                mid_bins = np.linspace(1, nb_bins, nb_bins)
                X, Y = np.meshgrid(
                    systematics_list_TJS_type[j], mid_bins
                )  # Meshgrid with correct shapes
                Z = y_obs[k]

                print("X shape:", X.shape)
                print("Y shape:", Y.shape)
                print("Z shape:", Z.shape)
                print("Z min/max:", Z.min(), Z.max())

                # Plot surface with facecolors set by normalized Z values
                normalised_color = (y_obs[k] - y_obs[k].min()) / (
                    y_obs[k].max() - y_obs[k].min() + 1e-8
                )
                mappable = cm.ScalarMappable(cmap="viridis")
                mappable.set_array(normalised_color)

                surf_signal = ax.plot_surface(
                    X, Y, Z, facecolors=plt.cm.viridis(normalised_color), shade=False
                )

                ax.set_xlabel("tes")
                ax.set_ylabel("Bins")
                ax.set_zlabel("Density shift")
                ax.view_init(elev=90, azim=-90)
                ax.grid(True)
                ax.set_title(
                    "%s pour %s" % (systematics_TJS_list_name[j], y_obs_name[k])
                )
                fig.colorbar(
                    mappable, ax=ax, location="right", shrink=0.6, label="Density shift"
                )

                plt.savefig(
                    "%s/Images/FeaturesAnalysis_VS_syst/%s_%s_%s_FAIRuniverse_StackedHist_tot_size%s.png"
                    % (
                        current_dir,
                        columns[h],
                        systematics_TJS_list_name[j],
                        y_obs_name[k],
                        THV_size[0] + THV_size[1] + THV_size[2],
                    )
                )

                plt.close(fig)

                surf_signal_bis = axi[2 * j + k].plot_surface(
                    X,
                    Y,
                    Z,
                    facecolors=plt.cm.viridis(
                        (y_obs[k] - y_obs[k].min())
                        / (y_obs[k].max() - y_obs[k].min() + 1e-8)
                    ),
                    shade=False,
                )
                axi[2 * j + k].set_xlabel("tes")
                axi[2 * j + k].set_ylabel("Bins")
                axi[2 * j + k].set_zlabel("Density shift")
                axi[2 * j + k].view_init(elev=90, azim=-90)
                axi[2 * j + k].grid(True)
                axi[2 * j + k].set_title(
                    "%s pour %s" % (systematics_TJS_list_name[j], y_obs_name[k])
                )

        for j, syst_type in enumerate(systematics_list_TJS_type):
            for k in range(len(y_obs)):
                normalised_color = (y_obs[k] - y_obs_absmin) / (
                    y_obs_absmax - y_obs_absmin + 1e-8
                )
                mappable = cm.ScalarMappable(cmap="viridis")
                mappable.set_array(normalised_color)
                figu.colorbar(
                    mappable,
                    ax=axi[2 * j + k],
                    location="right",
                    shrink=0.6,
                    label="Density shift",
                )

        figu.suptitle("%s" % (columns[h]), fontsize=16)
        plt.savefig(
            "%s/Images/FeaturesAnalysis_VS_syst/%s_BigPlot_FAIRuniverse_StackedHist_tot_size%s.png"
            % (current_dir, columns[h], THV_size[0] + THV_size[1] + THV_size[2])
        )

        plt.close(figu)


def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns
