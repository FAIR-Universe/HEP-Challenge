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


from parameter_management_scan import Parameter_Distribution

Tamp_parameter = Parameter_Distribution.get_all()
THV_size = Tamp_parameter["THV_size"]
ModelType = Tamp_parameter["ModelType"]


def histogram_dataset(dfall, weights, columns=None, nbin=25):
    """
    Plots histograms of the dataset features.

    Args:
        * columns (list): The list of column names to consider (default: None, which includes all columns).
        * nbin (int): The number of bins for the histogram (default: 25).

    .. Image:: images/histogram_datasets.png
    """

    if columns is None:
        columns = columns
    else:
        for col in columns:
            if col not in columns:
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")

    sns.set_theme(style="whitegrid")

    df = pd.DataFrame(dfall, columns=columns)

    # Number of rows and columns in the subplot grid
    n_cols = 2  # Number of columns in the subplot grid
    n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 6 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

    for i, column in enumerate(columns):
        # Determine the combined range for the current column

        print(f"[*] --- {column} histogram")

        lower_percentile = 0
        upper_percentile = 97.5

        lower_bound = np.percentile(df[column], lower_percentile)
        upper_bound = np.percentile(df[column], upper_percentile)

        df_clipped = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        weights_clipped = weights[
            (df[column] >= lower_bound) & (df[column] <= upper_bound)
        ]
        target_clipped = target[
            (df[column] >= lower_bound) & (df[column] <= upper_bound)
        ]

        min_value = df_clipped[column].min()
        max_value = df_clipped[column].max()

        # Define the bin edges
        bin_edges = np.linspace(min_value, max_value, nbin + 1)

        signal_field = df_clipped[target_clipped == 1][column]
        background_field = df_clipped[target_clipped == 0][column]
        signal_weights = weights_clipped[target_clipped == 1]
        background_weights = weights_clipped[target_clipped == 0]

        # Plot the histogram for label == 1 (Signal)
        axes[i].hist(
            signal_field,
            bins=bin_edges,
            alpha=0.4,
            color="blue",
            label="Signal",
            weights=signal_weights,
            density=True,
        )

        axes[i].hist(
            background_field,
            bins=bin_edges,
            alpha=0.4,
            color="red",
            label="Background",
            weights=background_weights,
            density=True,
        )

        # Set titles and labels
        axes[i].set_title(f"{column}", fontsize=16)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Density")

        # Add a legend to each subplot
        axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.savefig(
        "images/%s_histogram_T=%s_H=%s_V=%s.png"
        % (ModelType, THV_size[0], THV_size[1], THV_size[2])
    )
    plt.show()


def roc_curve_wrapper(score, labels, weights, plot_label="model", color="b", lw=2):
    """
    Plots the ROC curve.

    Args:
        * score (ndarray): The score.
        * labels (ndarray): The labels.
        * weights (ndarray): The weights.
        * plot_label (str, optional): The plot label. Defaults to "model".
        * color (str, optional): The color. Defaults to "b".
        * lw (int, optional): The line width. Defaults to 2.

    .. Image:: images/roc_curve.png
    """

    auc = roc_auc_score(y_true=labels, y_score=score, sample_weight=weights)

    plt.figure(figsize=(8, 7))

    fpr, tpr, _ = roc_curve(y_true=labels, y_score=score, sample_weight=weights)
    plt.plot(fpr, tpr, color=color, lw=lw, label=plot_label + " AUC :" + f"{auc:.3f}")

    plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    plt.show()
    plt.savefig(
        "images/%s_ROCcurve_T=%s_H=%s_V=%s.png"
        % (ModelType, THV_size[0], THV_size[1], THV_size[2])
    )

    plt.close()


def statistical_subset_info(data_set, data_name):
    print(" ~~~~~~~~~~~~ ")
    print("For %s \n" % (data_name))
    print("Training Data: ", data_set["data"].shape)
    print("Training Labels: ", data_set["labels"].shape)
    print("Training Weights: ", data_set["weights"].shape)
    number_sig_events_row = (data_set["data"][data_set["labels"] == 1].shape)[0]
    number_bkg_events_row = (data_set["data"][data_set["labels"] == 0].shape)[0]
    sum_bkg = data_set["weights"][data_set["labels"] == 0].sum()
    sum_sig = data_set["weights"][data_set["labels"] == 1].sum()
    print("Nbr of signal events: ", number_sig_events_row)
    print(
        "sum_signal_weights: ",
        sum_sig,
    )
    mean_sig = sum_sig / number_sig_events_row
    print("Mean signal value: ", mean_sig)
    devia_sig = np.sqrt(
        np.sum(np.power(data_set["weights"][data_set["labels"] == 1], 2))
    )
    print(
        "Standard deviation for signal: ",
        devia_sig,
        " which represents: ",
        (devia_sig / sum_sig) * 100,
        "% of the sum of sig",
    )
    print("Sig_effective/sig: ", 1 / (1 + ((devia_sig**2) / (mean_sig**2))))

    print("\n")

    print("Nbr of bkg events: ", number_bkg_events_row)
    print(
        "sum_bkg_weights: ",
        sum_bkg,
    )
    mean_bkg = sum_bkg / number_sig_events_row
    print("Mean bkg value: ", mean_bkg)
    devia_bkg = np.sqrt(
        np.sum(np.power(data_set["weights"][data_set["labels"] == 0], 2))
    )
    print(
        "Standard deviation for signal: ",
        devia_bkg,
        " which represents: ",
        (devia_bkg / sum_bkg) * 100,
        "% of the sum of bkg",
    )
    print("Bkg_effective/bkg: ", 1 / (1 + ((devia_bkg**2) / (mean_bkg**2))))
    print("\n")
    print(" ~~~~~~~~~~~~\n ")
