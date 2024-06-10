import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn for nice plot quicker
from sklearn.metrics import roc_curve
from IPython.display import display
from sklearn.metrics import roc_auc_score


class Dataset_visualise:
    """
    A class for visualizing datasets.

    Parameters:
    - data_set (dict): The dataset containing the data, labels, weights, and detailed labels.
    - name (str): The name of the dataset (default: "dataset").
    - columns (list): The list of column names to consider (default: None, which includes all columns).

    Attributes:
    - dfall (DataFrame): The dataset.
    - target (Series): The labels.
    - weights (Series): The weights.
    - detailed_label (ndarray): The detailed labels.
    - columns (list): The list of column names.
    - name (str): The name of the dataset.
    - keys (ndarray): The unique detailed labels.
    - weight_keys (dict): The weights for each detailed label.

    Methods:
    - examine_dataset(): Prints information about the dataset.
    - histogram_dataset(columns=None): Plots histograms of the dataset features.
    - correlation_plots(columns=None): Plots correlation matrices of the dataset features.
    - pair_plots(sample_size=10, columns=None): Plots pair plots of the dataset features.
    - stacked_histogram(field_name, mu_hat=1.0, bins=30): Plots a stacked histogram of a specific field in the dataset.
    - pair_plots_syst(df_syst, sample_size=10): Plots pair plots between the dataset and a system dataset.
    """

    def __init__(self, data_set, name="dataset", columns=None):
        self.dfall = data_set["data"]
        self.target = data_set["labels"]
        self.weights = data_set["weights"]
        self.detailed_label = np.array(data_set["detailed_labels"])
        if columns == None:
            self.columns = self.dfall.columns
        else:
            self.columns = columns
        self.name = name
        self.keys = np.unique(self.detailed_label)
        self.weight_keys = {}
        for key in self.keys:
            self.weight_keys[key] = self.weights[self.detailed_label == key]

    def examine_dataset(self):
        """
        Prints information about the dataset.
        """
        print(f"[*] --- Dataset name : {self.name}")
        print(f"[*] --- Number of events : {self.dfall.shape[0]}")
        print(f"[*] --- Number of features : {self.dfall.shape[1]}")

        for key in self.keys:
            print("  ", key, " ", self.weight_keys[key].sum())

        print(
            f"[*] --- Number of signal events : {self.dfall[self.target==1].shape[0]}"
        )
        print(
            f"[*] --- Number of background events : {self.dfall[self.target==0].shape[0]}"
        )

        print("[*] --- Examples of all features")
        display(self.dfall.head())

        print("[*] --- Description of all features")
        display(self.dfall.describe())

    def histogram_dataset(self, columns=None):
        """
        Plots histograms of the dataset features.

        Parameters:
        - columns (list): The list of column names to consider (default: None, which includes all columns).
        
        .. Image:: ../images/histogram_datasets.png
        """
        if columns == None:
            columns = self.columns
        sns.set_theme(rc={"figure.figsize": (40, 40)}, style="whitegrid")

        dfplot = pd.DataFrame(self.dfall, columns=columns)

        nbins = 25
        ax = dfplot[self.target == 0].hist(
            weights=self.weights[self.target == 0],
            figsize=(15, 12),
            color="b",
            alpha=0.5,
            density=True,
            bins=nbins,
            label="B",
        )
        ax = ax.flatten()[
            : dfplot.shape[1]
        ]  # to avoid error if holes in the grid of plots (like if 7 or 8 features)
        dfplot[self.target == 1].hist(
            weights=self.weights[self.target == 1],
            figsize=(15, 12),
            color="r",
            alpha=0.5,
            density=True,
            ax=ax,
            bins=nbins,
            label="S",
        )

        for i in range(len(ax)):
            ax[i].set_title(columns[i])
            ax[i].legend(["Background", "Signal"])
        plt.title("Histograms of features in" + self.name)
        plt.show()

    def correlation_plots(self, columns=None):
        """
        Plots correlation matrices of the dataset features.

        Parameters:
        - columns (list): The list of column names to consider (default: None, which includes all columns).
        
        .. Image:: ../images/correlation_plots.png
        """
        caption = ["Signal feature", "Background feature"]
        if columns == None:
            columns = self.columns
        sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")

        for i in range(2):

            dfplot = pd.DataFrame(self.dfall, columns=columns)

            print(caption[i], " correlation matrix")
            corrMatrix = dfplot[self.target == i].corr()
            sns.heatmap(corrMatrix, annot=True)
            plt.title("Correlation matrix of features in" + self.name)
            plt.show()

    def pair_plots(self, sample_size=10, columns=None):
        """
        Plots pair plots of the dataset features.

        Parameters:
        - sample_size (int): The number of samples to consider (default: 10).
        - columns (list): The list of column names to consider (default: None, which includes all columns).
        
        .. Image:: ../images/pair_plot.png
        """
        if columns == None:
            columns = self.columns
        df_sample = self.dfall[columns].copy()
        df_sample["Label"] = self.target

        df_sample_S = df_sample[self.target == 1].sample(n=sample_size)
        df_sample_B = df_sample[self.target == 0].sample(n=sample_size)
        frames = [df_sample_S, df_sample_B]
        del df_sample
        df_sample = pd.concat(frames)

        sns.set_theme(rc={"figure.figsize": (16, 14)}, style="whitegrid")

        ax = sns.PairGrid(df_sample, hue="Label")
        ax.map_upper(sns.scatterplot, alpha=0.5, size=0.3)
        ax.map_lower(
            sns.kdeplot, fill=True, levels=5, alpha=0.5
        )  # Change alpha value here
        ax.map_diag(
            sns.histplot,
            alpha=0.3,
            bins=25,
        )  # Change alpha value here
        ax.add_legend(title="Legend", labels=["Signal", "Background"], fontsize=12)

        legend = ax._legend
        for lh in legend.legendHandles:
            lh.set_alpha(0.5)
            lh._sizes = [10]

        plt.rcParams["figure.facecolor"] = "w"  # Set the figure facecolor to white
        ax.figure.suptitle("Pair plots of features in" + self.name)
        plt.show()
        plt.close()

    def stacked_histogram(self, field_name, mu_hat=1.0, bins=30):
        """
        Plots a stacked histogram of a specific field in the dataset.

        Parameters:
        - field_name (str): The name of the field to plot.
        - mu_hat (float): The value of mu (default: 1.0).
        - bins (int): The number of bins for the histogram (default: 30).
        
        .. Image:: ../images/stacked_histogram.png
        """
        field = self.dfall[field_name]
        sns.set_theme(rc={"figure.figsize": (8, 7)}, style="whitegrid")

        bins = 30

        hist_s, bins = np.histogram(
            field[self.target == 1],
            bins=bins,
            weights=self.weights[self.target == 1],
        )

        hist_b, bins = np.histogram(
            field[self.target == 0],
            bins=bins,
            weights=self.weights[self.target == 0],
        )

        hist_bkg = hist_b.copy()

        higgs = "htautau"

        for key in self.keys:
            if key != higgs:
                field_key = field[self.detailed_label == key]
                print(key, field_key.shape)
                print(key, self.weight_keys[key].shape)
                hist, bins = np.histogram(
                    field_key,
                    bins=bins,
                    weights=self.weight_keys[key],
                )
                plt.stairs(hist_b, bins, fill=True, label=f"{key} bkg")
                hist_b -= hist
            else:
                print(key, hist_s.shape)

        plt.stairs(
            hist_s * mu_hat + hist_bkg,
            bins,
            fill=False,
            color="orange",
            label=f"$H \\rightarrow \\tau \\tau (\mu = {mu_hat:.3f})$",
        )

        plt.stairs(
            hist_s + hist_bkg,
            bins,
            fill=False,
            color="red",
            label=f"$H \\rightarrow \\tau \\tau (\mu = {1.0:.3f})$",
        )

        plt.legend()
        plt.title(f"Stacked histogram of {field_name} in {self.name}")
        plt.xlabel(f"{field_name}")
        plt.ylabel("Weighted count")
        plt.show()


def histgram_compare(
    data_set1, data_set2, field_name, bins=30, plot_label=["data_set1", "data_set2"]
):
    """Compare the histograms of two data sets based on a given field.
    
    This function plots and compares the histograms of two data sets based on a specified field.
    It uses the seaborn and matplotlib libraries for visualization.

    Args:
        data_set1 (dict): The first data set containing the field of interest.
        data_set2 (dict): The second data set containing the field of interest.
        field_name (str): The name of the field to compare.
        bins (int, optional): The number of bins to use for the histogram. Defaults to 30.
        plot_label (list, optional): The labels for the two data sets in the plot legend. Defaults to ["data_set1", "data_set2"].

    Returns:
        None
        
    .. Image:: ../images/histogram_compare.png

    """

    field_1 = data_set1["data"][field_name]
    field_2 = data_set2["data"][field_name]

    sns.set_theme(rc={"figure.figsize": (8, 7)}, style="whitegrid")

    sns.set_theme(rc={"figure.figsize": (16, 14)}, style="whitegrid")

    hist_1, bins = np.histogram(
        field_1,
        bins=bins,
        weights=data_set1["weights"],
    )

    plt.stairs(hist_1, bins, fill=False, label=plot_label[0])

    hist_2, bins = np.histogram(
        field_2,
        bins=bins,
        weights=data_set2["weights"],
    )

    plt.stairs(hist_2, bins, fill=False, label=plot_label[1])
    plt.legend()

    plt.show()
    plt.close()


def roc_curve_wrapper(score, labels, weights, plot_label="model", color="b", lw=2):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for a binary classification model.

    Parameters:
    score (array-like): The predicted scores or probabilities for the positive class.
    labels (array-like): The true labels for the binary classification problem.
    weights (array-like): The sample weights for each data point.
    plot_label (str, optional): The label to be displayed in the plot legend. Defaults to "model".
    color (str, optional): The color of the ROC curve. Defaults to "b" (blue).
    lw (int, optional): The linewidth of the ROC curve. Defaults to 2.

    Returns:
    None
    
    .. Image:: ../images/roc_curve.png

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

    plt.close()


def events_histogram(field, labels, weights, plot_label=None, y_scale="log"):
    """
    Plot a histogram of events based on a given field, labels, and weights.

    Args:
        field (array-like): The field values for each event.
        labels (array-like): The labels for each event (0 for background, 1 for signal).
        weights (array-like): The weights for each event.
        plot_label (str, optional): The label for the plot. Defaults to None.
        y_scale (str, optional): The scale of the y-axis. Defaults to "log".
    """
    plt.figure()
    sns.set_theme(rc={"figure.figsize": (8, 7)}, style="whitegrid")
    fig, ax = plt.subplots()

    high_low = (0, 1)
    bins = 30

    weights_signal = weights[labels == 1]
    weights_background = weights[labels == 0]

    plt.hist(
        field[labels == 1],
        color="r",
        alpha=0.7,
        range=high_low,
        bins=bins,
        histtype="stepfilled",
        density=False,
        label="S",
        weights=weights_signal,
    )  # alpha is transparancy
    plt.hist(
        field[labels == 0],
        color="b",
        alpha=0.7,
        range=high_low,
        bins=bins,
        histtype="stepfilled",
        density=False,
        label="B",
        weights=weights_background,
    )

    plt.legend()
    plt.title(plot_label)
    plt.xlabel(" Score ")
    plt.ylabel(" Number of events ")
    ax.set_yscale(y_scale)

    plt.show()
    plt.close()
