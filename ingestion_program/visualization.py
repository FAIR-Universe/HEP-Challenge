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
        * data_set (dict): The dataset containing the data, labels, weights, and detailed labels.
        * name (str): The name of the dataset (default: "dataset").
        * columns (list): The list of column names to consider (default: None, which includes all columns).

    Attributes:
        * dfall (DataFrame): The dataset.
        * target (Series): The labels.
        * weights (Series): The weights.
        * detailed_label (ndarray): The detailed labels.
        * columns (list): The list of column names.
        * name (str): The name of the dataset.
        * keys (ndarray): The unique detailed labels.
        * weight_keys (dict): The weights for each detailed label.

    Methods:
        * examine_dataset(): Prints information about the dataset.
        * histogram_dataset(columns=None): Plots histograms of the dataset features.
        * correlation_plots(columns=None): Plots correlation matrices of the dataset features.
        * pair_plots(sample_size=10, columns=None): Plots pair plots of the dataset features.
        * stacked_histogram(field_name, mu_hat=1.0, bins=30): Plots a stacked histogram of a specific field in the dataset.
        * pair_plots_syst(df_syst, sample_size=10): Plots pair plots between the dataset and a system dataset.
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

        Args:
            * columns (list): The list of column names to consider (default: None, which includes all columns).

        .. Image:: ../images/histogram_datasets.png
        """
        if columns is None:
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

        Args:
        * columns (list): The list of column names to consider (default: None, which includes all columns).

        .. Image:: ../images/correlation_plots.png
        """
        caption = ["Signal feature", "Background feature"]
        if columns is None:
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

        Args:
            * sample_size (int): The number of samples to consider (default: 10).
            * columns (list): The list of column names to consider (default: None, which includes all columns).

        .. Image:: ../images/pair_plot.png
        """
        if columns is None:
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

        legend = ax.legend
        for line in legend.get_lines():  # For lines
            line.set_alpha(0.5)
            line.set_linewidth(1.5)

        plt.rcParams["figure.facecolor"] = "w"  # Set the figure facecolor to white
        ax.figure.suptitle("Pair plots of features in" + self.name)
        plt.show()
        plt.close()

    def stacked_histogram(self, field_name, mu_hat=1.0, bins=30):
        """
        Plots a stacked histogram of a specific field in the dataset.

        Args:
            * field_name (str): The name of the field to plot.
            * mu_hat (float): The value of mu (default: 1.0).
            * bins (int): The number of bins for the histogram (default: 30).

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

    def pair_plots_syst(self, df_syst, sample_size=10):
        """
        Plots pair plots between the dataset and a system dataset.

        Args:
            * df_syst (DataFrame): The system dataset.
            * sample_size (int): The number of samples to consider (default: 10).
        
        ..images:: ../images/pair_plot_syst.png
        """
        df_sample = self.dfall[self.columns].copy()
        df_sample_syst = df_syst[self.columns].copy()

        df_sample = df_sample.sample(n=sample_size)
        df_sample["syst"] = False

        df_sample_syst = df_sample_syst.sample(n=sample_size)
        df_sample_syst["syst"] = True

        frames = [df_sample, df_sample_syst]
        del df_sample
        df_sample = pd.concat(frames)

        sns.set_theme(rc={"figure.figsize": (16, 14)}, style="whitegrid")

        ax = sns.PairGrid(df_sample, hue="syst")
        ax.map_upper(sns.scatterplot, alpha=0.5, size=0.3)
        ax.map_lower(
            sns.kdeplot, fill=True, levels=5, alpha=0.5
        )  # Change alpha value here
        ax.map_diag(
            sns.histplot,
            alpha=0.3,
            bins=25,
        )  # Change alpha value here
        ax.add_legend(title="Legend", labels=["syst", "no_syst"], fontsize=12)

        ax.figure.suptitle("Pair plots of features between syst and no_syst")
        plt.show()
        plt.close()



def visualize_scatter(ingestion_result_dict, ground_truth_mus):
    """
    Plots a scatter Plot of ground truth vs. predicted mu values.

    Args:
        * ingestion_result_dict (dict): A dictionary containing the ingestion results.
        * ground_truth_mus (dict): A dictionary of ground truth mu values.
        
    .. Image:: ../images/scatter_plot_mu.png
    """
    plt.figure(figsize=(6, 4))
    for key in ingestion_result_dict.keys():
        ingestion_result = ingestion_result_dict[key]
        mu_hat = np.mean(ingestion_result["mu_hats"])
        mu = ground_truth_mus[key]
        plt.scatter(mu, mu_hat, c='b', marker='o')
    
    plt.xlabel('Ground Truth $\mu$')
    plt.ylabel('Predicted $\mu$ (averaged for 100 test sets)')
    plt.title('Ground Truth vs. Predicted $\mu$ Values')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
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


def visualize_coverage(ingestion_result_dict, ground_truth_mus):
    """
    Plots a coverage plot of the mu values.

    Args:
        * ingestion_result_dict (dict): A dictionary containing the ingestion results.
        * ground_truth_mus (dict): A dictionary of ground truth mu values.
        
    .. Image:: ../images/coverage_plot.png
    """

    for key in ingestion_result_dict.keys():
        plt.figure( figsize=(5, 5))

        ingestion_result = ingestion_result_dict[key]
        mu = ground_truth_mus[key]
        mu_hats = np.mean(ingestion_result["mu_hats"])
        p16s = ingestion_result["p16"]
        p84s = ingestion_result["p84"]
        
        # plot horizontal lines from p16 to p84
        for i, (p16, p84) in enumerate(zip(p16s, p84s)):
            plt.hlines(y=i, xmin=p16, xmax=p84, colors='b', label='p16-p84')

        plt.vlines(x=mu_hats, ymin=0, ymax=len(p16s), colors='r', linestyles='dashed', label='Predicted $\mu$')
        plt.vlines(x=mu, ymin=0, ymax=len(p16s), colors='g', linestyles='dashed', label='Ground Truth $\mu$')
        plt.xlabel('mu')
        plt.ylabel('pseudo-experiments')
        plt.title(f'mu distribution - Set_{key}')
        plt.legend()
        
    plt.show()
