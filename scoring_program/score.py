# ------------------------------------------
# Imports
# ------------------------------------------
import base64
import io
import json
import logging
import os
import sys
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)


class Scoring:
    """
    This class is used to compute the scores for the competition.
    For more details, see the :doc:`evaluation page <../pages/evaluation>`.

    Atributes:
        * start_time (datetime): The start time of the scoring process.
        * end_time (datetime): The end time of the scoring process.
        * ingestion_results (list): The ingestion results.
        * ingestion_duration (float): The ingestion duration.
        * scores_dict (dict): The scores dictionary.

    Methods:
        * start_timer(): Start the timer.
        * stop_timer(): Stop the timer.
        * get_duration(): Get the duration of the scoring process.
        * show_duration(): Show the duration of the scoring process.
        * load_ingestion_duration(ingestion_duration_file): Load the ingestion duration.
        * load_ingestion_results(prediction_dir="./",score_dir="./"): Load the ingestion results.
        * compute_scores(test_settings): Compute the scores.
        * RMSE_score(mu, mu_hat, delta_mu_hat): Compute the RMSE score.
        * MAE_score(mu, mu_hat, delta_mu_hat): Compute the MAE score.
        * Quantiles_Score(mu, p16, p84, eps=1e-3): Compute the Quantiles Score.
        * write_scores(): Write the scores.
        * write_html(content): Write the HTML content.
        * _print(content): Print the content.
        * save_figure(mu, p16s, p84s, set=0): Save the figure.

    """

    def __init__(self, name=""):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.ingestion_results = None
        self.ingestion_duration = None
        self.name = name
        self.scores_dict = {}

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            logger.warning("Timer was never started. Returning None")
            return None

        if self.end_time is None:
            logger.warning("Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def load_ingestion_duration(self, ingestion_duration_file):
        """
        Load the ingestion duration.

        Args:
            ingestion_duration_file (str): The ingestion duration file.
        """
        logger.info(f"Reading ingestion duration from {ingestion_duration_file}")

        with open(ingestion_duration_file) as f:
            self.ingestion_duration = json.load(f)["ingestion_duration"]

    def load_ingestion_results(self, prediction_dir="./", score_dir="./"):
        """
        Load the ingestion results.

        Args:
            prediction_dir (str, optional): location of the predictions. Defaults to "./".
            score_dir (str, optional): location of the scores. Defaults to "./".
        """
        ingestion_results_with_set_index = []
        # loop over sets (1 set = 1 value of mu)
        for file in os.listdir(prediction_dir):
            if file.startswith("result_"):
                set_index = int(
                    file.split("_")[1].split(".")[0]
                )  # file format: result_{set_index}.json
                results_file = os.path.join(prediction_dir, file)
                with open(results_file) as f:
                    ingestion_results_with_set_index.append(
                        {"set_index": set_index, "results": json.load(f)}
                    )
        ingestion_results_with_set_index = sorted(
            ingestion_results_with_set_index, key=lambda x: x["set_index"]
        )
        self.ingestion_results = [
            x["results"] for x in ingestion_results_with_set_index
        ]

        self.score_file = os.path.join(score_dir, "scores.json")
        self.html_file = os.path.join(score_dir, "detailed_results.html")
        self.score_dir = score_dir
        logger.info(f"Read ingestion results from {prediction_dir}")

    def compute_scores(self, test_settings, no_html=False):
        """
        Compute the scores for the competition based on the test settings.

        Args:
            test_settings (dict): The test settings.
        """

        logger.info("Computing scores")

        # loop over ingestion results
        rmses, maes = [], []
        all_p16s, all_p84s, all_mus = [], [], []
        set_intervals, set_coverages, set_quantiles_scores = [], [], []
        set_sizes = []
        
        self.ground_truth_mus = test_settings["ground_truth_mus"]

        for i in range(len(self.ingestion_results)):
            ingestion_result = self.ingestion_results[i]
            mu = test_settings["ground_truth_mus"][i]

            mu_hats = ingestion_result["mu_hats"]
            delta_mu_hats = ingestion_result["delta_mu_hats"]
            p16s = ingestion_result["p16"]
            p84s = ingestion_result["p84"]

            set_sizes.append(len(mu_hats))
            all_mus.extend(np.repeat(mu, len(p16s)))
            all_p16s.extend(p16s)
            all_p84s.extend(p84s)

            set_rmses, set_maes = [], []
            for mu_hat, delta_mu_hat in zip(mu_hats, delta_mu_hats):
                set_rmses.append(self.RMSE_score(mu, mu_hat, delta_mu_hat))
                set_maes.append(self.MAE_score(mu, mu_hat, delta_mu_hat))
            set_interval, set_coverage, set_quantiles_score = self.Quantiles_Score(
                np.repeat(mu, len(p16s)), np.array(p16s), np.array(p84s)
            )

            set_intervals.append(set_interval)
            set_coverages.append(set_coverage)
            set_quantiles_scores.append(set_quantiles_score)

            set_mae = np.mean(set_maes)
            set_rmse = np.mean(set_rmses)

            if not no_html:

                self._print("------------------")
                self._print(f"Set {i}")
                self._print("------------------")
                self._print(f"MAE (avg): {set_mae}")
                self._print(f"RMSE (avg): {set_rmse}")
                self._print(f"Interval: {set_interval}")
                self._print(f"Coverage: {set_coverage}")
                self._print(f"Quantiles Score: {set_quantiles_score}")

                self.save_figure(
                    mu=np.mean(mu_hats), p16s=p16s, p84s=p84s, set=i, true_mu=mu
                )

            # Save set scores in lists
            rmses.append(set_rmse)
            maes.append(set_mae)

        import pandas as pd

        self.set_score_df = pd.DataFrame(
            {
                "set": range(len(self.ingestion_results)),
                "mae": maes,
                "rmse": rmses,
                "interval": set_intervals,
                "coverage": set_coverages,
                "quantiles_score": set_quantiles_scores,
                "set_size": set_sizes,
            }
        )

        overall_interval, overall_coverage, overall_quantiles_score = (
            self.Quantiles_Score(
                np.array(all_mus), np.array(all_p16s), np.array(all_p84s)
            )
        )

        self.scores_dict = {
            "rmse": np.mean(rmses),
            "mae": np.mean(maes),
            "interval": overall_interval,
            "coverage": overall_coverage,
            "quantiles_score": overall_quantiles_score,
            "ingestion_duration": self.ingestion_duration,
        }
        
        if not no_html:

            self._print("\n\n==================")
            self._print("Overall Score")
            self._print("==================")
            self._print(f"[*] --- RMSE: {round(np.mean(rmses), 3)}")
            self._print(f"[*] --- MAE: {round(np.mean(maes), 3)}")
            self._print(f"[*] --- Interval: {round(overall_interval, 3)}")
            self._print(f"[*] --- Coverage: {round(overall_coverage, 3)}")
            self._print(f"[*] --- Quantiles score: {round(overall_quantiles_score, 3)}")
            self._print(f"[*] --- Ingestion duration: {self.ingestion_duration}")

        print("[✔]")

    def compute_bootstraped_scores(
        self, eps=1e-3, n_bootstraps=10, sample_size=50, random_seed=42
    ):

        def f(x, n_tries, max_coverage=1e4, one_sigma=0.6827):
            sigma68 = np.sqrt(((1 - one_sigma) * one_sigma * n_tries)) / n_tries

            if x >= one_sigma - 2 * sigma68 and x <= one_sigma + 2 * sigma68:
                out = 1
            elif x < one_sigma - 2 * sigma68:
                out = 1 + abs((x - (one_sigma - 2 * sigma68)) / sigma68) ** 4
            elif x > one_sigma + 2 * sigma68:
                out = 1 + abs((x - (one_sigma + 2 * sigma68)) / sigma68) ** 3
            return out

        coverages = []
        intervals = []
        quantiles_scores = []

        for i in range(n_bootstraps):
            random_state = np.random.RandomState(random_seed + i)

            bootstrap_indices = random_state.choice(
                self.set_score_df.index, size=sample_size, replace=True
            )
            bootstrap_df = self.set_score_df.loc[bootstrap_indices]
            bootstrap_df = bootstrap_df.reset_index(drop=True)

            coverage = np.mean(bootstrap_df["coverage"])
            interval = np.mean(bootstrap_df["interval"])
            n_rows = np.sum(bootstrap_df["set_size"])
            quantiles_score = -np.log((interval + eps) * f(coverage, n_tries=n_rows))

            coverages.append(coverage)
            intervals.append(interval)
            quantiles_scores.append(quantiles_score)

            # print(f"[*] --- Interval: {round(interval, 3)}")
            # print(f"[*] --- Coverage: {round(coverage, 3)}")
            # print(f"[*] --- Quantiles score: {round(quantiles_score, 3)}")

        self.bootstraped_scores = {
            "interval": intervals,
            "coverage": coverages,
            "quantiles_score": quantiles_scores,
        }

    def plot_bootstraped_scores(self):
        
        self._print_center("\n=====================================\n")
        self._print_center("Results")
        self._print_center("\n=====================================\n")
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].plot(
            self.ground_truth_mus,
            self.set_score_df["quantiles_score"],
            "o",
            color="m",
            label=f"Quantile Score {self.name}",
        )
        axs[0].set_xlabel("Mu")
        axs[0].set_ylabel("Quantile Score")
        axs[0].set_title("Quantile Score vs Mu")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(
            self.ground_truth_mus,
            self.set_score_df["coverage"],
            "o",
            color="darkblue",
            label=f"Coverage {self.name}",
        )
        axs[1].set_xlabel("Mu")
        axs[1].set_ylabel("Coverage")
        axs[1].set_title("Coverage vs Mu")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(
            self.ground_truth_mus,
            self.set_score_df["interval"],
            "o",
            color="orange",
            label=f"Interval {self.name}",
        )
        axs[2].set_xlabel("Mu")
        axs[2].set_ylabel("Interval")
        axs[2].set_title("Interval vs Mu")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        fig_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        self.write_html(f"<img src='data:image/png;base64,{fig_b64}'><br>")

        self._print_center("\n=====================================\n")
        self._print_center("Bootstraped Scores")
        self._print_center("\n=====================================\n")

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot histogram of bootstraped quantiles scores
        std_err_quantiles = np.array(self.bootstraped_scores["quantiles_score"]).std()
        axs[0].hist(
            self.bootstraped_scores["quantiles_score"],
            bins=20,
            color="r",
            alpha=0.7,
            label=f"Bootstraped Scores {self.name} (std_err={std_err_quantiles:.3f})",
        )
        axs[0].set_xlabel("Score")
        axs[0].set_ylabel("Frequency")
        axs[0].set_title("Histogram of Bootstraped Scores")
        axs[0].legend()
        axs[0].grid(True)

        # Plot histogram of bootstraped coverage
        std_err_coverage = np.array(self.bootstraped_scores["coverage"]).std()
        axs[1].hist(
            self.bootstraped_scores["coverage"],
            bins=20,
            color="g",
            alpha=0.7,
            label=f"Bootstraped Coverage {self.name} (std_err={std_err_coverage:.3f})",
        )
        axs[1].set_xlabel("Coverage")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Histogram of Bootstraped Coverage")
        axs[1].legend()
        axs[1].grid(True)

        # Plot histogram of bootstraped interval lengths
        std_err_interval = np.array(self.bootstraped_scores["interval"]).std()
        axs[2].hist(
            self.bootstraped_scores["interval"],
            bins=20,
            alpha=0.7,
            color="b",
            label=f"Bootstraped Interval Length {self.name} (std_err={std_err_interval:.3f})",
        )
        axs[2].set_xlabel("Interval Length")
        axs[2].set_ylabel("Frequency")
        axs[2].set_title("Histogram of Bootstraped Interval Length")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        fig_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        self.write_html(f"<img src='data:image/png;base64,{fig_b64}'><br>")

    def RMSE_score(self, mu, mu_hat, delta_mu_hat):
        """
        Compute the root mean squared error between the true value mu and the predicted value mu_hat.

        Args:
            * mu (float): The true value.
            * mu_hat (np.array): The predicted value.
            * delta_mu_hat (np.array): The uncertainty on the predicted value.
        """

        def MSE(mu, mu_hat):
            """Compute the mean squared error between scalar mu and vector mu_hat."""
            return np.mean((mu_hat - mu) ** 2)

        def MSE2(mu, mu_hat, delta_mu_hat):
            """Compute the mean squared error between computed delta_mu = mu_hat - mu and delta_mu_hat."""
            adjusted_diffs = (mu_hat - mu) ** 2 - delta_mu_hat**2
            return np.mean(adjusted_diffs**2)

        return np.sqrt(MSE(mu, mu_hat) + MSE2(mu, mu_hat, delta_mu_hat))

    def MAE_score(self, mu, mu_hat, delta_mu_hat):
        """
        Compute the mean absolute error between the true value mu and the predicted value mu_hat.

        Args:
            * mu (float): The true value.
            * mu_hat (np.array): The predicted value.
            * delta_mu_hat (np.array): The uncertainty on the predicted value
        """

        def MAE(mu, mu_hat):
            """Compute the mean absolute error between scalar mu and vector mu_hat."""
            return np.mean(np.abs(mu_hat - mu))

        def MAE2(mu, mu_hat, delta_mu_hat):
            """Compute the mean absolute error based on the provided definitions."""
            adjusted_diffs = np.abs(mu_hat - mu) - delta_mu_hat
            return np.mean(np.abs(adjusted_diffs))

        return MAE(mu, mu_hat) + MAE2(mu, mu_hat, delta_mu_hat)

    def Quantiles_Score(self, mu, p16, p84, eps=1e-3):
        """
        Compute the quantiles score based on the true value mu and the quantiles p16 and p84.

        Args:
            * mu (array): The true ${\\mu} value.
            * p16 (array): The 16th percentile.
            * p84 (array): The 84th percentile.
            * eps (float, optional): A small value to avoid division by zero. Defaults to 1e-3.
        """

        def Interval(p16, p84):
            """Compute the average of the intervals defined by vectors p16 and p84."""
            interval = np.mean(p84 - p16)
            if interval < 0:
                logger.warning(f"Interval is negative: {interval}")
            return np.mean(abs(p84 - p16))

        def Coverage(mu, p16, p84):
            """Compute the fraction of times scalar mu is within intervals defined by vectors p16 and p84."""
            return_coverage = np.mean((mu >= p16) & (mu <= p84))
            return return_coverage

        def f(x, n_tries, max_coverage=1e4, one_sigma=0.6827):
            sigma68 = np.sqrt(((1 - one_sigma) * one_sigma * n_tries)) / n_tries

            if x >= one_sigma - 2 * sigma68 and x <= one_sigma + 2 * sigma68:
                out = 1
            elif x < one_sigma - 2 * sigma68:
                out = 1 + abs((x - (one_sigma - 2 * sigma68)) / sigma68) ** 4
            elif x > one_sigma + 2 * sigma68:
                out = 1 + abs((x - (one_sigma + 2 * sigma68)) / sigma68) ** 3
            return out

        coverage = Coverage(mu, p16, p84)
        interval = Interval(p16, p84)
        score = -np.log((interval + eps) * f(coverage, n_tries=mu.shape[0]))
        return interval, coverage, score

    def write_scores(self):

        logger.info(f"Writing scores to {self.score_file}")

        with open(self.score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))

    def write_html(self, content):
        with open(self.html_file, "a", encoding="utf-8") as f:
            f.write(content)

    def _print(self, content):
        print(content)
        self.write_html(content + "<br>")
        
    def _print_center(self, content):
        self.write_html(f"<div style='text-align: center;'>{content}</div><br>")

    def save_figure(self, mu, p16s, p84s, true_mu=None, set=0):
        """
        Save the figure of the mu distribution.

        Args:
            * mu (array): The true ${\\mu} value.
            * p16 (array): The 16th percentile.
            * p84 (array): The 84th percentile.
            * set (int, optional): The set number. Defaults to 0.
        """
        fig = plt.figure(figsize=(8, 6))
        # plot horizontal lines from p16 to p84
        for i, (p16, p84) in enumerate(zip(p16s, p84s)):
            if p16 > p84:
                p16, p84 = 0, 0
            if i == 0:
                plt.hlines(
                    y=i, xmin=p16, xmax=p84, colors="b", label="Coverage interval"
                )
            else:
                plt.hlines(y=i, xmin=p16, xmax=p84, colors="b")
        plt.vlines(
            x=mu,
            ymin=0,
            ymax=len(p16s),
            colors="r",
            linestyles="dashed",
            label="average $\\mu$",
        )
        if true_mu is not None:
            plt.vlines(
                x=true_mu,
                ymin=0,
                ymax=len(p16s),
                colors="g",
                linestyles="dashed",
                label="true $\\mu$",
            )
        plt.xlabel("$\\mu$", fontdict={"size": 14})
        plt.ylabel("pseudo-experiments", fontdict={"size": 14})
        plt.xticks(fontsize=14)  # Set the x-tick font size
        plt.yticks(fontsize=14)  # Set the y-tick font size
        plt.title(f"Set {set}", fontdict={"size": 14})

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
        plt.grid()
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        fig_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        self.write_html(f"<img src='data:image/png;base64,{fig_b64}'><br>")
