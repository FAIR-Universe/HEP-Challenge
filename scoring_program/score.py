# ------------------------------------------
# Imports
# ------------------------------------------
import os
import numpy as np
import json
from datetime import datetime as dt
import matplotlib.pyplot as plt
import sys
import io
import base64

# ------------------------------------------
# Settings
# ------------------------------------------
# True when running on Codabench
# False when running locally
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)


class Scoring:
    def __init__(self):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.ingestion_results = None
        self.ingestion_duration = None

        self.scores_dict = {}

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def show_duration(self):
        print("\n---------------------------------")
        print(f"[✔] Total duration: {self.get_duration()}")
        print("---------------------------------")

    def load_ingestion_duration(self,ingestion_duration_file):
        print("[*] Reading ingestion duration")
        with open(ingestion_duration_file) as f:
            self.ingestion_duration = json.load(f)["ingestion_duration"]

        print("[✔]")

    def load_ingestion_results(self, prediction_dir = "./"):
        print("[*] Reading predictions")
        self.ingestion_results = []
        # loop over sets (1 set = 1 value of mu)
        for file in os.listdir(prediction_dir):
            if file.startswith("result_"):
                results_file = os.path.join(prediction_dir, file)
                with open(results_file) as f:
                    self.ingestion_results.append(json.load(f))
        
        self.score_file = os.path.join(prediction_dir, "scores.json")
        self.html_file = os.path.join(prediction_dir, "detailed_results.html")

        print("[✔]")

    def compute_scores(self, test_settings):
        print("[*] Computing scores")

        # loop over ingestion results
        rmses, maes = [], []
        all_p16s, all_p84s, all_mus = [], [], []
        print("[*] ",self.ingestion_results)
        print("[*] ",test_settings["ground_truth_mus"])

        for i in range(len(self.ingestion_results)):
            ingestion_result = self.ingestion_results[i]
            mu = test_settings["ground_truth_mus"][i]
            
            print(f"[*] mu_hats: {ingestion_result}")
            mu_hats = ingestion_result["mu_hats"]
            delta_mu_hats = ingestion_result["delta_mu_hats"]
            p16s = ingestion_result["p16"]
            p84s = ingestion_result["p84"]

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

            set_mae = np.mean(set_maes)
            set_rmse = np.mean(set_rmses)

            self._print("------------------")
            self._print(f"Set {i}")
            self._print("------------------")
            self._print(f"MAE (avg): {set_mae}")
            self._print(f"RMSE (avg): {set_rmse}")
            self._print(f"Interval: {set_interval}")
            self._print(f"Coverage: {set_coverage}")
            self._print(f"Quantiles Score: {set_quantiles_score}")

            self.save_figure(mu=np.mean(mu_hats), p16s=p16s, p84s=p84s, set=i)

            # Save set scores in lists
            rmses.append(set_rmse)
            maes.append(set_mae)

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

    def RMSE_score(self, mu, mu_hat, delta_mu_hat):
        """Compute the sum of MSE and MSE2."""

        def MSE(mu, mu_hat):
            """Compute the mean squared error between scalar mu and vector mu_hat."""
            return np.mean((mu_hat - mu) ** 2)

        def MSE2(mu, mu_hat, delta_mu_hat):
            """Compute the mean squared error between computed delta_mu = mu_hat - mu and delta_mu_hat."""
            adjusted_diffs = (mu_hat - mu) ** 2 - delta_mu_hat**2
            return np.mean(adjusted_diffs**2)

        return np.sqrt(MSE(mu, mu_hat) + MSE2(mu, mu_hat, delta_mu_hat))

    def MAE_score(self, mu, mu_hat, delta_mu_hat):
        """Compute the sum of MAE and MAE2."""

        def MAE(mu, mu_hat):
            """Compute the mean absolute error between scalar mu and vector mu_hat."""
            return np.mean(np.abs(mu_hat - mu))

        def MAE2(mu, mu_hat, delta_mu_hat):
            """Compute the mean absolute error based on the provided definitions."""
            adjusted_diffs = np.abs(mu_hat - mu) - delta_mu_hat
            return np.mean(np.abs(adjusted_diffs))

        return MAE(mu, mu_hat) + MAE2(mu, mu_hat, delta_mu_hat)

    def Quantiles_Score(self, mu, p16, p84, eps=1e-3):

        def Interval(p16, p84):
            """Compute the average of the intervals defined by vectors p16 and p84."""
            return np.mean(np.abs(p84 - p16))

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
        print("[*] Writing scores")

        with open(self.score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))

        print("[✔]")

    def write_html(self, content):
        with open(self.html_file, "a", encoding="utf-8") as f:
            f.write(content)

    def _print(self, content):
        print(content)
        self.write_html(content + "<br>")

    def save_figure(self, mu, p16s, p84s, set=0):
        fig = plt.figure(figsize=(5, 5))
        # plot horizontal lines from p16 to p84
        for i, (p16, p84) in enumerate(zip(p16s, p84s)):
            plt.hlines(y=i, xmin=p16, xmax=p84, colors="b")
        plt.vlines(
            x=mu,
            ymin=0,
            ymax=len(p16s),
            colors="r",
            linestyles="dashed",
            label="average $\mu$",
        )
        plt.xlabel("mu")
        plt.ylabel("psuedo-experiments")
        plt.title(f"mu distribution - Set {set}")
        plt.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        fig_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        self.write_html(f"<img src='data:image/png;base64,{fig_b64}'><br>")


if __name__ == "__main__":
    print("############################################")
    print("### Scoring Program")
    print("############################################\n")


    # Init scoring
    scoring = Scoring()

    # Start timer
    scoring.start_timer()


    # Load ingestion duration
    scoring.load_ingestion_duration()

    # Load ingestions results
    scoring.load_ingestion_results()

    # Compute Scores
    scoring.compute_scores()

    # Write scores
    scoring.write_scores()

    # Stop timer
    scoring.stop_timer()

    # Show duration
    scoring.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Scoring Program executed successfully!")
    print("----------------------------------------------\n\n")
