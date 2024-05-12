# ------------------------------------------
# Imports
# ------------------------------------------
import numpy as np
import os
import pandas as pd
from datetime import datetime as dt
import json
from itertools import product
import warnings
import sys

warnings.filterwarnings("ignore")

INPUT_DIR = None

# Load config
from config import (
    NUM_SETS,
    NUM_PSEUDO_EXPERIMENTS,
    USE_SYSTEAMTICS,
    DICT_SYSTEMATICS,
    USE_RANDOM_MUS,
    CODABENCH,
    USE_PUBLIC_DATA,
    INPUT_DIR,
    CSV,
    PARQUET,
)


# ------------------------------------------
# Ingestion Class
# ------------------------------------------
class Ingestion:

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.model = None
        self.train_set = None

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

    def save_duration(self):
        duration = self.get_duration()
        duration_in_mins = int(duration.total_seconds() / 60)
        duration_file = os.path.join(self.output_dir, "ingestion_duration.json")
        if duration is not None:
            with open(duration_file, "w") as f:
                f.write(json.dumps({"ingestion_duration": duration_in_mins}, indent=4))

    def set_directories(self):

        # set default directories
        module_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir_name = os.path.dirname(module_dir)

        input_data_dir_name = "public_data" if USE_PUBLIC_DATA else "input_data"
        output_dir_name = "sample_result_submission"
        program_dir_name = "ingestion_program"
        submission_dir_name = "sample_code_submission"

        if CODABENCH:
            root_dir_name = "/app"
            input_data_dir_name = "input_data"
            output_dir_name = "output"
            program_dir_name = "program"
            submission_dir_name = "ingested_program"

        # Input data directory to read training and test data from
        if INPUT_DIR is not None:
            self.input_dir = INPUT_DIR
        else:
            self.input_dir = os.path.join(root_dir_name, input_data_dir_name)
        # Output data directory to write predictions to
        self.output_dir = os.path.join(root_dir_name, output_dir_name)
        # Program directory
        self.program_dir = os.path.join(root_dir_name, program_dir_name)
        # Directory to read submitted submissions from
        self.submission_dir = os.path.join(root_dir_name, submission_dir_name)

        # In case submission dir and output dir are provided as args
        if len(sys.argv) > 1:
            self.submission_dir = sys.argv[1]
        if len(sys.argv) > 2:
            self.output_dir = sys.argv[2]

        # Add to path
        sys.path.append(self.input_dir)
        sys.path.append(self.output_dir)
        sys.path.append(self.program_dir)
        sys.path.append(self.submission_dir)

    def load_train_set(self):
        print("[*] Loading Train data")

        if CSV:
            train_data_file = os.path.join(self.input_dir, "train", "data", "data.csv")
        if PARQUET:
            train_data_file = os.path.join(
                self.input_dir, "train", "data", "data.parquet"
            )
        train_labels_file = os.path.join(
            self.input_dir, "train", "labels", "data.labels"
        )
        train_settings_file = os.path.join(
            self.input_dir, "train", "settings", "data.json"
        )
        train_weights_file = os.path.join(
            self.input_dir, "train", "weights", "data.weights"
        )
        train_detailed_labels_file = os.path.join(
            self.input_dir, "train", "detailed_labels", "data.detailed_labels"
        )

        # read train labels
        with open(train_labels_file, "r") as f:
            train_labels = np.array(f.read().splitlines(), dtype=float)

        # read train settings
        with open(train_settings_file) as f:
            train_settings = json.load(f)

        # read train weights
        with open(train_weights_file) as f:
            train_weights = np.array(f.read().splitlines(), dtype=float)

        # read train process flags
        with open(train_detailed_labels_file) as f:
            train_detailed_labels = f.read().splitlines()

        if PARQUET:
            self.train_set = {
                "data": pd.read_parquet(train_data_file, engine="pyarrow"),
                "labels": train_labels,
                "settings": train_settings,
                "weights": train_weights,
                "detailed_labels": train_detailed_labels,
            }

        if CSV:
            self.train_set = {
                "data": pd.read_csv(train_data_file),
                "labels": train_labels,
                "settings": train_settings,
                "weights": train_weights,
                "detailed_labels": train_detailed_labels,
            }

        del train_labels, train_settings, train_weights, train_detailed_labels

        print(self.train_set["data"].info(verbose=False, memory_usage="deep"))
        print("[*] Train data loaded successfully")

    def load_test_set(self):
        print("[*] Loading Test data")

        test_data_dir = os.path.join(self.input_dir, "test", "data")
        test_settings_file = os.path.join(test_data_dir, "settings", "data.json")

        # read test settings
        if USE_RANDOM_MUS:
            self.test_settings = {
                "ground_truth_mus": (np.random.uniform(0.1, 3, NUM_SETS)).tolist()
            }
            random_settings_file = os.path.join(self.output_dir, "random_mu.json")
            with open(random_settings_file, "w") as f:
                json.dump(self.test_settings, f)
        else:
            with open(test_settings_file) as f:
                self.test_settings = json.load(f)

        self.test_set = {
            "Z": pd.DataFrame(),
            "W": pd.DataFrame(),
            "Diboson": pd.DataFrame(),
            "TT": pd.DataFrame(),
            "H": pd.DataFrame(),
        }

        for key in self.test_set.keys():
            self.test_set[key] = self.test_set[key].round(3)
            if CSV:
                test_data_path = os.path.join(test_data_dir, f"{key}_data.csv")
                self.test_set[key] = pd.read_csv(test_data_path)

            if PARQUET:
                test_data_path = os.path.join(test_data_dir, f"{key}_data.parquet")
                self.test_set[key] = pd.read_parquet(test_data_path, engine="pyarrow")

        print("[*] Test data loaded successfully")

    def init_submission(self):
        print("[*] Initializing Submmited Model")
        from model import Model
        from systematics import Systematics

        self.model = Model(train_set=self.train_set, systematics=Systematics)

        del self.train_set

    def fit_submission(self):
        print("[*] Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self):
        print("[*] Calling predict method of submitted model")

        # get set indices
        set_indices = np.arange(0, NUM_SETS)
        # get test set indices per set
        test_set_indices = np.arange(0, NUM_PSEUDO_EXPERIMENTS)

        # create a product of set and test set indices all combinations of tuples
        all_combinations = list(product(set_indices, test_set_indices))
        # randomly shuffle all combinations of indices
        np.random.shuffle(all_combinations)

        self.results_dict = {}
        for set_index, test_set_index in all_combinations:
            if USE_SYSTEAMTICS:
                # random tes value (one per test set)
                if DICT_SYSTEMATICS["tes"]:
                    tes = np.random.uniform(0.9, 1.1)
                else:
                    tes = 1.0
                if DICT_SYSTEMATICS["jes"]:
                    jes = np.random.uniform(0.9, 1.1)
                else:
                    jes = 1.0
                if DICT_SYSTEMATICS["soft_met"]:
                    soft_met = np.random.uniform(1.0, 5)
                else:
                    soft_met = 1.0

                if DICT_SYSTEMATICS["w_scale"]:
                    w_scale = np.random.uniform(0.5, 2)
                else:
                    w_scale = None

                if DICT_SYSTEMATICS["bkg_scale"]:
                    bkg_scale = np.random.uniform(0.5, 2)
                else:
                    bkg_scale = None

            else:
                tes = 1.0
                jes = 1.0
                soft_met = 1.0
                w_scale = (None,)
                bkg_scale = (None,)
            # create a seed
            seed = (set_index * NUM_PSEUDO_EXPERIMENTS) + test_set_index
            # get mu value of set from test settings
            set_mu = self.test_settings["ground_truth_mus"][set_index]

            from systematics import get_bootstraped_dataset, get_systematics_dataset

            # get bootstrapped dataset from the original test set
            pesudo_exp_data = get_bootstraped_dataset(
                self.test_set,
                mu=set_mu,
                w_scale=w_scale,
                bkg_scale=bkg_scale,
                seed=seed,
            )
            test_set = get_systematics_dataset(
                pesudo_exp_data,
                tes=tes,
                jes=jes,
                soft_met=soft_met,
            )

            predicted_dict = self.model.predict(test_set)
            predicted_dict["test_set_index"] = test_set_index

            print(
                f"[*] - mu_hat: {predicted_dict['mu_hat']} - delta_mu_hat: {predicted_dict['delta_mu_hat']} - p16: {predicted_dict['p16']} - p84: {predicted_dict['p84']}"
            )

            if set_index not in self.results_dict:
                self.results_dict[set_index] = []
            self.results_dict[set_index].append(predicted_dict)

    def save_result(self):
        print("[*] Saving ingestion result")

        # loop over sets
        for i in range(0, NUM_SETS):
            set_result = self.results_dict[i]
            set_result.sort(key=lambda x: x["test_set_index"])
            mu_hats, delta_mu_hats, p16, p84 = [], [], [], []
            for test_set_dict in set_result:
                mu_hats.append(test_set_dict["mu_hat"])
                delta_mu_hats.append(test_set_dict["delta_mu_hat"])
                p16.append(test_set_dict["p16"])
                p84.append(test_set_dict["p84"])

            ingestion_result_dict = {
                "mu_hats": mu_hats,
                "delta_mu_hats": delta_mu_hats,
                "p16": p16,
                "p84": p84,
            }
            result_file = os.path.join(self.output_dir, "result_" + str(i) + ".json")
            with open(result_file, "w") as f:
                f.write(json.dumps(ingestion_result_dict, indent=4))


if __name__ == "__main__":

    print("############################################")
    print("### Ingestion Program")
    print("############################################\n")

    # Init Ingestion
    ingestion = Ingestion()

    ingestion.set_directories()

    # Start timer
    ingestion.start_timer()

    # load train set
    ingestion.load_train_set()

    # initialize submission
    ingestion.init_submission()

    # fit submission
    ingestion.fit_submission()

    # load test set
    ingestion.load_test_set()

    # predict submission
    ingestion.predict_submission()

    # save result
    ingestion.save_result()

    # Stop timer
    ingestion.stop_timer()

    # Show duration
    ingestion.show_duration()

    # Save duration
    ingestion.save_duration()

    print("\n----------------------------------------------")
    print("[✔] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
