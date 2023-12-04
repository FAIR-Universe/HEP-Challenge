# ------------------------------------------
# Imports
# ------------------------------------------
from sys import path
import numpy as np
import os
import pandas as pd
from datetime import datetime as dt
import json
from itertools import product
from numpy.random import RandomState
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")


# ------------------------------------------
# Default Directories
# ------------------------------------------
# # Root directory
# module_dir = os.path.dirname(os.path.realpath(__file__))
# root_dir = os.path.dirname(module_dir)
# # Input data directory to read training and test data from
# input_dir = os.path.join(root_dir,"input_data")
# # Output data directory to write predictions to
# output_dir = os.path.join(root_dir, "sample_result_submission")
# # Program directory
# program_dir = os.path.join(root_dir, "ingestion_program")
# # Directory to read submitted submissions from
# submission_dir = os.path.join(root_dir, "sample_code_submission")

# ------------------------------------------
# Codabench Directories
# ------------------------------------------
# Root directory
root_dir = "/app"
# Input data directory to read training and test data from
input_dir = os.path.join(root_dir, "input_data")
# Output data directory to write predictions to
output_dir = os.path.join(root_dir, "output")
# Program directory
program_dir = os.path.join(root_dir, "program")
# Directory to read submitted submissions from
submission_dir = os.path.join(root_dir, "ingested_program")

path.append(input_dir)
path.append(output_dir)
path.append(program_dir)
path.append(submission_dir)


# ------------------------------------------
# Import Systamtics
# ------------------------------------------
from systematics import Systematics, postprocess

# ------------------------------------------
# Import Model
# ------------------------------------------

from model import Model


class Ingestion():

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
        print(f'[✔] Total duration: {self.get_duration()}')
        print("---------------------------------")

    def load_train_set(self):
        print("[*] Loading Train data")

        train_data_file = os.path.join(input_dir, 'train', 'data', 'data.csv')
        train_labels_file = os.path.join(input_dir, 'train', 'labels', "data.labels")
        train_settings_file = os.path.join(input_dir, 'train', 'settings', "data.json")
        train_weights_file = os.path.join(input_dir, 'train', 'weights', "data.weights")

        # read train data
        train_data = pd.read_csv(train_data_file)

        # read train labels
        with open(train_labels_file, "r") as f:
            train_labels = np.array(f.read().splitlines(), dtype=float)

        # read train settings
        with open(train_settings_file) as f:
            train_settings = json.load(f)

        # read train weights
        with open(train_weights_file) as f:
            train_weights = np.array(f.read().splitlines(), dtype=float)

        self.train_set = {
            "data": train_data,
            "labels": train_labels,
            "settings": train_settings,
            "weights": train_weights
        }

    def load_test_set(self):
        print("[*] Loading Test data")

        test_data_file = os.path.join(input_dir, 'test', 'data', 'data.csv')
        test_settings_file = os.path.join(input_dir, 'test', 'settings', "data.json")
        test_weights_file = os.path.join(input_dir, 'test', 'weights', "data.weights")
        test_labels_file = os.path.join(input_dir, 'test', 'labels', "data.labels")

        # read test data
        test_data = pd.read_csv(test_data_file)

        # read test settings
        with open(test_settings_file) as f:
            self.test_settings = json.load(f)

        # read test weights
        with open(test_weights_file) as f:
            test_weights = np.array(f.read().splitlines(), dtype=float)

        # read test labels
        with open(test_labels_file) as f:
            test_labels = np.array(f.read().splitlines(), dtype=float)

        self.test_set = {
            "data": test_data,
            "weights": test_weights,
            "labels": test_labels
        }

    def get_bootstraped_dataset(self, mu=1.0, tes=1.0, seed=42):

        temp_df = deepcopy(self.test_set["data"])
        temp_df["weights"] = self.test_set["weights"]
        temp_df["labels"] = self.test_set["labels"]

        # Apply systematics to the sampled data
        data_syst = Systematics(
            data=temp_df,
            tes=tes
        ).data

        # Apply weight scaling factor mu to the data
        data_syst['weights'][data_syst["labels"] == 1] *= mu

        data_syst.pop("labels")

        prng = RandomState(seed)
        new_weights = prng.poisson(lam=data_syst['weights'])

        data_syst['weights'] = new_weights

        return {
            "data": data_syst.drop("weights", axis=1),
            "weights": new_weights
        }

    def init_submission(self):
        print("[*] Initializing Submmited Model")
        self.model = Model(
            train_set=self.train_set,
            systematics=Systematics
        )

    def fit_submission(self):
        print("[*] Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self):
        print("[*] Calling predict method of submitted model")

        # get set indices (0-9)
        # set_indices = np.arange(0, 10)
        set_indices = np.arange(0, 1)
        # get test set indices per set (0-99)
        test_set_indices = np.arange(0, 100)

        # create a product of set and test set indices all combinations of tuples
        all_combinations = list(product(set_indices, test_set_indices))
        # randomly shuffle all combinations of indices
        np.random.shuffle(all_combinations)

        self.results_dict = {}
        for set_index, test_set_index in all_combinations:
            # random tes value (one per test set)
            tes = np.random.uniform(0.9, 1.1)
            # create a seed
            seed = (set_index*100) + test_set_index
            # get mu value of set from test settings
            set_mu = self.test_settings["ground_truth_mus"][set_index]

            # get bootstrapped dataset from the original test set
            test_set = self.get_bootstraped_dataset(mu=set_mu, tes=tes, seed=seed)

            predicted_dict = self.model.predict(test_set)
            predicted_dict["test_set_index"] = test_set_index

            print(f"[*] - mu_hat: {predicted_dict['mu_hat']} - delta_mu_hat: {predicted_dict['delta_mu_hat']} - p16: {predicted_dict['p16']} - p84: {predicted_dict['p84']}")

            if set_index not in self.results_dict:
                self.results_dict[set_index] = []
            self.results_dict[set_index].append(predicted_dict)

    def save_result(self):
        print("[*] Saving ingestion result")

        # loop over sets
        # for i in range(0, 10):
        for i in range(0, 1):
            set_result = self.results_dict[i]
            set_result.sort(key=lambda x: x['test_set_index'])
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
            result_file = os.path.join(output_dir, "result_"+str(i)+".json")
            with open(result_file, 'w') as f:
                f.write(json.dumps(ingestion_result_dict, indent=4))


if __name__ == '__main__':

    print("############################################")
    print("### Ingestion Program")
    print("############################################\n")

    # Init Ingestion
    ingestion = Ingestion()

    # Start timer
    ingestion.start_timer()

    # load train set
    ingestion.load_train_set()

    # load test set
    ingestion.load_test_set()

    # initialize submission
    ingestion.init_submission()

    # fit submission
    ingestion.fit_submission()

    # predict submission
    ingestion.predict_submission()

    # save result
    ingestion.save_result()

    # Stop timer
    ingestion.stop_timer()

    # Show duration
    ingestion.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
