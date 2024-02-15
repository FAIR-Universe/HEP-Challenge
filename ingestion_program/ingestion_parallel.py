# ------------------------------------------
# Imports
# ------------------------------------------
import numpy as np
import os
import pandas as pd
from datetime import datetime as dt
import json
from itertools import product
from numpy.random import RandomState
import warnings
from copy import deepcopy
from multiprocessing import Process, Manager
import sys
warnings.filterwarnings("ignore")


# ------------------------------------------
# Settings
# ------------------------------------------
# True when running on Codabench
# False when running locally
CODABENCH = False
NUM_SETS = 1  # Total = 10
NUM_PSEUDO_EXPERIMENTS = 100  # Total = 100
USE_SYSTEAMTICS = True
NUM_PROCESS = 3


# ------------------------------------------
# Ingestion Class
# ------------------------------------------
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

    def set_directories(self):

        # set default directories
        module_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir_name = os.path.dirname(module_dir)

        input_data_dir_name = "input_data"
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

        train_data_file = os.path.join(self.input_dir, 'train', 'data', 'data.csv')
        train_labels_file = os.path.join(self.input_dir, 'train', 'labels', 'data.labels')
        train_settings_file = os.path.join(self.input_dir, 'train', 'settings', 'data.json')
        train_weights_file = os.path.join(self.input_dir, 'train', 'weights', 'data.weights')

        # read train labels
        with open(train_labels_file, "r") as f:
            train_labels = np.array(f.read().splitlines(), dtype=float)

        # read train settings
        with open(train_settings_file) as f:
            train_settings = json.load(f)

        # read train weights
        with open(train_weights_file) as f:
            train_weights = np.array(f.read().splitlines(), dtype=float)
        train_weights = train_weights

        self.train_set = {
            "data": pd.read_csv(train_data_file, dtype=np.float32),
            "labels": train_labels,
            "settings": train_settings,
            "weights": train_weights
        }

        del train_labels, train_settings, train_weights
        print(self.train_set["data"].info(verbose=False, memory_usage="deep"))
        print("[*] Train data loaded successfully")

    def load_test_set(self):
        print("[*] Loading Test data")

        test_data_file = os.path.join(self.input_dir, 'test', 'data', 'data.csv')
        test_settings_file = os.path.join(self.input_dir, 'test', 'settings', 'data.json')
        test_weights_file = os.path.join(self.input_dir, 'test', 'weights', 'data.weights')
        test_labels_file = os.path.join(self.input_dir, 'test', 'labels', 'data.labels')

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
            "data": pd.read_csv(test_data_file),
            "weights": test_weights,
            "labels": test_labels
        }
        del test_weights, test_labels

        print(self.test_set["data"].info(verbose=False, memory_usage="deep"))
        print("[*] Test data loaded successfully")

    def get_bootstraped_dataset(self, mu=1.0, tes=1.0, seed=42):

        temp_df = deepcopy(self.test_set["data"])
        temp_df["weights"] = self.test_set["weights"]
        temp_df["labels"] = self.test_set["labels"]

        # Apply systematics to the sampled data
        from systematics import Systematics
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

        del temp_df

        return {
            "data": data_syst.drop("weights", axis=1),
            "weights": new_weights
        }

    def init_submission(self):
        print("[*] Initializing Submmited Model")
        from model import Model
        from systematics import Systematics
        self.model = Model(
            train_set=self.train_set,
            systematics=Systematics
        )

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

        manager = Manager()
        return_dict = manager.dict()

        self.results_dict = {}
        
        # Define a function to process each combination in parallel
        def process_combination(combination, return_dict):
            set_index, test_set_index = combination
            # random tes value (one per test set)
            tes = np.random.uniform(0.9, 1.1)
            # create a seed
            seed = (set_index*100) + test_set_index
            # get mu value of set from test settings
            set_mu = self.test_settings["ground_truth_mus"][set_index]

            # get bootstrapped dataset from the original test set
            test_set = self.get_bootstraped_dataset(mu=set_mu, tes=tes, seed=seed)
            print (f"[*] Predicting process")
            predicted_dict = {}
            predicted_dict = self.model.predict(test_set)
            predicted_dict["test_set_index"] = test_set_index

            print(f"[*] - mu_hat: {predicted_dict['mu_hat']} - delta_mu_hat: {predicted_dict['delta_mu_hat']} - p16: {predicted_dict['p16']} - p84: {predicted_dict['p84']}")

            return_dict[seed] = predicted_dict
            return 0

        # Create a multiprocessing pool with 5 processes
          # List to hold the pools
        total_num = len(all_combinations)

        reminder = total_num % NUM_PROCESS
        for i in range(0, int(total_num/NUM_PROCESS)):
            some_combinations = all_combinations[i*NUM_PROCESS: (i+1)*NUM_PROCESS]
            pools = []
            for combination in some_combinations:
                pool = Process(target=process_combination, args= (combination, return_dict))
                pool.start()
                print(f"[*] Started process for combination: {combination}")
                pools.append(pool)
                

            # for combination in all_combinations:
            #     process_combination(combination)
            for pool in pools:
                pool.join()
        
            pool.close()

        if reminder > 0:
            some_combinations = all_combinations[-reminder:]
            pools = []
            for combination in some_combinations:
                pool = Process(target=process_combination, args= (combination, return_dict))
                pool.start()
                print(f"[*] Started process for combination: {combination}")
                pools.append(pool)
                

            # for combination in all_combinations:
            #     process_combination(combination)
            for pool in pools:
                pool.join()
        
            pool.close()

        for set_index in set_indices:
            for test_set_index in test_set_indices:
                seed = (set_index*100) + test_set_index
                if set_index not in self.results_dict:
                    self.results_dict[set_index] = []
                self.results_dict[set_index].append(return_dict[seed])


        print("[*] All processes done")

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
            result_file = os.path.join(self.output_dir, "result_"+str(i)+".json")
            with open(result_file, 'w') as f:
                f.write(json.dumps(ingestion_result_dict, indent=4))


if __name__ == '__main__':

    print("############################################")
    print("### Parallel Ingestion Program")
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

    print("\n----------------------------------------------")
    print("[✔] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
