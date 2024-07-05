# ------------------------------------------
# Imports
# ------------------------------------------
import numpy as np
import os
from datetime import datetime as dt
import json
from itertools import product
import warnings

warnings.filterwarnings("ignore")
DEFAULT_INGESTION_SEED = 31415

# ------------------------------------------
# Ingestion Class
# ------------------------------------------
class Ingestion:
    """
    Class for handling the ingestion process.

    Args:
        data (object): The data object.

    Attributes:
        start_time (datetime): The start time of the ingestion process.
        end_time (datetime): The end time of the ingestion process.
        model (object): The model object.
        data (object): The data object.
    """

    def __init__(self, data=None):
        """
        Initialize the Ingestion class.

        Args:
            data (object): The data object.
        """
        self.start_time = None
        self.end_time = None
        self.model = None
        self.data = data

    def start_timer(self):
        """
        Start the timer for the ingestion process.
        """
        self.start_time = dt.now()

    def stop_timer(self):
        """
        Stop the timer for the ingestion process.
        """
        self.end_time = dt.now()

    def get_duration(self):
        """
        Get the duration of the ingestion process.

        Returns:
            timedelta: The duration of the ingestion process.
        """
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stopped. Returning None")
            return None

        return self.end_time - self.start_time

    def show_duration(self):
        """
        Show the duration of the ingestion process.
        """
        print("\n---------------------------------")
        print(f"[✔] Total duration: {self.get_duration()}")
        print("---------------------------------")

    def save_duration(self, output_dir=None):
        """
        Save the duration of the ingestion process to a file.

        Args:
            output_dir (str): The output directory to save the duration file.
        """
        duration = self.get_duration()
        duration_in_mins = int(duration.total_seconds() / 60)
        duration_file = os.path.join(output_dir, "ingestion_duration.json")
        if duration is not None:
            with open(duration_file, "w") as f:
                f.write(json.dumps({"ingestion_duration": duration_in_mins}, indent=4))

    def load_train_set(self):
        """
        Load the training set.

        Returns:
            object: The loaded training set.
        """
        self.data.load_train_set()
        return self.data.get_train_set()

    def init_submission(self, Model):
        """
        Initialize the submitted model.

        Args:
            Model (object): The model class.
        """
        print("[*] Initializing Submmited Model")
        from systematics import systematics

        self.model = Model(get_train_set=self.load_train_set(), systematics=systematics)
        self.data.delete_train_set()

    def fit_submission(self):
        """
        Fit the submitted model.
        """
        print("[*] Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self, test_settings,initial_seed = DEFAULT_INGESTION_SEED):
        """
        Make predictions using the submitted model.

        Args:
            test_settings (dict): The test settings.
        """
        print("[*] Calling predict method of submitted model")

        dict_systematics = test_settings["systematics"]
        num_pseudo_experiments = test_settings["num_pseudo_experiments"]
        num_of_sets = test_settings["num_of_sets"]

        # get set indices
        set_indices = np.arange(0, num_of_sets)
        # get test set indices per set
        test_set_indices = np.arange(0, num_pseudo_experiments)

        # create a product of set and test set indices all combinations of tuples
        all_combinations = list(product(set_indices, test_set_indices))
        # randomly shuffle all combinations of indices
        random_state_initial = np.random.RandomState(initial_seed)
        random_state_initial.shuffle(all_combinations)

        self.results_dict = {}
        for set_index, test_set_index in all_combinations:

            # create a seed
            seed = (set_index * num_pseudo_experiments) + test_set_index + initial_seed

            # get mu value of set from test settings
            set_mu = test_settings["ground_truth_mus"][set_index]

            random_state = np.random.RandomState(seed)

            if dict_systematics["tes"]:
                tes = random_state.uniform(0.9, 1.1)
            else:
                tes = 1.0
            if dict_systematics["jes"]:
                jes = random_state.uniform(0.9, 1.1)
            else:
                jes = 1.0
            if dict_systematics["soft_met"]:
                soft_met = random_state.uniform(0.0, 5)
            else:
                soft_met = 0.0

            if dict_systematics["ttbar_scale"]:
                ttbar_scale = random_state.uniform(0.5, 2)
            else:
                ttbar_scale = None

            if dict_systematics["diboson_scale"]:
                diboson_scale = random_state.uniform(0.5, 2)
            else:
                diboson_scale = None

            if dict_systematics["bkg_scale"]:
                bkg_scale = random_state.uniform(0.995, 1.005)
            else:
                bkg_scale = None

            test_set = self.data.generate_psuedo_exp_data(
                set_mu=set_mu,
                tes=tes,
                jes=jes,
                soft_met=soft_met,
                ttbar_scale=ttbar_scale,
                diboson_scale=diboson_scale,
                bkg_scale=bkg_scale,
                seed=seed,
            )

            predicted_dict = self.model.predict(test_set)
            predicted_dict["test_set_index"] = test_set_index

            print(
                f"[*] - mu_hat: {predicted_dict['mu_hat']} - delta_mu_hat: {predicted_dict['delta_mu_hat']} - p16: {predicted_dict['p16']} - p84: {predicted_dict['p84']}"
            )

            if set_index not in self.results_dict:
                self.results_dict[set_index] = []
            self.results_dict[set_index].append(predicted_dict)

    def compute_result(self):
        """
        Compute the ingestion result.
        """
        print("[*] Saving ingestion result")

        # loop over sets
        for key in self.results_dict.keys():
            set_result = self.results_dict[key]
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
            self.results_dict[key] = ingestion_result_dict

    def save_result(self, output_dir=None):
        """
        Save the ingestion result to files.

        Args:
            output_dir (str): The output directory to save the result files.
        """
        for key in self.results_dict.keys():
            result_file = os.path.join(output_dir, "result_" + str(key) + ".json")
            with open(result_file, "w") as f:
                f.write(json.dumps(self.results_dict[key], indent=4))
