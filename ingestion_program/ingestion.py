# ------------------------------------------
# Imports
# ------------------------------------------
import numpy as np
import os
from datetime import datetime as dt
import json
from itertools import product
import logging

log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

DEFAULT_INGESTION_SEED = 31415

# ------------------------------------------
# Ingestion Class
# ------------------------------------------

def _generate_pseudo_exp_data(data, set_mu=1, tes=1.0, jes=1.0, soft_met=0.0, ttbar_scale=None, diboson_scale=None, bkg_scale=None, seed=0):

        from systematics import get_bootstrapped_dataset, get_systematics_dataset

        # get bootstrapped dataset from the original test set
        pesudo_exp_data = get_bootstrapped_dataset(
            data,
            mu=set_mu,
            ttbar_scale=ttbar_scale,
            diboson_scale=diboson_scale,
            bkg_scale=bkg_scale,
            seed=seed,
        )
        test_set = get_systematics_dataset(
            pesudo_exp_data,
            tes=tes,
            jes=jes,
            soft_met=soft_met,
        )

        return test_set

class Ingestion:
    """
    Class for handling the ingestion process.

    Args:
        data (object): The data object.

    Attributes:
        * start_time (datetime): The start time of the ingestion process.
        * end_time (datetime): The end time of the ingestion process.
        * model (object): The model object.
        * data (object): The data object.
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
            logger.warning("Timer was never started. Returning None")
            return None

        if self.end_time is None:
            logger.warning("Timer was never stopped. Returning None")
            return None

        return self.end_time - self.start_time

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

    def load_train_set(self,**kwargs):
        """
        Load the training set.

        Returns:
            object: The loaded training set.
        """
        self.data.load_train_set(**kwargs)
        return self.data.get_train_set()

    def init_submission(self, Model):
        """
        Initialize the submitted model.

        Args:
            Model (object): The model class.
        """
        logger.info("Initializing Submmited Model")
        from systematics import systematics

        self.model = Model(get_train_set=self.load_train_set, systematics=systematics)
        self.data.delete_train_set()

    def fit_submission(self):
        """
        Fit the submitted model.
        """
        logger.info("Fitting Submmited Model")
        self.model.fit()
        

    def predict_submission(self, test_settings,initial_seed = DEFAULT_INGESTION_SEED):
        """
        Make predictions using the submitted model.

        Args:
            test_settings (dict): The test settings.
        """
        logger.info("Calling predict method of submitted model with seed: %s", initial_seed)

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

        full_test_set = self.data.get_test_set()

        self.results_dict = {}
        for set_index, test_set_index in all_combinations:

            # create a seed
            seed = (set_index * num_pseudo_experiments) + test_set_index + initial_seed

            # get mu value of set from test settings
            set_mu = test_settings["ground_truth_mus"][set_index]

            random_state = np.random.RandomState(seed)

            if dict_systematics["tes"]:
                tes = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
            else:
                tes = 1.0
            if dict_systematics["jes"]:
                jes = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
            else:
                jes = 1.0
            if dict_systematics["soft_met"]:
                soft_met = np.clip(random_state.lognormal(mean=0.0, sigma=1.0), a_min=0.0, a_max=5.0)
            else:
                soft_met = 0.0

            if dict_systematics["ttbar_scale"]:
                ttbar_scale = np.clip(random_state.normal(loc=1.0, scale=0.02), a_min=0.8, a_max=1.2)
            else:
                ttbar_scale = None

            if dict_systematics["diboson_scale"]:
                diboson_scale = np.clip(random_state.normal(loc=1.0, scale=0.25), a_min=0.0, a_max=2.0)
            else:
                diboson_scale = None

            if dict_systematics["bkg_scale"]:
                bkg_scale = np.clip(random_state.normal(loc=1.0, scale=0.001), a_min=0.99, a_max=1.01)
            else:
                bkg_scale = None

            test_set = _generate_pseudo_exp_data(full_test_set,
                set_mu=set_mu,
                tes=tes,
                jes=jes,
                soft_met=soft_met,
                ttbar_scale=ttbar_scale,
                diboson_scale=diboson_scale,
                bkg_scale=bkg_scale,
                seed=seed,
            )

            logger.debug(
                f"set_index: {set_index} - test_set_index: {test_set_index} - seed: {seed}"
            )

            predicted_dict = self.model.predict(test_set)
            predicted_dict["test_set_index"] = test_set_index

            logger.debug(
                f"mu_hat: {predicted_dict['mu_hat']} - delta_mu_hat: {predicted_dict['delta_mu_hat']} - p16: {predicted_dict['p16']} - p84: {predicted_dict['p84']}"
            )

            if set_index not in self.results_dict:
                self.results_dict[set_index] = []
            self.results_dict[set_index].append(predicted_dict)

    def compute_result(self):
        """
        Compute the ingestion result.
        """
        logger.info("Computing Ingestion Result")

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
