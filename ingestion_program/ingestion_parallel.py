# ------------------------------------------
# Imports
# ------------------------------------------
import numpy as np
import os
import pandas as pd
from datetime import datetime as dt
import json
from itertools import product
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pickle
import logging

log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)


MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 30))
CHUNK_SIZE = 2
DEFAULT_INGESTION_SEED = 31415

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)
# for gpu in tf.config.list_physical_devices('GPU'):
#     tf.config.experimental.set_memory_growth(gpu, True)

# Here we set the environment variable to allow the GPU memory to grow
# rather than pre allocating. We use this rather than the API
# because we want to avoid loading tensorflow.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# initialize worker environment
_model = None


def _init_worker(using_tensorflow, pickled_model, device_queue):
    global _model

    # Get the device to use
    device = device_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    if using_tensorflow:
        import tensorflow as tf

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

    # Now that are framework parameters are set we can unpickle the model
    _model = pickle.loads(pickled_model)


_model = None

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


# Define a function to process a set of combinations, not an instance method
# to avoid pickling the instance and all its associated data.
def _process_combination(arrays, test_settings, initial_seed, combination):
    logger.debug(f"Processing combination: {combination}")
    dict_systematics = test_settings["systematics"]
    num_pseudo_experiments = test_settings["num_pseudo_experiments"]

    try:
        # Setup shared memory for the test set
        with SharedTestSet(arrays=arrays) as test_set:
            set_index, test_set_index = combination

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

            # get bootstrapped dataset from the original test set
            test_set = _generate_pseudo_exp_data(test_set,
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
            predicted_dict = {}
            # Call predict method of the model that was passed to the worker
            predicted_dict = _model.predict(test_set)
            predicted_dict["test_set_index"] = test_set_index


            logger.debug(
                f"mu_hat: {predicted_dict['mu_hat']} - delta_mu_hat: {predicted_dict['delta_mu_hat']} - p16: {predicted_dict['p16']} - p84: {predicted_dict['p84']}"
            )

            return (combination, predicted_dict)
    except Exception as e:
        logger.error(f"Error in _process_combination: {e}")
        raise e


# ------------------------------------------
# Shared test set class
# ------------------------------------------
class SharedTestSet:
    def __init__(self, arrays=None, test_set=None):
        self._data = {}
        self._sm = []
        self._owner = False

        if arrays is not None:
            self._load_arrays(arrays)

        if test_set is not None:
            self._load(test_set)
            self._owner = True

    def _shared_memory_name(self, dataset_key, column):
        return f"{dataset_key}_{column}"

    def _load_arrays(self, arrays):
        def _create_sm_array(name, dtype, shape):
            shm_b = shared_memory.SharedMemory(name=name, create=False)
            self._sm.append(shm_b)
            return np.ndarray(shape, dtype=dtype, buffer=shm_b.buf)

        for key, value in arrays.items():
            # Special case for DataFrame
            if isinstance(value, list):
                columns = {}
                for entry in value:
                    name = entry["name"]
                    dtype = entry["dtype"]
                    shape = entry["shape"]

                    array = _create_sm_array(self._shared_memory_name(key, name), dtype, shape)
                    columns[name] = array
                self._data[key] = pd.DataFrame(columns, copy=False)
                continue

            dtype = value.get("dtype")
            shape = value.get("shape")

            self._data[key] = _create_sm_array(key, dtype, shape)

    def _load(self, data_set):
        def _create_sm_array(name, dtype, shape, size):
            shm_b = shared_memory.SharedMemory(name=name, create=True, size=size)
            self._sm.append(shm_b)

            return np.ndarray(shape, dtype=dtype, buffer=shm_b.buf)

        def _data_frame_to_shared_memory(dataset_key, data_frame):
            d = {}
            for column in data_frame.columns:
                value = data_frame[column]
                size = value.nbytes

                d[column] = _create_sm_array(self._shared_memory_name(dataset_key, column), value.dtype, value.shape, size)
                d[column][:] = value

            return pd.DataFrame(d, copy=False)

        for key in data_set.keys():
            self._data[key] = _data_frame_to_shared_memory(key, data_set[key])

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    # context manager to close the shared memory
    def __enter__(self):
        return self

    def __exit__(self, *args):
        for sm_block in self._sm:
            sm_block.close()
            if self._owner:
                sm_block.unlink()

    def asdict(self):
        def _asdict(array):
            if isinstance(array, dict):
                d = {}
                for k, v in array.items():
                    d[k] = _asdict(v)

                return d
            elif isinstance(array, pd.DataFrame):
                arrays = []
                for column in array.columns:
                    arrays.append(
                        {
                            "name": column,
                            "dtype": array[column].dtype,
                            "shape": array[column].shape,
                        }
                    )

                return arrays
            else:
                return {"dtype": array.dtype, "shape": array.shape}

        return _asdict(self._data)


# ------------------------------------------
# Ingestion Class
# ------------------------------------------
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

    Methods:
        * start_timer: Start the timer for the ingestion process.
        * stop_timer: Stop the timer for the ingestion process.
        * get_duration: Get the duration of the ingestion process.
        * save_duration: Save the duration of the ingestion process to a file.
        * load_train_set: Load the training set.
        * init_submission: Initialize the submitted model.
        * fit_submission: Fit the submitted model.
        * predict_submission: Make predictions using the submitted model.
        * compute_result: Compute the ingestion result.
        * save_result: Save the ingestion result to a file.
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
            logger.warning("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            logger.warning("[-] Timer was never stopped. Returning None")
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
        from systematics import (
            systematics,
        )

        self.model = Model(get_train_set=self.load_train_set, systematics=systematics)
        self.data.delete_train_set()

    def fit_submission(self):
        """
        Fit the submitted model.
        """
        logger.info("Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self, test_settings, initial_seed=DEFAULT_INGESTION_SEED):
        """
        Make predictions using the submitted model.

        Args:
            test_settings (dict): The test settings.
        """
        logger.info("Calling predict method of submitted model with seed: %s", initial_seed)

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
        futures = []

        using_tensorflow = "tensorflow" in sys.modules

        test_set = self.data.get_test_set()
        del self.data

        with SharedTestSet(test_set=test_set) as test_set:
            mp_context = mp.get_context("spawn")

            # We want to round robin the devices. So we create a queue
            # and put the devices indexes in the queue. The workers will
            # then get the device index from the queue.
            import torch

            device_count = torch.cuda.device_count()
            devices = list(range(0, device_count))
            device_queue = mp_context.Queue()
            # round robin the devices
            for w in range(0, MAX_WORKERS):
                device_queue.put(devices[w % device_count])

            with ProcessPoolExecutor(
                mp_context=mp_context,
                max_workers=MAX_WORKERS,
                initializer=_init_worker,
                # We are pickling the model explicitly here rather than
                # letting multiprocessing do it implicitly, so we
                # initialize tensorflow parameters before the model potentially
                # initializes it.
                initargs=(
                    using_tensorflow,
                    pickle.dumps(self.model),
                    device_queue,
                ),
            ) as executor:
                # The description of the shared memory arrays for the test set
                test_set_sm_arrays = test_set.asdict()
                func = partial(
                    _process_combination,
                    test_set_sm_arrays,
                    test_settings,
                    initial_seed,
                )
                futures = executor.map(func, all_combinations, chunksize=CHUNK_SIZE)

                # Iterate over the futures
                for combination, predicted_dict in futures:
                    set_index, _ = combination

                    set_results = self.results_dict.setdefault(set_index, [])
                    set_results.append(predicted_dict)

        logger.info("All processes done")

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
