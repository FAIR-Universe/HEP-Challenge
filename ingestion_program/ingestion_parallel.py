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
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pickle

warnings.filterwarnings("ignore")


MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 30))
CHUNK_SIZE = 2

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

def _generate_psuedo_exp_data(data, set_mu=1, tes=1.0, jes=1.0, soft_met=1.0, ttbar_scale=None, diboson_scale=None, bkg_scale=None, seed=0):

        from systematics import get_bootstraped_dataset, get_systematics_dataset

        # get bootstrapped dataset from the original test set
        pesudo_exp_data = get_bootstraped_dataset(
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
def _process_combination(arrays, test_settings, combination):
    print("[*] Processing combination")
    dict_systematics = test_settings["systematics"]
    num_pseudo_experiments = test_settings["num_pseudo_experiments"]

    try:
        # Setup shared memory for the test set
        with SharedTestSet(arrays=arrays) as test_set:
            set_index, test_set_index = combination

            # random tes value (one per test set)
            # random tes value (one per test set)
            if dict_systematics["tes"]:
                tes = np.random.uniform(0.9, 1.1)
            else:
                tes = 1.0
            if dict_systematics["jes"]:
                jes = np.random.uniform(0.9, 1.1)
            else:
                jes = 1.0
            if dict_systematics["soft_met"]:
                soft_met = np.random.uniform(0.0, 5)
            else:
                soft_met = 0.0

            if dict_systematics["ttbar_scale"]:
                ttbar_scale = np.random.uniform(0.5, 2)
            else:
                ttbar_scale = None
                
            if dict_systematics["diboson_scale"]:
                diboson_scale = np.random.uniform(0.5, 2)
            else:
                diboson_scale = None

            if dict_systematics["bkg_scale"]:
                bkg_scale = np.random.uniform(0.995, 1.005)
            else:
                bkg_scale = None

            seed = (set_index * num_pseudo_experiments) + test_set_index
            # get mu value of set from test settings
            set_mu = test_settings["ground_truth_mus"][set_index]

            # get bootstrapped dataset from the original test set
            test_set = _generate_psuedo_exp_data(test_set,
                set_mu=set_mu,
                tes=tes,
                jes=jes,
                soft_met=soft_met,
                ttbar_scale=ttbar_scale,
                diboson_scale=diboson_scale,
                bkg_scale=bkg_scale,
                seed=seed,
            )
            # print(f"[*] Predicting process with seed {seed}")
            predicted_dict = {}
            # Call predict method of the model that was passed to the worker
            predicted_dict = _model.predict(test_set)
            predicted_dict["test_set_index"] = test_set_index

            print(
                f"[*] - mu_hat: {predicted_dict['mu_hat']} - delta_mu_hat: {predicted_dict['delta_mu_hat']} - p16: {predicted_dict['p16']} - p84: {predicted_dict['p84']}"
            )

            return (combination, predicted_dict)
    except Exception as e:
        print(f"[-] Error in _process_combination: {e}")
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
        start_time (datetime): The start time of the ingestion process.
        end_time (datetime): The end time of the ingestion process.
        model (object): The model object.
        data (object): The data object.

    Methods:
        start_timer: Start the timer for the ingestion process.
        stop_timer: Stop the timer for the ingestion process.
        get_duration: Get the duration of the ingestion process.
        show_duration: Display the duration of the ingestion process.
        save_duration: Save the duration of the ingestion process to a file.
        load_train_set: Load the training set.
        init_submission: Initialize the submitted model.
        fit_submission: Fit the submitted model.
        predict_submission: Make predictions using the submitted model.
        compute_result: Compute the ingestion result.
        save_result: Save the ingestion result to a file.
    """

    def __init__(self, data=None):
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
        Display the duration of the ingestion process.
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
            Model (class): The model class.

        Notes:
            This method initializes the submitted model by calling the `fit` method.

        Raises:
            ImportError: If the `systematics` module is not found.
        """
        print("[*] Initializing Submitted Model")
        from systematics import systematics

        self.model = Model(get_train_set=self.load_train_set(), systematics=systematics)
        self.data.delete_train_set()

    def fit_submission(self):
        """
        Fit the submitted model.

        Notes:
            This method calls the `fit` method of the submitted model.
        """
        print("[*] Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self, test_settings):
        """
        Make predictions using the submitted model.

        Args:
            test_settings (dict): The test settings.

        Notes:
            This method calls the `predict` method of the submitted model.
        """
        print("[*] Calling predict method of submitted model")

        num_pseudo_experiments = test_settings["num_pseudo_experiments"]
        num_of_sets = test_settings["num_of_sets"]

        # Rest of the code...

    def compute_result(self):
        """
        Compute the ingestion result.

        Notes:
            This method computes the ingestion result based on the predicted values.
        """
        print("[*] Saving ingestion result")

        # Rest of the code...

    def save_result(self, output_dir=None):
        """
        Save the ingestion result to a file.

        Args:
            output_dir (str): The output directory to save the result files.
        """
        for key in self.results_dict.keys():
            result_file = os.path.join(output_dir, "result_" + str(key) + ".json")
            with open(result_file, "w") as f:
                f.write(json.dumps(self.results_dict[key], indent=4))

if __name__ == "__main__":
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

    # Save duration
    ingestion.save_duration()

    print("\n----------------------------------------------")
    print("[✔] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
