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
                soft_met = np.random.uniform(1.0, 5)
            else:
                soft_met = 1.0

            if dict_systematics["w_scale"]:
                w_scale = np.random.uniform(0.5, 2)
            else:
                w_scale = None

            if dict_systematics["bkg_scale"]:
                bkg_scale = np.random.uniform(0.5, 2)
            else:
                bkg_scale = None

            seed = (set_index * num_pseudo_experiments) + test_set_index
            # get mu value of set from test settings
            set_mu = test_settings["ground_truth_mus"][set_index]

            # get bootstrapped dataset from the original test set
            test_set = test_set.generate_psuedo_exp_data(
                set_mu=set_mu,
                tes=tes,
                jes=jes,
                soft_met=soft_met,
                w_scale=w_scale,
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
        self._keys = []
        self._columns = []

        if arrays is not None:
            self._load_arrays(arrays)

        if test_set is not None:
            self._load(test_set)
            self._owner = True

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

                    array = _create_sm_array(name, dtype, shape)
                    columns[name] = array
                self._data[key] = pd.DataFrame(columns, copy=False)
                continue

            dtype = value.get("dtype")
            shape = value.get("shape")

            self._data[key] = _create_sm_array(key, dtype, shape)

    def _load(self,data_set):
        def _create_sm_array(name, dtype, shape, size):
            shm_b = shared_memory.SharedMemory(name=name, create=True, size=size)
            self._sm.append(shm_b)

            return np.ndarray(shape, dtype=dtype, buffer=shm_b.buf)

        # data
        for key in data_set.keys():
            self._keys.append(key)
            d = {}
            data = data_set[key]
            for column in data.columns:
                value = data[column]
                self._columns.append(column)
                new_column = f"{key}_" + column
                
                size = value.nbytes

                d[new_column] = _create_sm_array(new_column, value.dtype, value.shape, size)
                d[new_column][:] = value

            self._data[key] = pd.DataFrame(d, copy=False)

    def keys(self):
        return self._keys
    
    def generate_psuedo_exp_data(self, set_mu=1, tes=1.0, jes=1.0, soft_met=1.0, w_scale=None, bkg_scale=None, seed=42):
        
        from systematics import get_bootstraped_dataset, get_systematics_dataset

        # get bootstrapped dataset from the original test set
        pesudo_exp_data = get_bootstraped_dataset(
            self._data,
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
        
        return test_set

    # def __getitem__(self, key):
    #     return self._data[key]

    def __getitem__(self, key=None):
        out_data = pd.DataFrame()
        for column in self._columns:
            new_column = f"{key}_{column}"
            out_data[column] = self._data[new_column]
        return out_data
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
    def __init__(self, data=None):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.model = None
        self.data = data

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

    def load_train_set(self):
        self.data.load_train_set()
        return self.data.get_train_set()
    
    def load_test_set(self):
        self.data.load_test_set()
        self.test_set = self.data.get_test_set()

    def init_submission(self, Model):
        print("[*] Initializing Submmited Model")
        from systematics import (
            systematics,
        )

        self.model = Model(get_train_set=self.load_train_set(), systematics=systematics)
        self.data.delete_train_set()

    def fit_submission(self):
        print("[*] Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self, test_settings):
        print("[*] Calling predict method of submitted model")
        self.load_test_set()

        num_pseudo_experiments = test_settings["num_pseudo_experiments"]
        num_of_sets = test_settings["num_of_sets"]

        # get set indices
        set_indices = np.arange(0, num_of_sets)
        # get test set indices per set
        test_set_indices = np.arange(0, num_pseudo_experiments)

        # create a product of set and test set indices all combinations of tuples
        all_combinations = list(product(set_indices, test_set_indices))
        # randomly shuffle all combinations of indices
        np.random.shuffle(all_combinations)

        self.results_dict = {}
        futures = []

        using_tensorflow = "tensorflow" in sys.modules

        with SharedTestSet(test_set=self.test_set) as test_set:
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
                )
                futures = executor.map(func, all_combinations, chunksize=CHUNK_SIZE)

                # Iterate over the futures
                for combination, predicted_dict in futures:
                    set_index, _ = combination

                    set_results = self.results_dict.setdefault(set_index, [])
                    set_results.append(predicted_dict)

        print("[*] All processes done")

    def compute_result(self):
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
