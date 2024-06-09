import numpy as np
import pandas as pd
import json
import os
import requests
from pathlib import Path
from zipfile import ZipFile


test_set_settings = None


class Data:

    def __init__(self, input_dir):

        self.__train_set = None
        self.__test_set = None
        self.input_dir = input_dir

    def load_train_set(self):
        print("[*] Loading Train data")

        train_data_file = os.path.join(self.input_dir, "train", "data", "data.parquet")
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

        self.__train_set = {
                "data": pd.read_parquet(train_data_file, engine="pyarrow"),
                "labels": train_labels,
                "settings": train_settings,
                "weights": train_weights,
                "detailed_labels": train_detailed_labels,
            }

        del train_labels, train_settings, train_weights, train_detailed_labels

        print(self.__train_set["data"].info(verbose=False, memory_usage="deep"))

        test_settings_file = os.path.join(
            self.input_dir, "test", "settings", "data.json"
        )
        with open(test_settings_file) as f:
            test_settings = json.load(f)

        self.ground_truth_mus = test_settings["ground_truth_mus"]

        print("[*] Train data loaded successfully")

    def load_test_set(self):
        print("[*] Loading Test data")

        test_data_dir = os.path.join(self.input_dir, "test", "data")

        # read test setting
        test_set = {
            "ztautau": pd.DataFrame(),
            "diboson": pd.DataFrame(),
            "ttbar": pd.DataFrame(),
            "htautau": pd.DataFrame(),
        }

        for key in test_set.keys():

            test_data_path = os.path.join(test_data_dir, f"{key}_data.parquet")
            test_set[key] = pd.read_parquet(test_data_path, engine="pyarrow")

        self.__test_set = test_set

        print("[*] Test data loaded successfully")

    def generate_psuedo_exp_data(
        self,
        set_mu=1,
        tes=1.0,
        jes=1.0,
        soft_met=0.0,
        ttbar_scale=None,
        diboson_scale=None,
        bkg_scale=None,
        seed=42,
    ):
        from systematics import get_bootstraped_dataset, get_systematics_dataset

        # get bootstrapped dataset from the original test set
        pesudo_exp_data = get_bootstraped_dataset(
            self.__test_set,
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

    def get_train_set(self):
        return self.__train_set

    def get_test_set(self):
        return self.__test_set

    def delete_train_set(self):
        del self.__train_set

    def get_syst_train_set(
        self, tes=1.0, jes=1.0, soft_met=0.0, ttbar_scale=None, diboson_scale=None, bkg_scale=None
    ):
        from systematics import systematics

        if self.__train_set is None:
            self.load_train_set()
        return systematics(self.__train_set, tes, jes, soft_met, ttbar_scale, diboson_scale, bkg_scale)


current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)


def Neurips2024_public_dataset():
    current_path = Path.cwd()
    file_read_loc = current_path / "public_data"
    if not file_read_loc.exists():
        file_read_loc.mkdir()

    url = "https://www.codabench.org/datasets/download/9c99a23c-f199-405a-b795-b42ea2dd652d/"
    file = file_read_loc / "public_data.zip"
    chunk_size = 1024 * 1024
    if not file.exists():
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with file.open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

    input_data = file_read_loc / "input_data"
    if not input_data.exists():
        with ZipFile(file) as zip:
            zip.extractall(path=file_read_loc)

    return Data(
        str(current_path / "public_data" / "input_data")
    )
