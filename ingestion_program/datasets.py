
from frozendict import frozendict
import numpy as np
import pandas as pd
import json
import os
import subprocess


test_set_settings = None

class Data:

    def __init__(self, input_dir, data_format="csv"):

        self.__train_set = None
        self.__test_set = None
        self.data_format = data_format
        self.input_dir = input_dir
        
        print("==========================================")
        print("Data")
        print("==========================================")

    def load_train_set(self):
        print("[*] Loading Train data")

        if self.data_format == "csv":
            train_data_file = os.path.join(self.input_dir, "train", "data", "data.csv")
        if self.data_format == "parquet":
            train_data_file = os.path.join(self.input_dir, "train", "data", "data.parquet")
        train_labels_file = os.path.join(self.input_dir, "train", "labels", "data.labels")
        train_settings_file = os.path.join(self.input_dir, "train", "settings", "data.json")
        train_weights_file = os.path.join(self.input_dir, "train", "weights", "data.weights")
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

        if self.data_format == "parquet":
            self.__train_set = frozendict(
                {
                    "data": pd.read_parquet(train_data_file, engine="pyarrow"),
                    "labels": train_labels,
                    "settings": train_settings,
                    "weights": train_weights,
                    "detailed_labels": train_detailed_labels,
                }
            )
        else:
            self.__train_set = frozendict(
                {
                    "data": pd.read_csv(train_data_file),
                    "labels": train_labels,
                    "settings": train_settings,
                    "weights": train_weights,
                    "detailed_labels": train_detailed_labels,
                }
            )

        del train_labels, train_settings, train_weights, train_detailed_labels

        print(self.__train_set["data"].info(verbose=False, memory_usage="deep"))
        print("[*] Train data loaded successfully")

    def load_test_set(self):
        print("[*] Loading Test data")

        test_data_dir = os.path.join(self.input_dir, "test", "data")

        # read test setting
                
        test_set = {
            "ztautau": pd.DataFrame(),
            "wjets": pd.DataFrame(),
            "diboson": pd.DataFrame(),
            "ttbar": pd.DataFrame(),
            "htautau": pd.DataFrame(),
        }

        for key in test_set.keys():
            if self.data_format == "csv":
                test_data_path = os.path.join(test_data_dir, f"{key}_data.csv")
                test_set[key] = pd.read_csv(test_data_path)
                

            elif self.data_format == "parquet":
                test_data_path = os.path.join(test_data_dir, f"{key}_data.parquet")
                test_set[key] = pd.read_parquet(test_data_path, engine="pyarrow")


        self.__test_set = (test_set)
        print("[*] Test data loaded successfully")

    def get_train_set(self):
        return self.__train_set

    def get_test_set(self):
        return self.__test_set
    
    def delete_train_set(self):
        del self.__train_set
        
    def delete_test_set(self):
        del self.__test_set
        
    
    
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)


def Neurips2024_public_dataset():

    file_read_loc = os.path.join(parent_path, "public_data")
    file = "public_data.zip"
    if file not in os.listdir(file_read_loc):    
        subprocess.run(["wget","-O", os.path.join(file_read_loc, "public_data.zip"), "https://codalab.coresearch.club/my/datasets/download/0e2d7e8e-1b8b-4b3f-8b8b-3b3c5d4e4e3d"])
        
    if "input_data" not in os.listdir(file_read_loc):
        subprocess.run(["unzip", os.path.join(file_read_loc, file), "-d", file_read_loc])
           
    return Data(os.path.join(parent_path, "public_data", "input_data"), data_format="parquet")