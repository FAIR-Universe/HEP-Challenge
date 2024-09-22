import numpy as np
import pandas as pd
import json
import os
import requests
from zipfile import ZipFile
import logging
import io
from tqdm import tqdm

# Get the logging level from an environment variable, default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

test_set_settings = None

PUBLIC_DATA_URL = "https://www.codabench.org/datasets/download/b9e59d0a-4db3-4da4-b1f8-3f609d1835b2/"


class Data:
    """
    A class to represent a dataset.

    Parameters:
        * input_dir (str): The directory path of the input data.

    Attributes:
        * __train_set (dict): A dictionary containing the train dataset.
        * __test_set (dict): A dictionary containing the test dataset.
        * input_dir (str): The directory path of the input data.

    Methods:
        * load_train_set(): Loads the train dataset.
        * load_test_set(): Loads the test dataset.
        * get_train_set(): Returns the train dataset.
        * get_test_set(): Returns the test dataset.
        * delete_train_set(): Deletes the train dataset.
        * get_syst_train_set(): Returns the train dataset with systematic variations.
    """
    def __init__(self, input_dir):
        """
        Constructs a Data object.

        Parameters:
            input_dir (str): The directory path of the input data.
        """

        self.__train_set = None
        self.__test_set = None
        self.input_dir = input_dir

    def load_train_set(self):

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

        buffer = io.StringIO()
        self.__train_set["data"].info(buf=buffer, memory_usage="deep", verbose=False)
        info_str = "Training Data :\n" + buffer.getvalue()
        logger.debug(info_str)
        logger.info("Train data loaded successfully")

    def load_test_set(self):

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

        test_settings_file = os.path.join(
            self.input_dir, "test", "settings", "data.json"
        )
        with open(test_settings_file) as f:
            test_settings = json.load(f)

        self.ground_truth_mus = test_settings["ground_truth_mus"]
        
        for key in self.__test_set.keys():
            buffer = io.StringIO()
            self.__test_set[key].info(buf=buffer, memory_usage="deep", verbose=False)
            info_str = str(key) + ":\n" + buffer.getvalue()
            
            logger.debug(info_str)    
        
        logger.info("Test data loaded successfully")

    def get_train_set(self):
        """
        Returns the train dataset.

        Returns:
            dict: The train dataset.
        """
        return self.__train_set

    def get_test_set(self):
        """
        Returns the test dataset.

        Returns:
            dict: The test dataset.
        """
        return self.__test_set

    def delete_train_set(self):
        """
        Deletes the train dataset.
        """
        del self.__train_set

    def get_syst_train_set(
        self, tes=1.0, 
        jes=1.0, 
        soft_met=0.0, 
        ttbar_scale=None, 
        diboson_scale=None, 
        bkg_scale=None, 
        dopostprocess=False
    ):
        from systematics import systematics

        if self.__train_set is None:
            self.load_train_set()
        return systematics(self.__train_set, tes, jes, soft_met, ttbar_scale, diboson_scale, bkg_scale,dopostprocess=dopostprocess)


current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)


def Neurips2024_public_dataset():
    """
    Downloads and extracts the Neurips 2024 public dataset.

    Returns:
        Data: The path to the extracted input data.

    Raises:
        HTTPError: If there is an error while downloading the dataset.
        FileNotFoundError: If the downloaded dataset file is not found.
        zipfile.BadZipFile: If the downloaded file is not a valid zip file.
    """
    parent_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.dirname(parent_path)
    public_data_folder_path = os.path.join(current_path, "public_data")
    public_input_data_folder_path = os.path.join(current_path, "public_data", "input_data")
    public_data_zip_path = os.path.join(current_path, "public_data.zip")

    # Check if public_data dir exists
    if os.path.isdir(public_data_folder_path):
        # Check if public_data/input_data dir exists
        if os.path.isdir(public_input_data_folder_path):
            return Data(public_input_data_folder_path)
        else:
            logger.warning("public_data/input_dir directory not found")
            
    else:
        logger.warning("public_data directory not found")

    # Check if public_data.zip exists
    if not os.path.isfile(public_data_zip_path):
        logger.warning("public_data.zip does not exist")
        logger.info("Downloading public data, this may take few minutes")
        
        chunk_size = 1024 * 1024
        response = requests.get(PUBLIC_DATA_URL, stream=True)
        if response.status_code == 200:
            with open(public_data_zip_path, 'wb') as file:
                # Iterate over the response in chunks
                for chunk in tqdm(response.iter_content(chunk_size=chunk_size)):
                    # Filter out keep-alive new chunks
                    if chunk:
                        file.write(chunk)

    # Extract public_data.zip
    logger.info("Extracting public_data.zip")
    with ZipFile(public_data_zip_path, 'r') as zip_ref:
        zip_ref.extractall(public_data_folder_path)

    return Data(public_input_data_folder_path)
