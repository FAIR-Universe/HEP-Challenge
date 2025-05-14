import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import json
import os
import requests
from zipfile import ZipFile
import logging
import io

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

PUBLIC_DATA_URL = (
    "https://www.codabench.org/datasets/download/b9e59d0a-4db3-4da4-b1f8-3f609d1835b2/"
)

ZENODO_URL = "https://zenodo.org/records/15131565/files/FAIR_Universe_HiggsML_data.zip?download=1"


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

    def __init__(self, input_dir,test_size=0.3):
        """
        Constructs a Data object.

        Parameters:
            input_dir (str): The directory path of the input data.
        """

        self.__train_set = None
        self.__test_set = None
        
        train_data_file = os.path.join(input_dir, "FAIR_Universe_HiggsML_data.parquet")
        croissant_file = os.path.join(input_dir, "FAIR_Universe_HiggsML_data_metadata.json")
        
        try:
            with open(croissant_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            logger.warning("Metadata file not found. Proceeding without metadata.")
            self.metadata = {}
        except json.JSONDecodeError:
            logger.warning("Metadata file is not a valid JSON. Proceeding without metadata.")
            self.metadata = {}
        except Exception as e:
            logger.warning(f"An error occurred while reading the metadata file: {e}")
            self.metadata = {}

        self.parquet_file = pq.ParquetFile(train_data_file)

        # Step 1: Determine the total number of rows
        if "total_rows" in self.metadata:
            self.total_rows = self.metadata["total_rows"]
        else :
            # If total_rows is not in metadata, calculate it from the row groups
            self.total_rows = sum(self.parquet_file.metadata.row_group(i).num_rows for i in range(self.parquet_file.num_row_groups))        
        
        if test_size is not None:
            if isinstance(test_size, int):
                test_size = min(test_size, self.total_rows)
            elif isinstance(test_size, float):
                if 0.0 <= test_size <= 1.0:
                    test_size = int(test_size * self.total_rows)
                else:
                    raise ValueError("Test size must be between 0.0 and 1.0")
            else:
                raise ValueError("Test size must be an integer or a float")        
        
        self.test_size = test_size
        
        logger.info(f"Total rows: {self.total_rows}")
        logger.info(f"Test size: {self.test_size}")
        

    def load_train_set(self, train_size=None, selected_indices=None):
        if train_size is not None:
            if isinstance(train_size, int):
                train_size = min(train_size, self.total_rows - self.test_size)
            elif isinstance(train_size, float):
                if 0.0 <= train_size <= 1.0:
                    train_size = int(train_size * (self.total_rows - self.test_size))
                else:
                    raise ValueError("Sample size must be between 0.0 and 1.0")
            else:
                raise ValueError("Sample size must be an integer or a float")
        elif selected_indices is not None:
            if isinstance(selected_indices, list):
                selected_indices = np.array(selected_indices)
            elif isinstance(selected_indices, np.ndarray):
                pass
            else:
                raise ValueError("Selected indices must be a list or a numpy array")
            train_size = len(selected_indices)
        else:
            train_size = self.total_rows - self.test_size
            
        if train_size > self.total_rows - self.test_size:
            raise ValueError("Sample size exceeds the number of available rows")

        if selected_indices is None:
            selected_indices = np.random.choice((self.total_rows - self.test_size), size=train_size, replace=False)
        
        selected_train_indices = np.sort(selected_indices) + self.test_size
        
        logger.info(f"Selected train size: {len(selected_train_indices)}")
        
        
        # Step 2: Load the data
        self.__train_set = self.__load_data(selected_train_indices)
        
        # Balancing the weights 

        
        
    def __load_data(self, selected_indices):

        current_row = 0
        sampled_df = pd.DataFrame()
        for row_group_index in range(self.parquet_file.num_row_groups):
            row_group = self.parquet_file.read_row_group(row_group_index).to_pandas()
            row_group_size = len(row_group)

            # Determine indices within the current row group that fall in the selected range
            within_group_indices = selected_indices[(selected_indices >= current_row) & (selected_indices < current_row + row_group_size)] - current_row
            sampled_df = pd.concat([sampled_df, row_group.iloc[within_group_indices]], ignore_index=True)

            # Update the current row count
            current_row += row_group_size

        
        buffer = io.StringIO()
        sampled_df.info(buf=buffer, memory_usage="deep", verbose=False)
        info_str = "\n" + buffer.getvalue()
        logger.debug(info_str)
        logger.info("Data loaded successfully")
        
        if "sum_weights" in self.metadata:
            sum_weights = self.metadata["sum_weights"]
            if sum_weights > 0:
                sampled_df["weights"] = (sum_weights * sampled_df["weights"])/sum(sampled_df["weights"])
            else:
                logger.warning("Sum of weights is zero. No balancing applied.")
        
        return sampled_df

    def load_test_set(self):

        selected_test_indices = np.array(range(self.test_size))
        
        # Load the data
        test_df = self.__load_data(selected_test_indices)

        # read test setting
        test_set = {
            "ztautau": pd.DataFrame(),
            "diboson": pd.DataFrame(),
            "ttbar": pd.DataFrame(),
            "htautau": pd.DataFrame(),
        }

        for key in test_set.keys():

            test_set[key] = test_df[
                test_df["detailed_labels"] == key]
            test_set[key].pop("detailed_labels")
            test_set[key].pop("labels")

        self.__test_set = test_set

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
        train_set = self.__train_set
        return train_set

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
        self,
        tes=1.0,
        jes=1.0,
        soft_met=0.0,
        ttbar_scale=None,
        diboson_scale=None,
        bkg_scale=None,
        dopostprocess=False,
    ):
        from systematics import systematics

        if self.__train_set is None:
            self.load_train_set()
        return systematics(
            self.__train_set,
            tes,
            jes,
            soft_met,
            ttbar_scale,
            diboson_scale,
            bkg_scale,
            dopostprocess=dopostprocess,
        )


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
    public_input_data_folder_path = os.path.join(
        current_path, "public_data"
    )
    public_data_zip_path = os.path.join(current_path, "FAIR_Universe_HiggsML_data.zip")

    # Check if public_data dir exists
    if os.path.isdir(public_data_folder_path):
        # Check if public_data dir exists
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

        
        response = requests.get(ZENODO_URL, stream=True)
        response.raise_for_status()  # Will raise 403 if unauthorized

        logger.info("Status code: %s", response.status_code)

        # response = requests.get(PUBLIC_DATA_URL, stream=True)
        if response.status_code == 200:
            with open(public_data_zip_path, "wb") as file:
                # Iterate over the response in chunks
                for chunk in response.iter_content(chunk_size=chunk_size):
                    # Filter out keep-alive new chunks
                    if chunk:
                        file.write(chunk)
        else:
            logger.error(
                f"Failed to download the dataset. Status code: {response.status_code}"
            )
            raise requests.HTTPError(
                f"Failed to download the dataset. Status code: {response.status_code}"
            )
    else:
        logger.info("public_data.zip already exists")
        

    # Extract public_data.zip
    logger.info("Extracting FAIR_Universe_HiggsML_data.zip")
    with ZipFile(public_data_zip_path, "r") as zip_ref:
        zip_ref.extractall(public_data_folder_path)

    return Data(public_input_data_folder_path)
