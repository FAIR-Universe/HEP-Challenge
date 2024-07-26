import os
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
import pandas as pd
import json
import sys

def from_parquet(file_read_loc):
    print("[*] Loading Train data")

    train_data_file = os.path.join(file_read_loc, "train", "data", "data.parquet")
    train_labels_file = os.path.join(file_read_loc, "train", "labels", "data.labels")
    train_settings_file = os.path.join(file_read_loc, "train", "settings", "data.json")
    train_weights_file = os.path.join(file_read_loc, "train", "weights", "data.weights")
    train_detailed_labels_file = os.path.join(
        file_read_loc, "train", "detailed_labels", "data.detailed_labels"
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

    train_set = {
        "data": pd.read_parquet(train_data_file, engine="pyarrow"),
        "labels": train_labels,
        "settings": train_settings,
        "weights": train_weights,
        "detailed_labels": train_detailed_labels,
    }

    del train_labels, train_settings, train_weights, train_detailed_labels

    print(train_set["data"].info(verbose=False, memory_usage="deep"))

    print("[+] Train data loaded successfully")

    return train_set


def test_data_generator(data):

    test_data_raw = data["data"].copy()
    test_data_raw["weights"] = data["weights"]

    test_set = {
        "ztautau": pd.DataFrame(test_data_raw[data["detailed_labels"] == "ztautau"]),
        "diboson": pd.DataFrame(test_data_raw[data["detailed_labels"] == "diboson"]),
        "ttbar": pd.DataFrame(test_data_raw[data["detailed_labels"] == "ttbar"]),
        "htautau": pd.DataFrame(test_data_raw[data["detailed_labels"] == "htautau"]),
    }

    return test_set


def save_train_data(data_set, file_write_loc, output_format="csv"):
    # Create directories to store the label and weight files
    train_label_path = os.path.join(file_write_loc, "input_data", "train", "labels")
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    train_weights_path = os.path.join(file_write_loc, "input_data", "train", "weights")
    if not os.path.exists(train_weights_path):
        os.makedirs(train_weights_path)

    train_data_path = os.path.join(file_write_loc, "input_data", "train", "data")
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    train_detailed_labels_path = os.path.join(
        file_write_loc, "input_data", "train", "detailed_labels"
    )
    if not os.path.exists(train_detailed_labels_path):
        os.makedirs(train_detailed_labels_path)

    train_settings_path = os.path.join(
        file_write_loc, "input_data", "train", "settings"
    )
    if not os.path.exists(train_settings_path):
        os.makedirs(train_settings_path)

    train_settings = {
        "tes": 1.0,
        "jes": 1.0,
        "soft_met": 0.0,
        "ttbar_scale": 1.0,
        "diboson_scale": 1.0,
        "bkg_scale": 1.0,
        "ground_truth_mu": 1.0,
    }
    # Specify the file path
    Settings_file_path = os.path.join(train_settings_path, "data.json")

    # Save the settings to a JSON file
    with open(Settings_file_path, "w") as json_file:
        json.dump(train_settings, json_file, indent=4)

    if output_format == "csv":
        train_data_path = os.path.join(train_data_path, "data.csv")
        data_set["data"].to_csv(train_data_path, index=False)

    elif output_format == "parquet":
        train_data_path = os.path.join(train_data_path, "data.parquet")
        data_set["data"].to_parquet(train_data_path, index=False)

    # Save the label, detailed_labels and weight files for the training set
    train_labels_file = os.path.join(train_label_path, "data.labels")
    data_set["labels"].to_csv(train_labels_file, index=False, header=False)

    train_weights_file = os.path.join(train_weights_path, "data.weights")
    data_set["weights"].to_csv(train_weights_file, index=False, header=False)

    train_detailed_labels_file = os.path.join(
        train_detailed_labels_path, "data.detailed_labels"
    )
    data_set["detailed_labels"].to_csv(
        train_detailed_labels_file, index=False, header=False
    )

    print("\n[*] -- Train data saved")


def save_test_data(data_set, file_write_loc, output_format="csv"):
    # Create directories to store the label and weight files
    reference_settings_path = os.path.join(file_write_loc, "reference_data", "settings")
    if not os.path.exists(reference_settings_path):
        os.makedirs(reference_settings_path)

    test_data_loc = os.path.join(file_write_loc, "input_data", "test", "data")
    if not os.path.exists(test_data_loc):
        os.makedirs(test_data_loc)

    test_settings_path = os.path.join(file_write_loc, "input_data", "test", "settings")
    if not os.path.exists(test_settings_path):
        os.makedirs(test_settings_path)

    for key in data_set.keys():

        if output_format == "csv":
            if not os.path.exists(test_data_loc):
                os.makedirs(test_data_loc)
            test_data_path = os.path.join(test_data_loc, f"{key}_data.csv")

            data_set[key].to_csv(test_data_path, index=False)

        if output_format == "parquet":
            if not os.path.exists(test_data_loc):
                os.makedirs(test_data_loc)
            test_data_path = os.path.join(test_data_loc, f"{key}_data.parquet")

            data_set[key].to_parquet(test_data_path, index=False)

    mu = np.random.uniform(0, 3, 10)
    mu = np.round(mu, 3)
    mu_list = mu.tolist()
    print(f"[*] --- mu in test set : ", mu_list)

    test_settings = {"ground_truth_mus": mu_list}
    Settings_file_path = os.path.join(reference_settings_path, "data.json")
    with open(Settings_file_path, "w") as json_file:
        json.dump(test_settings, json_file, indent=4)

    Settings_file_path = os.path.join(test_settings_path, "data.json")
    with open(Settings_file_path, "w") as json_file:
        json.dump(test_settings, json_file, indent=4)

    print("\n[*] -- Test data saved")


def public_data_gen(file_read_loc, file_write_loc, output_format="csv"):

    # Specify the location of the input data

    # Load the data from the input location

    if output_format == "parquet":
        full_data = from_parquet(file_read_loc)
    else:
        print("Invalid output format")
        raise ValueError

    # Generate the sample data
    _ , temp_set = train_test_split(full_data, test_size= 1000000, random_state=42, reweight=True)

    del full_data

    # Generate the training data
    train_set , test_set_raw = train_test_split(temp_set, test_size=0.2, random_state=42, reweight=True)

    test_set = test_data_generator(test_set_raw)

    # Save the training data
    save_train_data(train_set, file_write_loc, output_format="parquet")

    # Save the test data
    save_test_data(test_set, file_write_loc, output_format="parquet")


def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):
    data = data_set["data"].copy()
    train_set = {}
    test_set = {}
    full_size = len(data)

    print(f"Full size of the data is {full_size}")

    for key in data_set.keys():
        if (key != "data") and (key != "settings"):
            data[key] = data_set[key]
    

    train_data, test_data = sk_train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    for key in data_set.keys():
        if (key != "data") and (key != "settings"):
            train_set[key] = train_data.pop(key)
            test_set[key] = test_data.pop(key)

    train_set["data"] = train_data
    test_set["data"] = test_data

    if reweight is True:
        signal_weight = np.sum(data_set["weights"][data_set["labels"] == 1])
        background_weight = np.sum(data_set["weights"][data_set["labels"] == 0])
        signal_weight_train = np.sum(train_set["weights"][train_set["labels"] == 1])
        background_weight_train = np.sum(train_set["weights"][train_set["labels"] == 0])
        signal_weight_test = np.sum(test_set["weights"][test_set["labels"] == 1])
        background_weight_test = np.sum(test_set["weights"][test_set["labels"] == 0])

        train_set["weights"][train_set["labels"] == 1] = train_set["weights"][
            train_set["labels"] == 1
        ] * (signal_weight / signal_weight_train)
        test_set["weights"][test_set["labels"] == 1] = test_set["weights"][
            test_set["labels"] == 1
        ] * (signal_weight / signal_weight_test)

        train_set["weights"][train_set["labels"] == 0] = train_set["weights"][
            train_set["labels"] == 0
        ] * (background_weight / background_weight_train)
        test_set["weights"][test_set["labels"] == 0] = test_set["weights"][
            test_set["labels"] == 0
        ] * (background_weight / background_weight_test)

    return train_set, test_set


if __name__ == "__main__":

    input_file_location = sys.argv[1]
    output_file_location = sys.argv[2]
    output_format = sys.argv[3]

    public_data_gen(input_file_location, output_file_location, output_format)

    print("\n[*] -- Data generation complete")
