import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import sys

LHC_NUMBERS = {
    "ztautau": 3574068,
    "diboson": 13602,
    "ttbar": 159079,
    "htautau": 3639,
}

def from_parquet(data, file_read_loc):
    for file in os.listdir(file_read_loc):
        if file.endswith(".parquet"):
            file_path = os.path.join(file_read_loc, file)
            print(f"[*] -- Reading {file_path}")
            key = file.split(".")[0]
            if key in data:
                data[key] = pd.read_parquet(file_path, engine="pyarrow")
                print(f"[*] --- {key} : {data[key].shape}")
            else:
                print(f"Invalid key: {key}")

            

def from_csv(data, file_read_loc):
    for file in os.listdir(file_read_loc):
        if file.endswith(".csv"):
            file_path = os.path.join(file_read_loc, file)
            key = file.split(".")[0]
            if key in data:
                data[key] = pd.read_csv(file_path,dtype=np.float32,index_col= False)
            else:
                print(f"Invalid key: {key}")

def sample_data_generator(full_data):

    # Remove the "label" and "weights" columns from the data    
    test_set = {}
    train_set = {}
    print("\n[*] -- full_data")
    for key in full_data.keys():
        print(f"[*] --- {key} : {full_data[key].shape}")

        test_number  = full_data[key].shape[0] * 0.3
        train_set[key], test_set[key] = train_test_split(
            full_data[key], test_size=int(test_number), random_state=42
        )
                        
    return train_set, test_set 

def train_data_generator(data):

    for key in data.keys():
        weights = np.ones(data[key].shape[0]) * LHC_NUMBERS[key] / data[key].shape[0]
        data[key]["weights"] = weights

    train_list = []
    print("\n[*] -- train_set")
    for key in data.keys():
        print(f"[*] --- {key} : {type(key)}")
        data[key]["detailed_label"] = key
        h = "htautau"
        if key == h:
            data[key]["Label"] = 1
        else:
            data[key]["Label"] = 0
        train_list.append(data[key])
        print(f"[*] --- {key} : {data[key].shape}")


    train_df = pd.concat(train_list)
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    train_set = {
        "labels": train_df.pop("Label"),
        "weights": train_df.pop("weights"),
        "detailed_labels": train_df.pop("detailed_label"),
        "data": train_df,
    }

    return train_set

def save_train_data(data_set, file_write_loc, output_format="csv"):
        # Create directories to store the label and weight files
    train_label_path = os.path.join(file_write_loc,"input_data", "train", "labels")
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    train_weights_path = os.path.join(file_write_loc,"input_data", "train", "weights")
    if not os.path.exists(train_weights_path):
        os.makedirs(train_weights_path)

    train_data_path = os.path.join(file_write_loc,"input_data", "train", "data")
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    train_detailed_labels_path = os.path.join(file_write_loc,"input_data", "train", "detailed_labels")
    if not os.path.exists(train_detailed_labels_path):
        os.makedirs(train_detailed_labels_path)


    train_settings_path = os.path.join(file_write_loc,"input_data", "train", "settings")
    if not os.path.exists(train_settings_path):
        os.makedirs(train_settings_path)

    train_settings = {"tes": 1.0, "jes" : 1.0,"soft_met" :0.0, "ttbar_scale": 1.0, "diboson_scale": 1.0,"bkg_scale" : 1.0 ,"ground_truth_mu": 1.0}
    # Specify the file path
    Settings_file_path = os.path.join(train_settings_path, "data.json")

    # Save the settings to a JSON file
    with open(Settings_file_path, "w") as json_file:
        json.dump(train_settings, json_file, indent=4)


    if output_format == "csv" :
        train_data_path = os.path.join(train_data_path, "data.csv")
        data_set["data"].to_csv(train_data_path, index=False)
        
    elif output_format == "parquet" :
        train_data_path = os.path.join(train_data_path, "data.parquet")
        data_set["data"].to_parquet(train_data_path, index=False)

    # Save the label, detailed_labels and weight files for the training set
    train_labels_file = os.path.join(train_label_path,"data.labels")
    data_set["labels"].to_csv(train_labels_file, index=False, header=False)
        
    train_weights_file = os.path.join(train_weights_path,"data.weights")
    data_set["weights"].to_csv(train_weights_file, index=False, header=False)
    
    train_detailed_labels_file = os.path.join(train_detailed_labels_path,"data.detailed_labels")
    data_set["detailed_labels"].to_csv(train_detailed_labels_file, index=False, header=False)

    print("\n[*] -- Train data saved")

def save_test_data(data_set, file_write_loc, output_format="csv"):
    # Create directories to store the label and weight files
    reference_settings_path = os.path.join(file_write_loc, "reference_data", "settings")
    if not os.path.exists(reference_settings_path):
        os.makedirs(reference_settings_path)

    test_data_loc = os.path.join(file_write_loc,"input_data", "test", "data")
    if not os.path.exists(test_data_loc):
        os.makedirs(test_data_loc)

    test_settings_path = os.path.join(file_write_loc,"input_data", "test", "settings")
    if not os.path.exists(test_settings_path):
        os.makedirs(test_settings_path)

    for key in data_set.keys():
        
        if output_format == "csv" :
            if not os.path.exists(test_data_loc):
                os.makedirs(test_data_loc)
            test_data_path = os.path.join(test_data_loc, f"{key}_data.csv")

            data_set[key].to_csv(test_data_path, index=False)

        if output_format == "parquet" :
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
    full_data = {
        "ztautau": pd.DataFrame(),
        "diboson": pd.DataFrame(),
        "ttbar": pd.DataFrame(),
        "htautau": pd.DataFrame(),
    }

    if output_format == "parquet":
        from_parquet(full_data, file_read_loc)
    elif output_format == "csv":
        from_csv(full_data, file_read_loc)
    else:
        print("Invalid output format")
        raise ValueError


    # Generate the sample data
    sample_set = sample_data_generator(full_data)

    # Generate the training data
    train_set = train_data_generator(sample_set[0])

    # Save the training data
    save_train_data(train_set, file_write_loc, output_format="parquet")

    # Save the test data
    save_test_data(sample_set[1], file_write_loc, output_format="parquet")


if __name__ == "__main__":

    input_file_location = sys.argv[1]
    output_file_location = sys.argv[2]   
    output_format = sys.argv[3]

    public_data_gen(input_file_location, output_file_location, output_format)

    print("\n[*] -- Data generation complete")

