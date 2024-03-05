import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import sys

def public_data_gen():

    data_location = sys.argv[1]

    input_dir = os.path.join(data_location, 'input_data')
    output_dir = os.path.join(data_location, 'public_data')
    sample_dir = os.path.join(data_location, 'sample_data')

    train_data_file = os.path.join(input_dir, 'train', 'data', 'data.parquet')
    train_labels_file = os.path.join(input_dir, 'train', 'labels', 'data.labels')
    train_settings_file = os.path.join(input_dir, 'train', 'settings', 'data.json')
    train_weights_file = os.path.join(input_dir, 'train', 'weights', 'data.weights')

    # read train labels
    with open(train_labels_file, "r") as f:
        train_labels = np.array(f.read().splitlines(), dtype=float)

    # read train settings
    with open(train_settings_file) as f:
        train_settings = json.load(f)

    # read train weights
    with open(train_weights_file) as f:
        train_weights = np.array(f.read().splitlines(), dtype=float)

    # read train data
    train_df = pd.read_parquet(train_data_file, engine='pyarrow')
    signal_weights = train_weights[train_labels == 1].sum()
    background_weights = train_weights[train_labels == 0].sum()

    # Split the data into training and validation sets while preserving the proportion of samples with respect to the target variable
    train_df, valid_df, train_labels, valid_labels, train_weights, valid_weights = train_test_split(
        train_df,
        train_labels,
        train_weights,
        test_size=0.5,
        stratify=train_labels
    )

    train_df, sample_set_df, train_labels, sample_set_labels, train_weights, sample_set_weights = train_test_split(
        train_df,
        train_labels,
        train_weights,
        test_size=0.01,
        shuffle=True,
        stratify=train_labels
    )

    sample_test_df, sample_train_df, sample_test_labels, sample_train_labels, sample_test_weights, sample_train_weights = train_test_split(
        sample_set_df,
        sample_set_labels,
        sample_set_weights,
        test_size=0.5,
        shuffle=True,
        stratify=sample_set_labels
    )
    # Calculate the sum of weights for signal and background in the training and validation sets
    train_signal_weights = train_weights[train_labels == 1].sum()
    train_background_weights = train_weights[train_labels == 0].sum()
    valid_signal_weights = valid_weights[valid_labels == 1].sum()
    valid_background_weights = valid_weights[valid_labels == 0].sum()
    sample_train_signal_weights = sample_train_weights[sample_train_labels == 1].sum()
    sample_train_background_weights = sample_train_weights[sample_train_labels == 0].sum()
    sample_test_signal_weights = sample_test_weights[sample_test_labels == 1].sum()
    sample_test_background_weights = sample_test_weights[sample_test_labels == 0].sum()

    # Balance the sum of weights for signal and background in the training and validation sets
    train_weights[train_labels == 1] *= signal_weights / train_signal_weights
    train_weights[train_labels == 0] *= background_weights / train_background_weights
    valid_weights[valid_labels == 1] *= signal_weights / valid_signal_weights
    valid_weights[valid_labels == 0] *= background_weights / valid_background_weights
    sample_train_weights[sample_train_labels == 1] *= signal_weights / sample_train_signal_weights
    sample_train_weights[sample_train_labels == 0] *= background_weights / sample_train_background_weights
    sample_test_weights[sample_test_labels == 1] *= signal_weights / sample_test_signal_weights
    sample_test_weights[sample_test_labels == 0] *= background_weights / sample_test_background_weights


    train_set = {
        "data": train_df,
        "labels": train_labels,
        "weights": train_weights,
        "settings": train_settings
    }

    valid_set = {
        "data": valid_df,
        "labels": valid_labels,
        "weights": valid_weights
    }

    sample_train_set = {
        "data": sample_train_df,
        "labels": sample_train_labels,
        "weights": sample_train_weights,
        "settings": train_settings
    }

    sample_test_set = {
        "data": sample_test_df,
        "labels": sample_test_labels,
        "weights": sample_test_weights
    }

    
    if not os.path.exists(os.path.join(output_dir, 'train', 'data')):
        os.makedirs(os.path.join(output_dir, 'train', 'data'))

    if not os.path.exists(os.path.join(output_dir, 'train', 'labels')):
        os.makedirs(os.path.join(output_dir, 'train', 'labels'))

    if not os.path.exists(os.path.join(output_dir, 'train', 'weights')):
        os.makedirs(os.path.join(output_dir, 'train', 'weights'))

    if not os.path.exists(os.path.join(output_dir, 'train', 'settings')):
        os.makedirs(os.path.join(output_dir, 'train', 'settings'))

    with open(os.path.join(output_dir, 'train', 'settings', 'data.json'), 'w') as f:
        json.dump(train_set["settings"], f)

    train_set["data"].to_parquet(os.path.join(output_dir, 'train', 'data', 'data.parquet'), index=False)
    np.savetxt(os.path.join(output_dir, 'train', 'labels', 'data.labels'), train_labels, fmt='%f')
    np.savetxt(os.path.join(output_dir, 'train', 'weights', 'data.weights'), train_weights, fmt='%f')

    if not os.path.exists(os.path.join(output_dir, 'valid', 'data')):
        os.makedirs(os.path.join(output_dir, 'valid', 'data'))

    if not os.path.exists(os.path.join(output_dir, 'valid', 'labels')):
        os.makedirs(os.path.join(output_dir, 'valid', 'labels'))

    if not os.path.exists(os.path.join(output_dir, 'valid', 'weights')):
        os.makedirs(os.path.join(output_dir, 'valid', 'weights'))

    if not os.path.exists(os.path.join(output_dir, 'valid', 'settings')):
        os.makedirs(os.path.join(output_dir, 'valid', 'settings'))

    valid_set["data"].to_parquet(os.path.join(output_dir, 'valid', 'data', 'data.parquet'), index=False)
    np.savetxt(os.path.join(output_dir, 'valid', 'labels', 'data.labels'), valid_set["labels"], fmt='%f')
    np.savetxt(os.path.join(output_dir, 'valid', 'weights', 'data.weights'), valid_set["weights"], fmt='%f')

    print("Public data generated")


    if not os.path.exists(os.path.join(sample_dir, 'train', 'data')):
        os.makedirs(os.path.join(sample_dir, 'train', 'data'))

    if not os.path.exists(os.path.join(sample_dir, 'train', 'labels')):
        os.makedirs(os.path.join(sample_dir, 'train', 'labels'))

    if not os.path.exists(os.path.join(sample_dir, 'train', 'weights')):
        os.makedirs(os.path.join(sample_dir, 'train', 'weights'))

    if not os.path.exists(os.path.join(sample_dir, 'train', 'settings')):
        os.makedirs(os.path.join(sample_dir, 'train', 'settings'))

    with open(os.path.join(sample_dir, 'train', 'settings', 'data.json'), 'w') as f:
        json.dump(sample_train_set["settings"], f)

    sample_train_set["data"].to_parquet(os.path.join(sample_dir, 'train', 'data', 'data.parquet'), index=False)
    np.savetxt(os.path.join(sample_dir, 'train', 'labels', 'data.labels'), sample_train_labels, fmt='%f')
    np.savetxt(os.path.join(sample_dir, 'train', 'weights', 'data.weights'), sample_train_weights, fmt='%f')

    if not os.path.exists(os.path.join(sample_dir, 'test', 'data')):
        os.makedirs(os.path.join(sample_dir, 'test', 'data'))
        
    if not os.path.exists(os.path.join(sample_dir, 'test', 'labels')):
        os.makedirs(os.path.join(sample_dir, 'test', 'labels'))

    if not os.path.exists(os.path.join(sample_dir, 'test', 'weights')):
        os.makedirs(os.path.join(sample_dir, 'test', 'weights'))

    if not os.path.exists(os.path.join(sample_dir, 'test', 'settings')):
        os.makedirs(os.path.join(sample_dir, 'test', 'settings'))

    sample_test_set["data"].to_parquet(os.path.join(sample_dir, 'test', 'data', 'data.parquet'), index=False)
    np.savetxt(os.path.join(sample_dir, 'test', 'labels', 'data.labels'), sample_test_labels, fmt='%f')
    np.savetxt(os.path.join(sample_dir, 'test', 'weights', 'data.weights'), sample_test_weights, fmt='%f')

    print("Sample data generated")

public_data_gen()

