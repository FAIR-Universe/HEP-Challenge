# Example: Statistical-Only and 1-Systematic Case Analysis

This example demonstrates how to obtain results both with and without systematic uncertainties in a statistical analysis. It is divided into two parts: **Machine Learning** and **Statistical Analysis**. 

## Problem Overview

Six systematic uncertainties are parametrized by nuisance parameters (NPs). The NPs and their corresponding uncertainties are:

- **Tau Energy Scale** (`tes`)
- **Jet Energy Scale** (`jes`)
- **Soft missing energy component** (`soft_met`) [default: 0]
- **Total background scale** (`bkg_scale`)
- **ttbar background** (`ttbar_scale`)
- **Diboson background** (`diboson_scale`)

The default NP values are 1.0, except for `SOFT_MET`, which has a default value of 0. 

For two scenarios:
1. **Statistical-Only**: NPs are fixed to default values.
2. **1-Systematic Case**: For each pseudo-experiment, only one NP floats while the others remain fixed.

---

## Example Breakdown

The process is split into two main sections:

### 1. **Machine Learning**:
   - **Files**: 
     - [`model.py`](https://github.com/FAIR-Universe/HEP-Challenge/blob/master/sample_code_submission/model.py)
     - [`boosted_decision_tree.py`](https://github.com/FAIR-Universe/HEP-Challenge/blob/master/sample_code_submission/statistical_analysis.py)
   - Objective: Train a classifier to distinguish between [signal events](https://fair-universe.lbl.gov/docs/pages/data.html#higgs-signal) and [background events](https://fair-universe.lbl.gov/docs/pages/data.html#z-boson-background).

### 2. **Statistical Analysis**:
   - **File**: [`statistical_analysis.py`](https://github.com/FAIR-Universe/HEP-Challenge/blob/master/sample_code_submission/statistical_analysis.py)
   - Objective: Use the classifier output to extract the signal strength through statistical analysis.

> ⚠️ **Note**: 
> **You can find all related files in the [`sample_code_submission`](https://github.com/FAIR-Universe/HEP-Challenge/tree/master/sample_code_submission)**.

---

## Workflow Overview

Because training a classifier on all systematic variations is impractical, this workflow suggests training on the **nominal dataset** and using that trained classifier to analyze systematic variations. This is feasible for low-dimensional cases (i.e., 1 or 2 systematics). Here’s how it works:

### Steps:
1. **Train the Classifier**: Train on the nominal dataset.
2. **Set Binning**: Define binning for the classifier output.
3. **Gaussian Variation**: For a chosen systematic (e.g., `TES`), generate points following a Gaussian distribution with the NP's mean and uncertainty.
4. **Generate Datasets**: Use `systematics(dataset, tes=new_tes)` to generate datasets for each systematic variation point.
5. **Classifier Application**: Apply the trained classifier to these derived datasets.
6. **Template Creation**: For each classifier output bin, perform a polynomial fit on the systematic variation points to create templates (e.g., for `TES`). Save templates in `saved_info_model_XGB/tes.pkl`.
7. **Repeat for Other Systematics**: Follow steps 3-6 for each systematic.
8. **Perform the Fit**: Load the systematic templates and perform a fit to extract the signal strength.

---

## Machine Learning: Classifier Training

The machine learning task involves using **XGBoost** to distinguish between signal and background events. Below is the setup for the **XGBClassifier**:

```python
from xgboost import XGBClassifier

# Training setup
model = XGBClassifier(
    n_estimators=150, 
    max_depth=5,
    learning_rate=0.15,
    eval_metric=["error", "logloss", "rmse"],
    early_stopping_rounds=10,
)

# Fit model
model.fit(X_train_data, labels, weights, eval_set=[(X_valid_data, valid_set[1])], sample_weight_eval_set=[valid_set[2]], verbose=True)

# Predict probabilities
model.predict_proba(data)[:, 1]
```

### Key Details:
- A `StandardScaler` is used to normalize features before training. The same scaler should be applied to the test data during prediction.
- `early_stopping_rounds` halts training if validation performance doesn’t improve.
- Hyperparameters can be tuned further.
- Saved models are stored in `model_XGB.pkl` and `model_XGB.json`.

> **Tip**: If you want to implement a custom classifier, ensure it follows the general model interface provided in `model.py`.

---

## Statistical Analysis: Signal Strength Extraction

This part deals with extracting the signal strength while accounting for systematic uncertainties.

### 1. **Building Templates**
The templates for each systematic are built using the function `calculate_saved_info(self, holdout_set, file_path)`.

### 2. **Fitting Procedure**
1. **Load Templates**: Load pre-built templates (e.g., `tes.pkl`) or generate them with `calculate_saved_info()` if missing.
2. **Polynomial Coefficients**: Retrieve coefficients for polynomial fits from `alpha_function()`.
3. **Fit the Signal Strength**: Use `compute_mu()` to perform the fit.

   - The signal and background yields are modeled as: $ \mu \cdot S(\alpha) + B(\alpha) $
   - `alpha` represents the current NP values, and the fit minimizes the negative log likelihood (NLL) using the [iminuit](https://scikit-hep.org/iminuit/about.html) package.

> **Note**: Gaussian constraints are applied to NPs based on their default values and uncertainties.

---

## A sample running script for the example

```python
from sys import path
import shutil
import numpy as np
import json
import matplotlib.pyplot as plt
import warnings
import os
import sys
import argparse
import time
warnings.filterwarnings("ignore")


root_dir = "./HEP-Challenge"
input_dir = os.path.join(root_dir, "input_data")
output_dir = os.path.join(root_dir, "sample_result_submission")
submission_dir = os.path.join(root_dir, "sample_code_submission")
program_dir = os.path.join(root_dir, "ingestion_program")
score_dir = os.path.join(root_dir, "scoring_program")

path.append(program_dir)
path.append(submission_dir)
path.append(score_dir)

from systematics import systematics
from model import Model
from datasets import Data
from datasets import Neurips2024_public_dataset as public_dataset


def do_work(TEST_SETTINGS: dict, syst_fix: dict, stats_only: bool = False, tag: str="tag", train_only: bool = False):
    USE_RANDOM_MUS = True
    USE_PUBLIC_DATASET = True

    
    test_settings = TEST_SETTINGS.copy()

    if USE_RANDOM_MUS:
        os.makedirs(os.path.join(root_dir, "sample_result_submission"), exist_ok=True)

        # test_settings[ "ground_truth_mus"] = (np.random.uniform(0.1, 3, test_settings["num_of_sets"])).tolist()
        test_settings["ground_truth_mus"] = (np.linspace(0, test_settings["num_of_sets"] - 1, test_settings["num_of_sets"])).tolist()

        
        random_settings_file = os.path.join(output_dir, "random_mu.json")
        with open(random_settings_file, "w") as f:
            json.dump(test_settings, f)
    else:
        test_settings_file = os.path.join(input_dir, "test", "settings", "data.json")
        with open(test_settings_file) as f:
            test_settings = json.load(f)


    plot_dir = './plots'
    os.makedirs(plot_dir, exist_ok=True)

    if USE_PUBLIC_DATASET:
        data = public_dataset()
    else:
        data = Data(input_dir)

    data.load_train_set()
    data.load_test_set()

    sys.stdout.flush()
    
    # from ingestion import Ingestion
    from ingestion_parallel import Ingestion

    ingestion = Ingestion(data)
    ingestion.init_submission(Model)
    ingestion.model.stat_analysis.stat_only = stats_only
    ingestion.model.stat_analysis.syst_fixed_setting = syst_fix
    ingestion.model.stat_analysis.run_syst = tag
    ingestion.fit_submission()
    sys.stdout.flush()

    if train_only: 
        exit(0)

    ingestion.predict_submission(test_settings)
    ingestion.compute_result()
    ingestion.save_result(output_dir)
    sys.stdout.flush()

    from score import Scoring
    score = Scoring()
    score.load_ingestion_results(output_dir)
    score.compute_scores(test_settings)
    sys.stdout.flush()


    def visualize_scatter(ingestion_result_dict, ground_truth_mus, save_name):
        plt.figure(figsize=(6, 4))
        for key in ingestion_result_dict.keys():
            ingestion_result = ingestion_result_dict[key]
            mu_hat = np.mean(ingestion_result["mu_hats"])
            mu = ground_truth_mus[key]
            plt.scatter(mu, mu_hat, c='b', marker='o')
        
        plt.xlabel('Ground Truth $\mu$')
        plt.ylabel('Predicted $\mu$ (averaged for 100 test sets)')
        plt.title('Ground Truth vs. Predicted $\mu$ Values')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        plt.savefig(save_name)


    def visualize_coverage(ingestion_result_dict, ground_truth_mus, save_prefix):

        for key in ingestion_result_dict.keys():
            plt.figure( figsize=(5, 5))

            ingestion_result = ingestion_result_dict[key]
            mu = ground_truth_mus[key]
            mu_hats = np.mean(ingestion_result["mu_hats"])
            p16s = ingestion_result["p16"]
            p84s = ingestion_result["p84"]
            
            # plot horizontal lines from p16 to p84
            for i, (p16, p84) in enumerate(zip(p16s, p84s)):
                plt.hlines(y=i, xmin=p16, xmax=p84, colors='b', label='p16-p84')

            plt.vlines(x=mu_hats, ymin=0, ymax=len(p16s), colors='r', linestyles='dashed', label='Predicted $\mu$')
            plt.vlines(x=mu, ymin=0, ymax=len(p16s), colors='g', linestyles='dashed', label='Ground Truth $\mu$')
            plt.xlabel('mu')
            plt.ylabel('psuedo-experiments')
            plt.title(f'mu distribution - Set_{key}')
            plt.legend()
            
            plt.show()

            plt.savefig(f"{save_prefix}_Set_{key}.png")


    # Visualize scatter plot of ground truth mu and predicted mu
    visualize_scatter(ingestion_result_dict=ingestion.results_dict, ground_truth_mus=test_settings["ground_truth_mus"], save_name=f"{plot_dir}/scatter.png")
    
    # Visualize coverage
    visualize_coverage(ingestion_result_dict=ingestion.results_dict, ground_truth_mus=test_settings["ground_truth_mus"], save_prefix=f"{plot_dir}/coverage.png")

    print("Done - 1")

    # post process
    save_tag = f"{tag}" if not stats_only else f"stats.{tag}"
    os.rename(plot_dir, f"./sys_results/{plot_dir}.{save_tag}")
    shutil.copytree("./HEP-Challenge/sample_result_submission", f"./sys_results/results.{save_tag}")

    print("Done - 2")


if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Add arguments
    parser.add_argument('--syst_name', type=str, help='Name of the system')
    parser.add_argument('--stat_only_run', action='store_true', 
                    help='Run in stat-only mode (default: False)')
    parser.add_argument('--train_only', action='store_true', 
                    help='Only train the model (default: False)')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    syst_name = args.syst_name
    stat_only_run = args.stat_only_run
    train_only = args.train_only

    # Example output to verify
    print(f'syst_name: {syst_name}')
    print(f'stat_only_run: {stat_only_run}')

    os.makedirs("sys_results", exist_ok=True)

    TEST_SETTINGS = {
        "systematics": {  # Systematics to use
            "tes": False, # tau energy scale
            "jes": False, # jet energy scale
            "soft_met": False, # soft term in MET
            "ttbar_scale": False, # ttbar scale factor
            "diboson_scale": False, # diboson scale factor
            "bkg_scale": False, # Background scale factor
        },
    "num_pseudo_experiments" : 500 , # Number of pseudo-experiments to run per set
    "num_of_sets" : 5, # Number of sets of pseudo-experiments to run
    }
    syst_fixed_setting = {
            'tes': 1.0,
            'bkg_scale': 1.0,
            'jes': 1.0,
            'soft_met': 0.0,
            'ttbar_scale': 1.0,
            'diboson_scale': 1.0,
    }
 
    print(f"--> Launching {syst_name} systematics")
    # add time start
    start = time.time()

    TEST_SETTINGS["systematics"][syst_name] = True
    SYST_FIXED = {k: v for k, v in syst_fixed_setting.items() if k != syst_name}
    if stat_only_run:
        do_work(TEST_SETTINGS=TEST_SETTINGS, syst_fix=SYST_FIXED, stats_only=True, tag=syst_name, train_only=train_only)
    else:
        do_work(TEST_SETTINGS=TEST_SETTINGS, syst_fix=SYST_FIXED, stats_only=False, tag=syst_name, train_only=train_only)
    TEST_SETTINGS["systematics"][syst_name] = False

    # add time end
    end = time.time()
    print(f"--> {syst_name} systematics done in {end - start} seconds")
```

The python script can be run with the following command to get the statistical only results and the results with 1-systematics case:
```bash
working_dir="directory to the script"

python3 ${working_dir}/run.py --train_only

python3 ${working_dir}/run.py --syst_name tes --stat_only_run
python3 ${working_dir}/run.py --syst_name jes --stat_only_run
python3 ${working_dir}/run.py --syst_name soft_met --stat_only_run
python3 ${working_dir}/run.py --syst_name bkg_scale --stat_only_run
python3 ${working_dir}/run.py --syst_name ttbar_scale --stat_only_run
python3 ${working_dir}/run.py --syst_name diboson_scale --stat_only_run

python3 ${working_dir}/run.py --syst_name tes 
python3 ${working_dir}/run.py --syst_name jes 
python3 ${working_dir}/run.py --syst_name soft_met 
python3 ${working_dir}/run.py --syst_name bkg_scale 
python3 ${working_dir}/run.py --syst_name ttbar_scale 
python3 ${working_dir}/run.py --syst_name diboson_scale
```