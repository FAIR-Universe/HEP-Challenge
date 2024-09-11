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
     - [`model.py`](/simple_one_syst_model/model.py)
     - [`boosted_decision_tree.py`](/simple_one_syst_model/statistical_analysis.py)
   - Objective: Train a classifier to distinguish between [signal events](https://fair-universe.lbl.gov/docs/pages/data.html#higgs-signal) and [background events](https://fair-universe.lbl.gov/docs/pages/data.html#z-boson-background).

### 2. **Statistical Analysis**:
   - **File**: [`statistical_analysis.py`](https://github.com/FAIR-Universe/HEP-Challenge/blob/master/simple_one_syst_model/statistical_analysis.py)
   - Objective: Use the classifier output to extract the signal strength through statistical analysis.

> ⚠️ **Note**: 
> **You can find all related files in the [`simple_one_syst_model`](https://github.com/FAIR-Universe/HEP-Challenge/tree/master/simple_one_syst_model)**.

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

## How to run it
The model fits one systematics at a time hence we need to decide which one it would be beforehand with `MODEL_SYST_NAME`.
```bash
export MODEL_SYST_NAME="jes" # if you want to fit the Jet Energy Scale systematics

python3 HEP-Challenge/run_ingestion.py \ 
--use-random-mus \ 
--systematics-tes \ 
--systematics-soft-met \ 
--systematics-jes \ 
--systematics-ttbar-scale \ 
--systematics-diboson-scale \ 
--systematics-bkg-scale \
--num-pseudo-experiments 100 \ 
--num-of-sets 1 
--submission HEP-Challenge/simple_one_syst_model
```