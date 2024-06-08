import numpy as np
from sys import path
from systematics import systematics
import pickle

path.append("../")
path.append("../ingestion_program")


def compute_mu(score, weight, saved_info):
    """
    Perform calculations to calculate mu.

    Args:
        score (numpy.ndarray): Array of scores.
        weight (numpy.ndarray): Array of weights.
        saved_info (dict): Dictionary containing saved information.

    Returns:
        dict: Dictionary containing calculated values of mu_hat, del_mu_stat, del_mu_sys, and del_mu_tot.
    """

    score = score.flatten() > 0.5
    score = score.astype(int)

    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    del_mu_stat = (
        np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    )
    del_mu_sys = abs(0.1 * mu)
    del_mu_tot = (1 / 2) * np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


def calculate_saved_info(model, train_set, file_path="saved_info.pkl"):
    """
    Calculate the saved_info dictionary for mu calculation.

    Args:
        model: The model used for prediction.
        train_set (dict): Dictionary containing training set data and labels.
        file_path (str, optional): File path to save the calculated saved_info dictionary. Defaults to "saved_info.pkl".

    Returns:
        dict: Dictionary containing calculated values of beta and gamma.
    """

    train_systematic = systematics(
        train_set,
        bkg_scale=1.01,
        jes=1.1,
        soft_met=1.1,
        tes=1.1,
        ttbar_scale=1.1,
        diboson_scale=1.1,
    )

    score = model.predict(train_set["data"])

    print("score shape before threshold", score.shape)

    score = score.flatten() > 0.5
    score = score.astype(int)

    label = train_set["labels"]

    print("score shape after threshold", score.shape)

    gamma = np.sum(train_set["weights"] * score * label)

    beta = np.sum(train_set["weights"] * score * (1 - label))

    saved_info = {"beta": beta, "gamma": gamma}

    print("saved_info", saved_info)

    return saved_info
