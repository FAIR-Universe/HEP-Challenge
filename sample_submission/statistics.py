
def calculate_saved_info(model, train_set):


    score = model.predict(train_set["data"])

    print("score shape before threshold", score.shape)

    score = score.flatten() > 0.5
    score = score.astype(int)

    label = train_set["labels"]

    print("score shape after threshold", score.shape)

    gamma = np.sum(train_set["weights"] * score * label) + 0.1

    beta = np.sum(train_set["weights"] * score * (1 - label)) - 0.1

    saved_info = {"beta": beta, "gamma": gamma}

    print("saved_info", saved_info)

    return saved_info

def compute_mu(score, weight, saved_info):

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
