# ------------------------------------------
# Settings
# ------------------------------------------
# True when running on Codabench
# False when running locally
CODABENCH = False
NUM_SETS = 4  # Total = 10
NUM_PSEUDO_EXPERIMENTS = 50  # Total = 100
USE_SYSTEAMTICS = True # True when using systematics
USE_PUBLIC_DATA = False # True when using public data
DICT_SYSTEMATICS = { # Systematics to use
    "tes": True,
    "jes": False,
    "soft_met": False,
    "w_scale": False,
    "bkg_scale": False,
}
NUM_SYSTEMATICS = len(DICT_SYSTEMATICS.values())
USE_RANDOM_MUS = True