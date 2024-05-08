import json
import os
import sys
# ------------------------------------------
# Settings
# ------------------------------------------
# True when running on Codabench
# False when running locally
CODABENCH = False
NUM_SETS = 4  # Total = 10
NUM_PSEUDO_EXPERIMENTS = 50  # Total = 100
USE_SYSTEAMTICS = True  # True when using systematics
USE_PUBLIC_DATA = False  # True when using public data
DICT_SYSTEMATICS = {  # Systematics to use
    "tes": True,
    "jes": False,
    "soft_met": False,
    "w_scale": False,
    "bkg_scale": False,
}
NUM_SYSTEMATICS = len(DICT_SYSTEMATICS.values())
USE_RANDOM_MUS = True

LUMINOCITY = 140  # 1/fb
INPUT_DIR = os.path.dirname(os.path.realpath(__file__))
CSV = True
PARQUET = False

cross_section_path = os.path.join(INPUT_DIR, "crosssection.json")

with open(cross_section_path) as json_file:
    DICT_CROSSSECTION = json.load(json_file)

