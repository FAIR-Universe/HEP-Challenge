import json
import os
import sys
# ------------------------------------------
# Settings
# ------------------------------------------
# True when running on Codabench
# False when running locally
CODABENCH = False
NUM_SETS = 1  # Total = 10
NUM_PSEUDO_EXPERIMENTS = 20  # Total = 100
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
# INPUT_DIR = "/global/cfs/cdirs/m4287/hep/Delphes_PYTHIA8_output/challenge_data/input_data"
INPUT_DIR = "/home/chakkappai/Work/Fair-Universe/trial_data_set/input_data"
# REFERENCE_DIR = "/global/cfs/cdirs/m4287/hep/Delphes_PYTHIA8_output/challenge_data/reference_data"
REFERENCE_DIR = "/home/chakkappai/Work/Fair-Universe/trial_data_set/reference_data"

CSV = False
PARQUET = True
current_path = os.path.dirname(os.path.realpath(__file__))
cross_section_path = os.path.join(current_path, "crosssection.json")

with open(cross_section_path) as json_file:
    DICT_CROSSSECTION = json.load(json_file)
    
LHC_NUMBERS= {
        "ztautau":2569787,
        "wjets": 2964267,
        "diboson": 9220,
        "ttbar": 320318,
        "htautau": 9220,
    }


