import json
import os
import sys
# ------------------------------------------
# Settings
# ------------------------------------------


TEST_SETTINGS = {
"systematics": {  # Systematics to use
    "tes": True,
    "jes": False,
    "soft_met": False,
    "w_scale": False,
    "bkg_scale": False,
},
"num_pseudo_experiments" : 1 ,"num_of_sets" : 10}  # Number of sets to use

USE_RANDOM_MUS = True

LUMINOCITY = 140  # 1/fb


# INPUT_DIR = "/global/cfs/cdirs/m4287/hep/Delphes_PYTHIA8_output/public_data/input_data"
INPUT_DIR = "/home/chakkappai/Work/Fair-Universe/trial_data_set/input_data"


# REFERENCE_DIR = "/global/cfs/cdirs/m4287/hep/Delphes_PYTHIA8_output/public_data/reference_data"
REFERENCE_DIR = "/home/chakkappai/Work/Fair-Universe/trial_data_set/reference_data"

current_path = os.path.dirname(os.path.realpath(__file__))
cross_section_path = os.path.join(current_path, "crosssection.json")
with open(cross_section_path) as json_file:
    DICT_CROSSSECTION = json.load(json_file)


LHC_NUMBERS = {
    "ztautau": 7306660,
    "wjets": 3812700,
    "diboson": 2398564,
    "ttbar": 616017,
    "htautau": 285
}


