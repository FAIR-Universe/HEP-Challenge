import sys

sys.path.append("..")
# from ingestion_parallel import Ingestion
from ingestion_parallel import Ingestion
from datasets import Data
import argparse
import pathlib
import os
import numpy as np
import json

module_dir = os.path.dirname(os.path.realpath(__file__))
root_dir_name = os.path.dirname(module_dir)

parser = argparse.ArgumentParser(
    description="This is script to run ingestion program for the competition"
)
parser.add_argument(
    "--input",
    "-i",
    type=pathlib.Path,
    help="Input file location",
    default=os.path.join(root_dir_name, "input_data"),
)
parser.add_argument(
    "--output",
    "-o",
    help="Output file location",
    default=os.path.join(root_dir_name, "sample_result_submission"),
)
parser.add_argument(
    "--submission",
    "-s",
    help="Submission file location",
    default=os.path.join(root_dir_name, "sample_code_submission"),
)
parser.add_argument(
    "--public-dataset",
    help="True when using public dataset",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--codabench",
    help="True when running on Codabench",
    action="store_true",
)
parser.add_argument(
    "--use-random-mus",
    help="Use random mus for testing",
    action="store_true",
)
parser.add_argument(
    "--systematics",
    type=dict,
    help="Systematics to be used",
    default={  # Systematics to use
        "tes": False,
        "jes": False,
        "soft_met": False,
        "w_scale": False,
        "bkg_scale": False,
    },
)
parser.add_argument(
    "--num-pseudo-experiments",
    type=int,
    help="Number of pseudo experiments",
    default=2,
)
parser.add_argument(
    "--num-of-sets",
    type=int,
    help="Number of sets",
    default=5,
)


args = parser.parse_args()

if not args.codabench:
    input_dir = args.input
    output_dir = args.output
    submission_dir = args.submission
    program_dir = os.path.join(root_dir_name, "ingestion_program")
else:
    input_dir = "/app/input_data"
    output_dir = "/app/output"
    submission_dir = "/app/ingested_program"
    program_dir = "/app/program"


if args.public_dataset:
    from datasets import Neurips2024_public_dataset as public_dataset

    data = public_dataset()
else:
    data = Data(input_dir)


sys.path.append(program_dir)
sys.path.append(submission_dir)

from model import Model


ingestion = Ingestion(data)

# Start timer
ingestion.start_timer()

# initialize submission
ingestion.init_submission(Model)

# fit submission
ingestion.fit_submission()
test_settings = {}
test_settings["systematics"] = args.systematics
test_settings["num_pseudo_experiments"] = args.num_pseudo_experiments
test_settings["num_of_sets"] = args.num_of_sets

if args.use_random_mus:
    test_settings["ground_truth_mus"] = (
        np.random.uniform(0.1, 3, test_settings["num_of_sets"])
    ).tolist()
    test_settings["random_mu"] = True
    random_settings_file = os.path.join(output_dir, "test_settings.json")
    with open(random_settings_file, "w") as f:
        json.dump(test_settings, f)
else:
    test_settings["ground_truth_mus"] = data.ground_truth_mus

# load test data
data.load_test_set()

# predict submission
ingestion.predict_submission(test_settings)

# compute result
ingestion.compute_result()

# save result
ingestion.save_result(output_dir)

# Stop timer
ingestion.stop_timer()

# Show duration
ingestion.show_duration()

# Save duration
ingestion.save_duration(output_dir)
