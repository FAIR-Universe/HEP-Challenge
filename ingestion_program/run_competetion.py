import sys
sys.path.append('..')
from ingestion import Ingestion

from datasets import Data
from config import *
import argparse
import pathlib
import os
import numpy as np

module_dir = os.path.dirname(os.path.realpath(__file__))
root_dir_name = os.path.dirname(module_dir)

parser = argparse.ArgumentParser(
    description="This is script to generate data for the HEP competition."
)
parser.add_argument("--input", 
                    "-i", 
                    type=pathlib.Path,
                    help="Input file location",
                    default=pathlib.Path(INPUT_DIR),
                    ) 
parser.add_argument("--output", 
                    "-o", 
                    help="Output file location",
                    default=os.path.join(root_dir_name, "sample_result_submission")
                    )
parser.add_argument("--submission",
                    "-s",
                    help="Submission file location",
                    default=os.path.join(root_dir_name, "sample_code_submission")
                    )
parser.add_argument("--codabench",
                    help="True when running on Codabench",
                    action="store_true",
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
    
test_settings = TEST_SETTINGS.copy()

if USE_RANDOM_MUS:
    test_settings[ "ground_truth_mus"] = (np.random.uniform(0.1, 3, test_settings["num_of_sets"])).tolist()
    
    random_settings_file = os.path.join(output_dir, "random_mu.json")
    with open(random_settings_file, "w") as f:
        json.dump(test_settings, f)
else:
    test_settings_file = os.path.join(input_dir, "test", "settings", "data.json")
    with open(test_settings_file) as f:
        test_settings = json.load(f)
        
print(test_settings)
print(TEST_SETTINGS)
        
sys.path.append(program_dir)
sys.path.append(submission_dir)

from model import Model

USE_PUBLIC_DATASET = True

if USE_PUBLIC_DATASET:
    from datasets import Neurips2024_public_dataset as public_dataset
    data = public_dataset()
else:
    data = Data(input_dir)

ingestion = Ingestion(data)

# Start timer
ingestion.start_timer()

# load train set
ingestion.load_train_set()

# initialize submission
ingestion.init_submission(Model)

# fit submission
ingestion.fit_submission()

# load test set
ingestion.load_test_set()

# predict submission
ingestion.predict_submission(test_settings)

# save result
ingestion.save_result()

# Stop timer
ingestion.stop_timer()

# Show duration
ingestion.show_duration()

# Save duration
ingestion.save_duration()

