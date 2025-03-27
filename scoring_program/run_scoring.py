import argparse
import pathlib
import os
import json
import sys

sys.path.append("..")

module_dir = os.path.dirname(os.path.realpath(__file__))
root_dir_name = os.path.dirname(module_dir)

parser = argparse.ArgumentParser(
    description="This is script to generate data for the HEP competition."
)
parser.add_argument(
    "--prediction",
    "-p",
    type=pathlib.Path,
    help="Prediction file location",
    default=os.path.join(root_dir_name, "sample_result_submission"),
)
parser.add_argument(
    "--output",
    "-o",
    help="Output file location",
    default=os.path.join(root_dir_name, "scoring_output"),
)
parser.add_argument(
    "--reference",
    "-r",
    help="Reference file location",
    default=os.path.join(root_dir_name, "reference_data"),
)
parser.add_argument(
    "--codabench",
    help="True when running on Codabench",
    action="store_true",
)
parser.add_argument(
    "--phase-2",
    help="True when running phase 2 of the competition",
    action="store_true",
)

args = parser.parse_args()

if not args.codabench:
    prediction_dir = args.prediction
    output_dir = args.output
    reference_dir = args.reference
    program_dir = os.path.join(root_dir_name, "ingestion_program")
else:
    prediction_dir = "/app/input/res"
    output_dir = "/app/output"
    reference_dir = "/app/input/ref"
    program_dir = os.path.join(root_dir_name, "ingestion_program")

sys.path.append(program_dir)

settings_file = os.path.join(prediction_dir, "test_settings.json")

if os.path.exists(settings_file):
    with open(settings_file) as f:
        test_settings = json.load(f)
else:
    settings_file = os.path.join(reference_dir, "settings", "data.json")

    with open(settings_file) as f:
        test_settings = json.load(f)

from score import Scoring

# Init scoring
scoring = Scoring()

# Start timer
scoring.start_timer()

# Load ingestion duration
ingestion_duration_file = os.path.join(prediction_dir, "ingestion_duration.json")
scoring.load_ingestion_duration(ingestion_duration_file)

# Load ingestions results
scoring.load_ingestion_results(prediction_dir, output_dir)

num_samples = len(test_settings["ground_truth_mus"])

if args.phase_2:
    scoring.compute_scores(test_settings, no_html=True)

    scoring.compute_bootstraped_scores(n_bootstraps=1000, sample_size=num_samples)

    scoring.plot_bootstraped_scores()
else:
    scoring.compute_scores(test_settings)

# Write scores
scoring.write_scores()

# Stop timer
scoring.stop_timer()

# Show duration
print("\n---------------------------------")
print(f"[✔] Total duration: {scoring.get_duration()}")
print("---------------------------------")

print("\n----------------------------------------------")
print("[✔] Scoring Program executed successfully!")
print("----------------------------------------------\n\n")
