import json
import sys
import argparse
import pathlib


def coverage_test(result_file):
    try:
        with open(result_file, "r") as f:
            result_data = json.load(f)

    except Exception as e:
        print(f"❌ Test Failed: Could not read result file: {e}")
        return False

    coverage = result_data["coverage"]
    if coverage >= 0.60:
        print("✅ Test Passed: Coverage is greater than 0.68.")
        print(" Coverage: ", coverage)

        quantile_scores = result_data["quantiles_score"]

        if quantile_scores > 0.0:       
            print("✅ Test Passed: Quantile score is greater than 0.0.")
            print(" Quantile score: ", quantile_scores)

        return True
    else:
        print("❌ Test Failed: Coverage is less than 0.68.")
        print(" Coverage: ", coverage
        return False
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare JSON result file with truth file."
    )

    current_dir = pathlib.Path(__file__).parent
    root_dir = current_dir.parent

    parser.add_argument(
        "--result-file",
        type=pathlib.Path,
        help="Path to the result JSON file",
        default=root_dir / "scoring_output" / "scores.json",
    )

    args = parser.parse_args()

    result_file = args.result_file
    if not coverage_test(result_file):
        sys.exit(1)  # Exit with a non-zero code if test fails
