import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Get input data from args
parser = argparse.ArgumentParser(description="Convert csv to parquet")
parser.add_argument("--input", type=str, help="Input data")
args = parser.parse_args()

csv = Path(args.input)

# Check file ends with .csv
if csv.suffix != ".csv":
    raise ValueError("Input file must be a csv")

df = pd.read_csv(csv, dtype=np.float32)

# Save to parquet
output = csv.with_suffix(".parquet")
df.to_parquet(output)
