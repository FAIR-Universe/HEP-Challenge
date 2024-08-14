# HEP-Challenge Scoring Program
This folder contains the Scoring program for the HEP-Challenge.

## Overview
This program is designed to calculate the scores for the HEP Challenge based on the provided data. It takes input files containing predictions and ground truth values and outputs the calculated scores.

## Usage
To run the scoring program, use the following command:

```
python run_scoring.py 
```

Check out the flags of `run_scoring.py` by using `python run_scoring.py -h `

## Output
The program will output the calculated scores, including 
1. Mean Average Estimator (MAE)
2. Root Mean Square Estimator (RMSE)
3. Quantile Score (QS)

The detailed coverage plots will be available in the [detailed_results.html](/scoring_output/detailed_results.html)

For more information on the workings of the scoring program check out our [whitepaper](https://fair-universe.lbl.gov/whitepaper.pdf)


