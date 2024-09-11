# Sample code submission
This folder contains the sample code submission
## Description
The main part of the sample code submission is in the `model.py` which contains the `Model` class. 
* The model class uses the BDT classifier using [xgboost](https://xgboost.ai/) defined in [boosted_decision_tree](/simple_one_syst_model/boosted_decision_tree.py). 
* The statistical analysis part is written in [statistical_analysis](/simple_one_syst_model/statistical_analysis.py) 

Workings of the Statistical Analysis part can be found in our [whitepaper](https://fair-universe.lbl.gov/whitepaper.pdf)

## How to run it
The model fits one systematics at a time hence we need to decide which one it would be beforehand with `MODEL_SYST_NAME`.
```bash
export MODEL_SYST_NAME="jes" # if you want to fit the Jet Energy Scale systematics

python3 HEP-Challenge/run_ingestion.py \ 
--use-random-mus \ 
--systematics-tes \ 
--systematics-soft-met \ 
--systematics-jes \ 
--systematics-ttbar-scale \ 
--systematics-diboson-scale \ 
--systematics-bkg-scale \
--num-pseudo-experiments 100 \ 
--num-of-sets 1 
--submission HEP-Challenge/simple_one_syst_model
```

Additionally, a full chain test is available at [Launch_tests.sh](/test/Launch_tests.sh) which is designed to test the model for all 6 systematics in NERSC
