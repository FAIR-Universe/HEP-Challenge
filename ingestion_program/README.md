# HEP-Challenge Ingestion Program

This folder contains the ingestion program for the HEP-Challenge.

## Overview
The HEP-Challenge Ingestion Program is responsible for processing and ingesting data for the HEP-Challenge. It provides a set of functionalities to handle data ingestion, transformation, and running the participants' model.

## Running Ingestion Program locally.
To run the ingestion in a CPU-only system use 
```bash
python3 run_ingestion.py \ 
--use-random-mus \ 
--systematics-tes \ 
--systematics-soft-met \ 
--systematics-jes \ 
--systematics-ttbar-scale \ 
--systematics-diboson-scale \ 
--systematics-bkg-scale \
--num-pseudo-experiments 100 \ 
--num-of-sets 1 
```

If you have GPU you can use the `--parallel` flag to parallelize the pseudo experiments.

For more information on flags run `python3 run_ingestion.py -h`
For more information on the workings of the ingestion program check out our [whitepaper](https://fair-universe.lbl.gov/whitepaper.pdf)



