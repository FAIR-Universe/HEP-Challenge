## Local Setup

### With Conda :
If you are running this starting kit locally, you may want to use a dedicated conda env.  
[Instructions to setup a conda env](https://github.com/FAIR-Universe/HEP-Challenge/tree/master/conda)

### With Docker
The main advantage of using docker is that you dont have to worry about dependencies since a good docker container generally has all the dependencies.

* Step 1. Check if have docker **if not** install docker 
Instalation process can be found in [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

* Step 2. Pull the right version of docker image
```shell
docker pull docker.io/nersc/fair_universe:1298f0a8
```

* Step 3. Start a shell within the container to run the challenge. Assuming that <source> is the path to your source checkout on your host machine, the following command will make the source directory available as /HEP-Challenge in the container and start an interactive bash shell
```shell
docker run --volume=<source>:/HEP-Challenge --interactive \ 
--tty docker.io/nersc/fair_universe:1298f0a8 /bin/bash
```
* Step 4. Start the notebook, The next command will give you a link which can copy to your browser and run the notebook
```shell
jupyter notebook StartingKit_HiggsML_Uncertainty_Challenge.ipynb --no-browser
``` 

#### ⚠️ Note:
If you are running this starting kit on MAC OS, you may want to check and install `libomp` package. 
This package is needed to run xgboost model. Follow the steps below for complete installations.

If still you are facing problems with XGBoost, you can uninstall the current xgboost and install py-xgboost in your environment

Uninstall XGBoost
```
pip uninstall xgboost
```

Install py-xgboost using conda
```
conda install py-xgboost
```
