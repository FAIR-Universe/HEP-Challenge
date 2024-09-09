# Test model
This folder contains the test model. unlike sample code submissions, The test model designed to test if the ingestion and scoring in running properly. 

## Description
The main part of the sample code submission is in the `model.py` which contains the `Model` class. 
* The model class can use 3  different binary classifier. 
    1. BDT classifier using [xgboost](https://xgboost.ai/) defined in [boosted_decision_tree](/test_model/boosted_decision_tree.py).
    2. NN classifier using [tensorflow](https://www.tensorflow.org/) defined in [neural_network_TF](/test_model/neural_network_TF.py)
    3. NN classifier using [pytorch](https://pytorch.org/) defined in [neural_network_torch](/test_model/neural_network_torch.py)

* The statistical analysis part is written in [statistical_analysis](/test_model/statistical_analysis.py) 

The statistical method is a simple counting method. i.e. $\mu = \frac{N -\beta}{ \gamma} $ where $\beta$ and $\gamma$ are prior probability of background and signal respectively and $N$ is the number of events 