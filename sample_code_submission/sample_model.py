import numpy as np


class SampleModel:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    def __init__(self):
        pass

    def fit(self, train_data, labels, weights=None):
        pass

    def predict(self, test_data):

        return np.array(test_data["DER_mass_vis"])
