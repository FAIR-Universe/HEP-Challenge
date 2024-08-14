import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle

current_dir = os.path.dirname(__file__)


class BoostedDecisionTree:
    """
    This class implements a boosted decision tree classifier.

    Attributes:
        model (XGBClassifier): The underlying XGBoost classifier model.
        scaler (StandardScaler): The scaler used to normalize the input data.

    Methods:
        __init__(self, train_data): Initializes the BoostedDecisionTree object.
        fit(self, train_data, labels, weights=None): Fits the model to the training data.
        predict(self, test_data): Predicts the class probabilities for the test data.
        save(self, model_name): Saves the model and scaler to disk.
        load(self, model_path): Loads the model and scaler from disk.

    """

    def __init__(self):
        self.model = XGBClassifier()
        self.name = "model_XGB"
        self.scaler = StandardScaler()

    def fit(self, train_data, labels, weights=None):
        """
        Fits the model to the training data.

        Args:
            train_data (pandas.DataFrame): The input training data.
            labels (array-like): The labels corresponding to the training data.
            weights (array-like, optional): The sample weights for the training data.

        """
        self.scaler.fit_transform(train_data)

        X_train_data = self.scaler.transform(train_data)
        self.model.fit(X_train_data, labels, weights, eval_metric="logloss")

    def predict(self, test_data):
        """
        Predicts the class probabilities for the test data.

        Args:
            test_data (pandas.DataFrame): The test data.

        Returns:
            array-like: The predicted class probabilities.

        """
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]

    def save(self):
        """
        Saves the model and scaler to disk.

        Args:
            model_name (str): The name of the model file.

        """
        model_path = current_dir + "/model_XGB.json"
        self.model.save_model(model_path)

        scaler_path = current_dir + "/scaler_XGB.pkl"
        pickle.dump(self.scaler, open(scaler_path, "wb"))

    def load(self, model_path):
        """
        Loads the model and scaler from disk.

        Args:
            model_path (str): The path to the model file.

        Returns:
            XGBClassifier: The loaded model.

        """
        self.model.load_model(model_path)
        scaler_path = current_dir + "/scaler_XGB.pkl"
        self.scaler = pickle.load(open(scaler_path, "rb"))
        
        return self.model
