import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import pickle


class BoostedDecisionTree:
    """
    This class implements a boosted decision tree classifier.

    Attributes:
        * model (XGBClassifier): The underlying XGBoost classifier model.
        * scaler (StandardScaler): The scaler used to normalize the input data.

    Methods:
        * fit(self, train_data, labels, weights=None): Fits the model to the training data.
        * predict(self, test_data): Predicts the class probabilities for the test data.
        * save(self, model_name): Saves the model and scaler to disk.
        * load(self, model_path): Loads the model and scaler from disk.

    """

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.15,
            eval_metric=["error", "logloss", "rmse"],
            early_stopping_rounds=10,
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, labels, weights=None, valid_set=None):
        """
        Fits the model to the training data.

        Args:
            * train_data (pandas.DataFrame): The input training data.
            * labels (array-like): The labels corresponding to the training data.
            * weights (array-like, optional): The sample weights for the training data.

        """

        self.scaler.fit_transform(train_data)

        X_train_data = self.scaler.transform(train_data)
        X_valid_data = self.scaler.transform(valid_set[0])
        self.model.fit(
            X_train_data, labels, weights,
            eval_set=[(X_valid_data, valid_set[1])],
            sample_weight_eval_set=[valid_set[2]],
            verbose=True,
        )

        # printout the accuracy and AUC of the test set using sklearn
        print(f"Accuracy: {accuracy_score(valid_set[1], self.model.predict(X_valid_data)):.3%}")
        print(f"AUC: {roc_auc_score(valid_set[1], self.model.predict_proba(X_valid_data)[:, 1]):.3f}")

    def predict(self, data):
        """
        Predicts the class probabilities for the input data.

        Args:
            data (pandas.DataFrame): The input data.

        Returns:
            array-like: The predicted class probabilities.

        """

        data = self.scaler.transform(data)
        return self.model.predict_proba(data)[:, 1]

    def save(self, model_name):
        """
        Saves the model and scaler to disk.

        Args:
            model_name (str): The name of the model file.

        """
        model_path = model_name + ".json"
        self.model.save_model(model_path)

        scaler_path = model_name + ".pkl"
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
        self.scaler = pickle.load(open(model_path.replace(".json", ".pkl"), "rb"))
        
        return self.model
