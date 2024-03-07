import os
from sys import path
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import json

# import mplhep as hep

# hep.set_style("ATLAS")
# ------------------------------
# Absolute path to submission dir
# ------------------------------
submissions_dir = os.path.dirname(os.path.abspath(__file__))
path.append(submissions_dir)

# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps

hist_analysis_dir = os.path.dirname(submissions_dir)
path.append(hist_analysis_dir)

from hist_analysis import compute_result,plot_score
import torch
import torch.nn as nn

class PyTorchModel(nn.Module):
    def __init__(self, n_cols):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(n_cols, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        return x
# ------------------------------
# Baseline Model
# ------------------------------
class Model():
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init : initialize a classifier
    2) fit : can be used to train a classifier
    3) predict: predict mu_hats,  delta_mu_hat and q1,q2

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods
    """

    def __init__(
            self,
            train_set=None,
            systematics=None
    ):
        """
        Model class constructor

        Params:
            train_set:
                labelled train set
                
            systematics:
                systematics class

        Returns:
            None
        """

        # Set class variables from parameters

        current_dir = os.path.dirname(os.path.abspath(__file__))
        module_file = os.path.join(current_dir, "model.pt")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.SYST = True
        self.plot_count = 0
        self.model_name = "NN_NLL"




        if os.path.exists(module_file):
            self.model_exists = True
            self._read_settings()
            self.model = PyTorchModel(n_cols=train_set["data"].shape[1])
            self.model.load_state_dict(torch.load(module_file))
            self.model.to(self.device)
            del train_set
            del systematics

        else:
            self.model_exists = False
            self.train_set = train_set
            self.systematics = systematics
            self.threshold = 0.8
            self.bins = 30
            self.bin_nums = 30
            self.batch_size = 1000
            self.max_num_epochs = 100
            self.calibration = 0
            self.scaler = StandardScaler()


    def _read_settings(self):

        settings_file = os.path.join(submissions_dir, "settings.pkl")
        scaler_file = os.path.join(submissions_dir, "scaler.pkl")

        settings = pickle.load(open(settings_file, "rb"))

        self.threshold = settings["threshold"]
        self.bin_nums = settings["bin_nums"]
        self.control_bins = settings["control_bins"]
        self.coef_s_list = settings["coef_s_list"]
        self.coef_b_list = settings["coef_b_list"]
        self.calibration = settings["calibration"]
        self.bins = np.linspace(0, 1, self.bin_nums + 1)

        fit_line_s_list = []
        fit_line_b_list = []

        for coef_s_,coef_b_ in zip(self.coef_s_list,self.coef_b_list):

            coef_s = np.array(coef_s_)
            coef_b = np.array(coef_b_)

            fit_line_s_list.append(np.poly1d(coef_s))
            fit_line_b_list.append(np.poly1d(coef_b))

        self.fit_dict = {
            "gamma_roi":fit_line_s_list,
            "beta_roi":fit_line_b_list
        }

        print(f"[*] --- length of fit_line_s_list: {len(fit_line_s_list)}")
        print(f"[*] --- length of fit_line_b_list: {len(fit_line_b_list)}")

        self.fit_dict_control = {
            "gamma_roi":fit_line_s_list[-self.control_bins:],
            "beta_roi":fit_line_b_list[-self.control_bins:]
        }

        self.scaler = pickle.load(open(scaler_file, 'rb'))


    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """

        if self.model_exists:
            return

        self._generate_holdout_sets()
        self._init_model()
        self._train()
        self.mu_hat_calc()
        self.save_model()

        self.plot_dir = os.path.join(submissions_dir, "plots/")
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def predict(self, test_set):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        print("[*] - Testing")
        test_df = test_set['data']
        test_df = self.scaler.transform(test_df)
        test_score = self._return_score(test_df)
        test_set['score'] = test_score

        print("[*] - Computing Test result")
        weights_test = test_set["weights"].copy()


        test_hist , bins = np.histogram(test_score,
                    bins=self.bins, density=False, weights=weights_test)
        
        test_hist_control = test_hist[-self.control_bins:]

        mu_hat, mu_p16, mu_p84, alpha = compute_result(test_hist_control,self.fit_dict_control,SYST=self.SYST)

        delta_mu_hat = mu_p84 - mu_p16
        
        mu_p16 = mu_p16-self.calibration
        mu_p84 = mu_p84-self.calibration
        mu_hat = mu_hat-self.calibration

        if self.plot_count > 0:
            hist_fit_s = []
            hist_fit_b = []

            if self.SYST:
                for i in range(self.bin_nums):
                    hist_fit_s.append(self.fit_dict["gamma_roi"][i](alpha))
                    hist_fit_b.append(self.fit_dict["beta_roi"][i](alpha))

            else:       
                hist_fit_s = self.fit_dict["gamma_roi"]
                hist_fit_b = self.fit_dict["beta_roi"]

            hist_fit_s = np.array(hist_fit_s)
            hist_fit_b = np.array(hist_fit_b)

            plot_score(test_hist,hist_fit_s,hist_fit_b,mu_hat,bins,threshold=self.threshold,save_path=(self.plot_dir + f"NN_score_{self.plot_count}.png"))

            self.plot_count = self.plot_count - 1 
            
        print(f"[*] --- mu_hat: {mu_hat}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")
        print(f"[*] --- alpha: {alpha}")

        return {
            "mu_hat": mu_hat,
            "delta_mu_hat": delta_mu_hat,
            "p16": mu_p16,
            "p84": mu_p84
        }

    def _init_model(self):

        print("[*] - Intialize Baseline Model (NN bases Uncertainty Estimator Model)")

        n_cols = self.train_set["data"].shape[1]

        self.model = PyTorchModel(n_cols)

        self.criterion = nn.BCELoss()


        
    def _generate_holdout_sets(self):
        print("[*] - Generating Validation sets")

        # Calculate the sum of weights for signal and background in the original dataset
        signal_weights = self.train_set["weights"][self.train_set["labels"] == 1].sum()
        background_weights = self.train_set["weights"][self.train_set["labels"] == 0].sum()

        # Split the data into training and holdout sets while preserving the proportion of samples with respect to the target variable
        train_df, holdout_df, train_labels, holdout_labels, train_weights, holdout_weights =  train_test_split(
            self.train_set["data"],
            self.train_set["labels"],
            self.train_set["weights"],
            test_size=0.1,
            stratify=self.train_set["labels"]
        )



        # Calculate the sum of weights for signal and background in the training and holdout sets
        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()

        holdout_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_background_weights = holdout_weights[holdout_labels == 0].sum()

        # Balance the sum of weights for signal and background in the training and holdout sets
        train_weights[train_labels == 1] *= signal_weights / train_signal_weights
        train_weights[train_labels == 0] *= background_weights / train_background_weights

        holdout_weights[holdout_labels == 1] *= signal_weights / holdout_signal_weights
        holdout_weights[holdout_labels == 0] *= background_weights / holdout_background_weights

        train_df = train_df.copy()
        train_df["weights"] = train_weights
        train_df["labels"] = train_labels

        train_df = self.systematics(
            data=train_df.copy(),
            tes=1.0
        ).data

        train_weights = train_df.pop('weights')
        train_labels = train_df.pop('labels')
        

        self.train_df = train_df

        self.train_set = {
            "data": train_df,
            "labels": train_labels,
            "weights": train_weights,
            "settings": self.train_set["settings"]
        }

        self.holdout = {
                "data": holdout_df,
                "labels": holdout_labels,
                "weights": holdout_weights
            }

        
        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()

        holdout_set_signal_weights = holdout_weights[holdout_labels == 1].sum()
        holdout_set_background_weights = holdout_weights[holdout_labels == 0].sum()

        print(f"[*] --- original signal: {signal_weights} --- original background: {background_weights}")
        print(f"[*] --- train signal: {train_signal_weights} --- train background: {train_background_weights}")
        print(f"[*] --- holdout_set signal: {holdout_set_signal_weights} --- holdout_set background: {holdout_set_background_weights}")
  

    def _train(self):


        weights_train = self.train_set["weights"].copy()
        train_labels = self.train_set["labels"].copy()
        train_data = self.train_set["data"].copy()
        class_weights_train = (weights_train[train_labels == 0].sum(), weights_train[train_labels == 1].sum())

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= max(class_weights_train) / class_weights_train[i]
            # test dataset : increase test weight to compensate for sampling

        print("[*] --- Training Model")
        train_data = self.scaler.fit_transform(train_data)
        train_labels = np.array(train_labels).ravel()
        weights_train = np.array(weights_train).ravel()

        print("[*] --- shape of train tes data", train_data.shape)

        self._fit(train_data, train_labels, weights_train)

        del self.train_set

        # print("[*] --- Predicting Train set")
        # self.train_set['predictions'] = (self.train_set['data'], self.threshold)

        # self.train_set['score'] = self._return_score(self.train_set['data'])

        # auc_train = roc_auc_score(
        #     y_true=self.train_set['labels'],
        #     y_score=self.train_set['score'],
        #     sample_weight=self.train_set['weights']
        # )
        # print(f"[*] --- AUC train : {auc_train}")

    def _batch_fit(self, X_batch, y_batch, w_batch):
        
        self.optimizer.zero_grad()
        outputs = self.model(X_batch).ravel()
        loss = self.criterion(outputs, y_batch)
        loss.backward()
        self.optimizer.step()

        return loss


    def _fit(self, X, y, w):
        print("[*] --- Fitting Model")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

        X, X_val, y, y_val, w, w_val = train_test_split(X, y, w, test_size=0.01, stratify=y)


        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)
        w_train = torch.tensor(w, dtype=torch.float32)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        w_val = torch.tensor(w_val, dtype=torch.float32)

        iterations = int(len(X_train) / self.batch_size)
        loss = 0
        val_loss = 0
        for epoch in range(self.max_num_epochs):
            pre_loss = loss
            pre_val_loss = val_loss
            loss = 0
            for i in tqdm(range(iterations)):
                X_batch = X_train[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = y_train[i*self.batch_size:(i+1)*self.batch_size]
                w_batch = w_train[i*self.batch_size:(i+1)*self.batch_size]

                loss +=self._batch_fit(X_batch, y_batch, w_batch)


            loss = loss / iterations
            diff_loss = abs(loss - pre_loss)

            val_outputs = self.model(X_val).ravel()
            val_loss = self.criterion(val_outputs, y_val)
            diff_val_loss = abs(val_loss - pre_val_loss)

            print(f"[*] --- epoch: {epoch} --- loss: {loss.item()} --- diff_loss: {diff_loss} --- val_loss: {val_loss.item()} --- diff_val_loss: {diff_val_loss}")

            if diff_loss < 0.0001:
                break

            if diff_val_loss < 0.0005:
                break

        print(f"[*] --- Training done with loss: {loss.item()} in {epoch} epochs")

    def _return_score(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_predict = self.model(X)
        y_predict = y_predict.detach().numpy().ravel()
        return y_predict


    def mu_hat_calc(self):

        X_holdout = self.holdout['data'].copy()
        X_holdout['weights'] = self.holdout['weights'].copy()
        X_holdout['labels'] = self.holdout['labels'].copy()

        holdout_post = self.systematics(
            data=X_holdout.copy(),
            tes=1.0
        ).data

        label_holdout = holdout_post.pop('labels')
        weights_holdout  = holdout_post.pop('weights')

        X_holdout_sc = self.scaler.transform(holdout_post)
        holdout_array = self._return_score(X_holdout_sc)
        print("[*] --- Predicting Holdout set done")
        print("[*] --- score = ", holdout_array)

        # compute gamma_roi

        self.control_bins = int(self.bin_nums * (1 - self.threshold))

        if self.SYST:
            self.theta_function(plot_count=2)

        else:
            s , b = self.nominal_histograms(1)
            self.fit_dict = {
                "gamma_roi": s,
                "beta_roi": b,
                "error_s": [0 for _ in range(self.bins)],
                "error_b": [0 for _ in range(self.bins)]
            }

            self.fit_dict_control = {
                "gamma_roi": s[-self.control_bins:],
                "beta_roi": b[-self.control_bins:],
                "error_s": [0 for _ in range(self.control_bins)],
                "error_b": [0 for _ in range(self.control_bins)]
            }

            

        holdout_hist , _ = np.histogram(holdout_array,
                    bins = self.bins, density=False, weights=weights_holdout)
        
        
        holdout_hist_control = holdout_hist[-self.control_bins:]
        # holdout_hist_control = (s + b)[-self.control_bins:]

        mu_hat, mu_p16, mu_p84, alpha = compute_result(holdout_hist_control,self.fit_dict_control,SYST=self.SYST)

        self.calibration = mu_hat - 1
        
        print(f"[*] --- mu_hat: {mu_hat} --- mu_p16: {mu_p16} --- mu_p84: {mu_p84} --- alpha: {alpha}")

        del self.holdout


    def nominal_histograms(self,theta):

        X_holdout = self.holdout['data'].copy()
        X_holdout['weights'] = self.holdout['weights'].copy()
        X_holdout['labels'] = self.holdout['labels'].copy()

        holdout_syst = self.systematics(
            data=X_holdout.copy(),
            tes=theta
        ).data


        label_holdout = holdout_syst.pop('labels')
        weights_holdout = holdout_syst.pop('weights')

        X_holdout_sc = self.scaler.transform(holdout_syst)
        holdout_val = self._return_score(X_holdout_sc)

        weights_holdout_signal = weights_holdout[label_holdout == 1]
        weights_holdout_background = weights_holdout[label_holdout == 0]

        holdout_signal_hist , _ = np.histogram(holdout_val[label_holdout == 1],
                    bins= self.bins, density=False, weights=weights_holdout_signal)
        
        holdout_background_hist , _ = np.histogram(holdout_val[label_holdout == 0],
                    bins= self.bins, density=False, weights=weights_holdout_background)


        return holdout_signal_hist , holdout_background_hist


    def theta_function(self, plot_count=25):

        fit_line_s_list = []
        fit_line_b_list = []
        self.coef_b_list = []
        self.coef_s_list = []

        error_s = []
        error_b = []

        theta_list = np.linspace(0.9,1.1,10)
        s_list = [[] for _ in range(self.bins)]
        b_list = [[] for _ in range(self.bins)]
        
        for theta in tqdm(theta_list):
            s , b = self.nominal_histograms(theta)
            # print(f"[*] --- s: {s}")
            # print(f"[*] --- b: {b}")

            for i in range(len(s)):
                s_list[i].append(s[i])
                b_list[i].append(b[i])

        print(f"[*] --- s_list shape: {np.array(s_list).shape}")
        print(f"[*] --- b_list shape: {np.array(b_list).shape}")
        print(f"[*] --- theta_list shape: {np.array(theta_list).shape}")

        for i in range(len(s_list)):
            s_array = np.array(s_list[i])
            b_array = np.array(b_list[i])


            coef_s = np.polyfit(theta_list, s_array, 3)
            coef_b = np.polyfit(theta_list, b_array, 3)

            fit_fun_s = np.poly1d(coef_s)
            fit_fun_b = np.poly1d(coef_b)

            error_s.append(np.sqrt(np.mean((s_array - fit_fun_s(theta_list))**2)))

            error_b.append(np.sqrt(np.mean((b_array - fit_fun_b(theta_list))**2)))

            fit_line_s_list.append(fit_fun_s)
            fit_line_b_list.append(fit_fun_b)

            coef_b_ = coef_b.tolist()
            coef_s_ = coef_s.tolist()

            self.coef_s_list.append(coef_s_)
            self.coef_b_list.append(coef_b_)


        for i in range(min(plot_count,len(s_list))):

            _ = plt.figure(figsize=(8, 8))
            plt.plot(theta_list,s_list[i],'b.',label="s")
            plt.plot(theta_list,fit_line_s_list[i](theta_list),'green',label="fit s")
            plt.legend()
            plt.title(f"Bin {i}")
            plt.xlabel("theta")
            plt.ylabel("Events")
            save_path = os.path.join(submissions_dir, "plots/")
            plot_file = os.path.join(save_path, f"NN_s_{i}.png")
            # plt.savefig(plot_file)
            plt.show()

            _ = plt.figure(figsize=(8, 8))
            plt.plot(theta_list,b_list[i],'r.',label="b")
            plt.plot(theta_list,fit_line_b_list[i](theta_list),'brown',label="fit b")
            plt.legend()
            plt.title(f"Bin {i}")
            plt.xlabel("theta")
            plt.ylabel("Events")
            save_path = os.path.join(submissions_dir, "plots/")
            plot_file = os.path.join(save_path, f"NN_b_{i}.png")
            # plt.savefig(plot_file)
            plt.show()



            plot_count = plot_count - 1

            if plot_count <= 0:
                break
        


        self.fit_dict = {
            "gamma_roi": fit_line_s_list,
            "beta_roi": fit_line_b_list,
            "error_s": error_s,
            "error_b": error_b
        }

        print(f"[*] --- number of bins: {self.bins}")
        print(f"[*] --- number of control bins: {self.control_bins}")

        self.fit_dict_control = {
            "gamma_roi": fit_line_s_list[-self.control_bins:],
            "beta_roi": fit_line_b_list[-self.control_bins:],
            "error_s": error_s[-self.control_bins:],
            "error_b": error_b[-self.control_bins:]
        }



    def save_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "model.pt")
        settings_path = os.path.join(current_dir, "settings.pkl")
        scaler_path = os.path.join(current_dir, "scaler.pkl")

        print("[*] - Saving Model")
        print(f"[*] --- model path: {model_path}")
        print(f"[*] --- settings path: {settings_path}")
        print(f"[*] --- scaler path: {scaler_path}")

        torch.save(self.model.state_dict(), model_path)

        settings = {
            "threshold": self.threshold,
            "bin_nums": self.bin_nums,
            "control_bins": self.control_bins,
            "coef_s_list": self.coef_s_list,
            "coef_b_list": self.coef_b_list,
            "calibration": self.calibration,
        }

        pickle.dump(settings, open(settings_path, "wb"))

        pickle.dump(self.scaler, open(scaler_path, "wb"))

        print("[*] - Model saved")
