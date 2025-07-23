from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler


class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    def __init__(self, train_data,Load_Classifier):

        if Load_Classifier == True :
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists("%s/Saved_Classifier/BDT"%(current_dir)):
                print("Problem : Saved BDT Not Found")
            else :
                self.model = XGBClassifier()
                self.model.load_model('%s/Saved_Classifier/BDT/BDT_trained_wt_%s_events.json'%(current_dir,len(train_data)))

        else :
            self.model = XGBClassifier(
            n_estimators=100,     # Number of trees
            learning_rate=0.1,    # Step size shrinkage
            max_depth=10,          # Depth of each tree
            subsample=0.8,        # Row sampling
            colsample_bytree=0.8, # Feature sampling
            use_label_encoder=False,
            eval_metric='logloss' # For classification
            )
            self.scaler = StandardScaler()

    def fit(self, train_data, labels, weights=None):

        self.scaler.fit_transform(train_data)

        X_train_data = self.scaler.transform(train_data)
        self.model.fit(X_train_data, labels, weights)

        import matplotlib.pyplot as plt

        booster = self.model.get_booster() 
        booster.feature_names = [
                    "PRI_lep_pt",
                    "PRI_lep_eta",
                    "PRI_lep_phi",
                    "PRI_had_pt",
                    "PRI_had_eta",
                    "PRI_had_phi",
                    "PRI_jet_leading_pt",
                    "PRI_jet_leading_eta",
                    "PRI_jet_leading_phi",
                    "PRI_jet_subleading_pt",
                    "PRI_jet_subleading_eta",
                    "PRI_jet_subleading_phi",
                    "PRI_n_jets",
                    "PRI_jet_all_pt",
                    "PRI_met",
                    "PRI_met_phi",
                    "DER_mass_transverse_met_lep",
                    "DER_mass_vis",
                    "DER_pt_h",
                    "DER_deltaeta_jet_jet",
                    "DER_mass_jet_jet",
                    "DER_prodeta_jet_jet",
                    "DER_deltar_had_lep",
                    "DER_pt_tot",
                    "DER_sum_pt",
                    "DER_pt_ratio_lep_had",
                    "DER_met_phi_centrality",
                    "DER_lep_eta_centrality",
                ]
        plt.figure(1, figsize=(15,20))
        plot_importance(booster)
        plt.show()

        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists("%s/Saved_Classifier/BDT"%(current_dir)):
            os.makedirs("%s/Saved_Classifier/BDT"%(current_dir))
        self.model.save_model('%s/Saved_Classifier/BDT/BDT_trained_wt_%s_events.json'%(current_dir,len(train_data)))


    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]
