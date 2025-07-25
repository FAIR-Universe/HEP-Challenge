from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler
import joblib

class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    def __init__(self, train_size,seed,Load_Classifier):

        if Load_Classifier == True :
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists("%s/Saved_Classifier/BDT"%(current_dir)):
                print("Problem : Saved BDT Not Found")
            else :
                self.model = XGBClassifier()
                self.model.load_model('%s/Saved_Classifier/BDT/model_BDT_trained_wt_%s_events_seed%s.json'%(current_dir,train_size,seed))
                self.scaler = joblib.load('%s/Saved_Classifier/BDT/scaler_BDT_trained_wt_%s_events_seed%s.pkl'%(current_dir,train_size,seed))
        
        else :
            self.model = XGBClassifier(
            tree_method='gpu_hist',  # New for GPU
            predictor='gpu_predictor',  # New for GPU
            n_estimators=100,     # Number of trees
            learning_rate=0.1,    # Step size shrinkage
            max_depth=10,          # Depth of each tree
            subsample=0.8,        # Row sampling
            colsample_bytree=0.8, # Feature sampling
            use_label_encoder=False,
            eval_metric='logloss', # For classification
            #n_jobs=100,
            )

            self.scaler = StandardScaler()


    def fit(self,train_data, labels, train_size,seed, weights_train=None):

        self.scaler.fit_transform(train_data)

        X_train_data = self.scaler.transform(train_data)
        self.model.fit(X_train_data, labels, weights_train)

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
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists("%s/Images/BDT/Features_Importances"%(current_dir)):
            os.makedirs("%s/Images/BDT/Features_Importances"%(current_dir))
        plt.figure(figsize=(10, 6))
        plot_importance(booster)
        plt.tight_layout()
        plt.savefig("%s/Images/BDT/Features_Importances/BDT_trained_wt_%s_events_seed%s.png"%(current_dir,train_size,seed))
        plt.close()

        
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(self.model, X_train_data, labels, 
                                                n_repeats=20, random_state=1, 
                                                scoring="roc_auc", n_jobs=5)
        sorted_idx = perm_importance.importances_mean.argsort()
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [f'feature_{i}' for i in sorted_idx])
        plt.xlabel('Permutation Feature Importance (based on AUC)')
        plt.ylabel('Feature')
        plt.title('Permutation Feature Importance (XGBoost)')
        plt.tight_layout()
        if not os.path.exists("%s/Images/BDT/Features_Permutation"%(current_dir)):
            os.makedirs("%s/Images/BDT/Features_Permutation"%(current_dir))
        plt.savefig("%s/Images/BDT/Features_Permutation/BDT_trained_wt_%s_events_seed%s.png"%(current_dir,train_size,seed))
        plt.close()
        
        if not os.path.exists("%s/Saved_Classifier/BDT"%(current_dir)):
            os.makedirs("%s/Saved_Classifier/BDT"%(current_dir))
        self.model.save_model('%s/Saved_Classifier/BDT/model_BDT_trained_wt_%s_events_seed%s.json'%(current_dir,train_size,seed))
        joblib.dump(self.scaler, '%s/Saved_Classifier/BDT/scaler_BDT_trained_wt_%s_events_seed%s.pkl'%(current_dir,train_size,seed) )
        print("BDT saved information (model and scaler)")

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]
