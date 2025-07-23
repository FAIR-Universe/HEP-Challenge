from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def __init__(self, train_data,Load_Classifier):
        if Load_Classifier == True :
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists("%s/Saved_Classifier/NN"%(current_dir)):
                print("Problem : Saved NN Not Found")
            else :
                self.model=load_model('%s/Saved_Classifier/NN/NN_trained_wt_%s_events.h5'%(current_dir,len(train_data)))
                
        else:
            self.model = Sequential()

            n_dim = train_data.shape[1]

            self.model.add(Dense(100, input_dim=n_dim, activation="relu"))
            self.model.add(Dense(100, activation="relu"))
            self.model.add(Dense(1, activation="si" \
                                 "gmoid"))

            self.model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=5, verbose=2
        )

        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists("%s/Saved_Classifier/NN"%(current_dir)):
            os.makedirs("%s/Saved_Classifier/NN"%(current_dir))
        self.model.save('%s/Saved_Classifier/NN/NN_trained_wt_%s_events.h5'%(current_dir,len(train_data)))

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()
