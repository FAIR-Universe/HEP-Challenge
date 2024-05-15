# ------------------------------
# Dummy Sample Submission
# ------------------------------
class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init : 
        takes 2 arguments: train_set and systematics, 
        can be used for intiializing variables, classifier etc.
    2) fit : 
        takes no arguments
        can be used to train a classifier
    3) predict: 
        takes 1 argument: test sets
        can be used to get predictions of the test set.
        returns a dictionary

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods

            When you add another file with the submission model e.g. a trained model to be loaded and used,
            load it in the following way:

            # get to the model directory (your submission directory)
            model_dir = os.path.dirname(os.path.abspath(__file__))

            your trained model file is now in model_dir, you can load it from here
    """


    def __init__(
            self,
            get_train_set=None,
            systematics=None
    ):
        """
        Model class constructor

        Params:
            train_set:
                a dictionary with data, labels, weights and settings

            systematics:
                a class which you can use to get a dataset with systematics added
                See sample submission for usage of systematics
        

        Returns:
            None
        """

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model

        Returns:
            None
        """
        pass

    def predict(self, test_set):
        """
        Params:
            test_set

        Functionality:
            this function can be used for predictions using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        return {
            "mu_hat": 0.0,
            "delta_mu_hat": 0.0,
            "p16": 0.0,
            "p84": 0.0
        }
