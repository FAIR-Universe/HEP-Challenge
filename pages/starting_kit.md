# Starting Kit and Sample Submission

## Starting Kit
We are providing a starting kit as a Google Colab notebook to demonstrate the problem and a solution for it which can be submitted as a submission the competition. You can copy the Colab notebook and make changes as you want.  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FAIR-Universe/HEP-Challenge/blob/master/StartingKit_HiggsML_Uncertainty_Challenge.ipynb)



## Dummy Sample Submission
Dummy sample submission is provided to make you understand what is expected as a submission. You can modify the sample submission the way you want but make sure the format is the same as instructed in the sample submission

### ⬇️ [Dummy Sample Submission](https://www.codabench.org/datasets/download/cf1b2c74-02d8-45a7-885b-6f2c29e2ee9e/)


#### ⚠️ Note:
- Please register with your university or official email for security reasons. 

- Participants can submit a pre-trained model zipped together with the `model.py`. It is the participants responsibility to load the pre-trained model in the Model class. In this case participants may want to ignore the train_set provided as input of the Model class. 

- You can download and see real sample submissions from our github repo: https://github.com/FAIR-Universe/HEP-Challenge/tree/master/example_submissions/

- Participants are encouraged to submit pre-trained models to use the maximum time for predictions rather than using the allowed time for training and then failing to evaluate your model. 

- To download training data click the **`Files`** tab.

- Submissions will be evaluated approximately once per day

- In the `Dummy Sample Submission` we have given you code to locate the pre-trained model in your submission directory. The following code sample can also be helpful.

    ```
    # Files placed in the same zip archive as model.py will be placed here
    model_dir = os.path.dirname(os.path.abspath(__file__))

    ### Loading a PyTorch model
    model = Net().to(device)
    model.load_state_dict(torch.load(model_dir+'/Model.pth',  map_location=device))

            
    ### Loading a NumPy file
    hists = np.load(model_dir+"/Templates.npy")
    ```
-  The Submissions are tested in parallel hence the models have to be serialisable. 



