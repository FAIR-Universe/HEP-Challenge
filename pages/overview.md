# Overview 
***

## Introduction 
In 2012, the Nobel-prize-winning discovery of the Higgs Boson by the ATLAS and CMS experiments at the Large Hadron Collider (LHC) at CERN in Geneva, Switzerland was a major milestone in the history of physics. However, despite the validation it provided of the Standard Model of particle physics (SM), there are still numerous questions in physics that the SM does not answer. One promissing approach to uncover some of these mysteries is to study the Higgs Boson in great detail, as the rate of Higgs Boson production and its decay properties may hold the secrets to the nature of dark matter and other phenomena not explained by the SM.

The LHC collides protons together at high energy and at a high rate. Each proton collision produces many outgoing particles. A small fraction of the time, a Higgs boson is produced and then decays into other particles that can be detected by experiments such as ATLAS and CMS. In this challenge we focus on the Higgs boson decay to two $\tau$ particles which themselves further decay into other particles before being detected. Measuring this “signal” of interest is complicated by the presence of “background” decays which can produce the same detector signatures. The aim of this challenge is to determine methods to improve the precision of the Higgs boson production rate (“signal strength” or $\mu$), which is defined below, based on feature variables that are derived from simulated measurements made in an ATLAS-like detector. This builds on a previous [Kaggle challenge](https://www.kaggle.com/c/higgs-boson) but focuses directly on the statistical inference, where the goal is not only to isolate Higgs boson events (previous challenge), but to determine the precision on $\mu$ in the presence of uncertainty. Technical details on the signal and background decay channels and features of the dataset are given below, but participants not familiar with particle physics can consider any of these simply as input feature variables for their proposed method except the EventId, Weight and Label (which is essentially the target).

## Problem Setting
***
The objective of this challenge is to improve the signal strength of the Higgs boson decay, specifically in the $H \rightarrow \tau \tau$ decay mode in the presence of the background $Z \rightarrow \tau \tau$ process. In both signal and background events, one of the $\tau$ leptons decays leptonically (electron and muon) and the other one decays hadronically. The dataset includes systematic uncertainty corresponding to the Tau Energy Scale (TES). The main focus of the challenge is to develop an optimal algorithm that is robust in the presence of systematic uncertainty.   


The Large Hadron Collider (LHC) produces high energy events which can be modeled as a Poisson process with an expected arrival rate of $\nu$. Among these high-energy events, physicists search for interesting events that may provide insights into new discoveries. However, most of the events observed are "background" events that are expected and unrelated to new physics. The expected arrival rate of background events, denoted as $\beta$, is well characterized from past experiments and simulations. On the other hand, the arrival rate of signal events, denoted as $\mu \gamma$, is to be measured. Here, $\gamma$ represents the estimated arrival rate of signal events obtained from the Standard Model (SM), which is known to be incomplete. The factor $\mu$ represents the "signal strength" that must be evaluated from the new experiment. All three arrival processes (high energy events, signal, and background) are assumed to be Poisson processes, and their arrival rate expectations are related by the equation $\nu = \beta + \mu \gamma$.

The standard approach used in LHC analyses is to construct a 1D feature $f(x)$ (where $x$ is the set of available feature variables), make a histogram, and then estimate $\mu$ (and its uncertainty) using maximum likelihood estimation.  The likelihood is a product of bin-by-bin Poisson probabilities where the expected counts are determined from simulations. Traditionally, $f(x)$ is selected manually using expert knowledge. It can also be estimated using a machine learning classifier. In the absence of systematic uncertainties from nuisance parameters, and for a particular value of $\mu$, the optimal $f(x)$ is the likelihood ratio. Systematic uncertainties however give rise to distribution shifts, and their impact is represented by nuisance parameters,  which may be constrained on the test set to improve the accuracy and precision of the measurement. In this challenge we construct a realistic setting with nuisance parameters and ask participants to determine $\mu$ and its uncertainty.

## Challenge target: Estimation of $\mu$ and uncertainty
***
The aim of this challenge is to determine the signal strength $\mu$ and its uncertainty. In addition participants submissions will be evaluated against test datasets which can have a range of true “$\mu$” values (for the first “phase” a single $\mu$ test set will be used).  Furthermore the test datasets have systematic uncertainties applied on nuisance parameters at values that may be different to those available during training. Currently for this challenge, we will have a single parameter, representing one of the most important sources of uncertainty: the energy scale of the tau particles (`TES`). 

**The specific target of this challenge is to determine a 68% confidence interval for \mu on these test dataset(s).** We desire the interval to be as small as possible so long as the coverage is close to 68%. **Participants should consult the “Evaluation” tab for more details on what quantity should be returned and how it will be scored, and the “Starting Kit” tab for examples of how to return these values from your models within their submission. In addition the “Starting Kit” provides examples of how to apply TES systematics to the training data when training your model.** Though the values of TES applied in this way may not be the one in the test set, this information can be used to constrain (“profile”) the TES value. It is not necessary for participants to provide the value of TES on the test data (we only care about $\mu$), but it may be necessary to determine TES and its uncertainty in order to win the challenge.




## How to join this challenge?
***
- Go to the "Starting Kit" tab
- Download the "Dummy sample submission" or "sample submission"
- Go to the "My Submissions" tab
- Submit the downloaded file


## Submissions
***
This competition allows code submissions. Participants can submit either of the following:
- code submission without any trained model
- code submission with pre-trained model

## Credits
***
#### Lawrence Berkeley National Laboratory 
- Benjamin Nachman
- Ben Thorne
- Chris Harris
- Sascha Diefenbacher
- Steven Farrell
- Wahid Bhimji

#### University of Washington
- Elham E Khoda
- Shih-Chieh Hsu

#### ChaLearn
- Isabelle Guyon
- Ihsan Ullah

#### Université Paris-Saclay
- David Rousseau
- Ragansu Chakkappai

#### UC Irvine
- Aishik Ghosh


## Contact
***
Visit our website: https://fair-universe.lbl.gov/

Email: fair-universe@lbl.gov

Updates will be announced through fair-universe-announcements google group. [Click to join Google Group](https://groups.google.com/u/0/a/lbl.gov/g/Fair-Universe-Announcements/)

Join #higgsml-uncertainty-challenge-spring-24 channel in [FAIR Universe slack workspace](https://join.slack.com/t/fairuniverse/shared_invite/zt-2dt9ovrp1-jvi0DnCK9jzL3VGrdwYNMA)