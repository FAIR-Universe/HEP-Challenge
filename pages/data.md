# Data
***

The dataset is created using the particle physics simulation tools Pythia 8.2 and Delphes 3.5.0. The proton-proton collision events are generated with a center of mass energy of 13 TeV using Pythia8. Subsequently, these events undergo the Delphes tool to produce simulated detector measurements. We used an ATLAS-like detector description to make the dataset closer to experimental data. The events are divided into two groups: 
1. Higgs signal ($H \rightarrow \tau \tau$)
2. $Z$ boson background ($Z \rightarrow \tau \tau$) 

### Higgs Signal: 
The Higgs bosons are produced with all possible production modes and decay into two tau leptons. The taus leptons are further allowed to decay into all possible final states. 

### Z boson Background: 
Only background events coming from Z boson are included in this challenge. While simulating the process, interference effects between Z boson and Photon are included. Similar to signal events, only the tau-tau decay mode of the Z boson is included in the dataset.

 
> **NOTE**   
> 
> The training events have weights.
>
> **Event Weights:**
>
> The challenge is considering a scenario of analyzing $139 ~\textrm{fb} ^{-1}$ of proton-proton collision data collected by the ATLAS experiment during the Run-II phase (2015-2018) of the LHC.
>   
> Event weights are defined as:
>
> $ w = \frac{\textrm{Cross-Section} ~ \times ~ \textrm{Luminosity}}{\textrm{Total number of generated events}} $




---

**The following are the features in this data set.**

|     | Variable                     | Description                                                                                       |
| --- | ---------------------------- | ------------------------------------------------------------------------------------------------- |
| 1.  | EventId                      | A unique integer identifier of the event. {should **NOT** be used for training} |
| 2.  | DER_mass_transverse_met_lep  | The transverse mass between the missing transverse energy and the lepton.                         |
| 3.  | DER_mass_vis                 | The invariant mass of the hadronic tau and the lepton.                                           |
| 4.  | DER_pt_h                     | The modulus of the vector sum of the transverse momentum of the hadronic tau, the lepton and the missing transverse energy vector. |
| 5.  | DER_deltaeta_jet_jet         | The absolute value of the pseudorapidity separation between the two jets (undefined if PRI_jet_num ≤ 1). |
| 6.  | DER_mass_jet_jet             | The invariant mass of the two jets (undefined if PRI_jet_num ≤ 1).                                |
| 7.  | DER_prodeta_jet_jet          | The product of the pseudorapidities of the two jets (undefined if PRI_jet_num ≤ 1).              |
| 8.  | DER_deltar_had_lep           | The R separation between the hadronic tau and the lepton.                                        |
| 9. | DER_pt_tot                   | The modulus of the vector sum of the missing transverse momenta and the transverse momenta of the hadronic tau, the lepton, the leading jet (if PRI_jet_num ≥ 1) and the subleading jet (if PRI_jet_num = 2) (but not of any additional jets). |
| 10. | DER_sum_pt                   | The sum of the moduli of the transverse momenta of the hadronic tau, the lepton, the leading jet (if PRI_jet_num ≥ 1) and the subleading jet (if PRI_jet_num = 2) and the other jets (if PRI_jet_num = 3). |
| 11. | DER_pt_ratio_lep_tau         | The ratio of the transverse momenta of the lepton and the hadronic tau.                           |
| 12. | DER_met_phi_centrality       | The centrality of the azimuthal angle of the missing transverse energy vector w.r.t. the hadronic tau and the lepton. |
| 13. | DER_lep_eta_centrality       | The centrality of the pseudorapidity of the lepton w.r.t. the two jets (undefined if PRI_jet_num ≤ 1). |
| 14. | Eletron Flag                 | 1 if it is an electron 0 if it not.                                                               |
| 15. | muon Flag                    | 1 if it is a muon 0 if it not.                                                                   |
| 16. | PRI_had_pt                   | The transverse momentum $\sqrt{{p_x}^2 + {p_x}^2}$ of the hadronic tau.                          |
| 17. | PRI_had_eta                  | The pseudorapidity $\eta$ of the hadronic tau.                                                    |
| 18. | PRI_had_phi                  | The azimuth angle $\phi$ of the hadronic tau.                                                     |
| 19. | PRI_lep_pt                   | The transverse momentum $\sqrt{{p_x}^2 + {p_x}^2}$ of the lepton (electron or muon).             |
| 20. | PRI_lep_eta                  | The pseudorapidity $\eta$ of the lepton.                                                           |
| 21. | PRI_lep_phi                  | The azimuth angle $\phi$ of the lepton.                                                            |
| 22. | PRI_met                      | The missing transverse energy $\overrightarrow{E}^{miss}_{T}$.                                    |
| 23. | PRI_met_phi                  | The azimuth angle $\phi$ of the missing transverse energy.                                        |
| 24. | PRI_jet_num                  | The number of jets (integer with a value of 0, 1, 2 or 3; possible larger values have been capped at 3). |
| 25. | PRI_jet_leading_pt           | The transverse momentum $\sqrt{{p_x}^2 + {p_x}^2}$ of the leading jet, that is the jet with the largest transverse momentum (undefined if PRI_jet_num = 0). |
| 26. | PRI_jet_leading_eta          | The pseudorapidity $\eta$ of the leading jet (undefined if PRI_jet_num = 0).                     |
| 27. | PRI_jet_leading_phi          | The azimuth angle $\phi$ of the leading jet (undefined if PRI_jet_num = 0).                      |
| 28. | PRI_jet_subleading_pt        | The transverse momentum $\sqrt{{p_x}^2 + {p_x}^2}$ of the leading jet, that is, the jet with the second largest transverse momentum (undefined if PRI_jet_num ≤ 1). |
| 29. | PRI_jet_subleading_eta       | The pseudorapidity $\eta$ of the subleading jet (undefined if PRI_jet_num ≤ 1).                  |
| 30. | PRI_jet_subleading_phi       | The azimuth angle $\phi$ of the subleading jet (undefined if PRI_jet_num ≤ 1).                   |
| 31. | PRI_jet_all_pt               | The scalar sum of the transverse momentum of all the jets of the events.                          |
| 32. | Weight                       | The event weight $w_i$.                                                                                  |
| 33. | Label                        | The event label $y_i \in \{1,0\}$ (1 for signal, 0 for background). {should **NOT** be used as a  training feature}                    |



## How to get Public Data?
***
- Go to the "Files" tab
- Download the "Uncertainty challenge Open data"

or use the following command to download using terminal
```
wget -O Uncertainty_challenge_open_data.zip https://www.codabench.org/datasets/download/b51439bf-9221-41d4-aca6-51a0a5a13f9c/
```