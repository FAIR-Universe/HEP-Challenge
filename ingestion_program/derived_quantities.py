#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


__doc__ = """
ATLAS Higgs Machine Learning Challenge 2014
Read CERN Open Data Portal Dataset http://opendata.cern.ch/record/328
and manipulate it
 - KaggleWeight and KaggleSet are removed
  - Label is changd from charcter to integer 0 or 1
 - DetailLabel is introduced indicating subpopulations
 - systematics effect are simulated
     - bkg_weight_norm : manipulates the background weight of the W background
     - had_energy_scale : manipulates PRI_had_pt and recompute other quantities accordingly
             Some WARNING : variable DER_mass_MMC is not properly manipulated (modification is linearised), 
             and I advocate to NOT use DER_mass_MMC when doSystTauEnergyScale is enabled
             There is a threshold in the original HiggsML file at 20GeV on PRI_had_energy. 
             This threshold is moved when changing sysTauEnergyScale which is unphysicsal. 
             So if you're going to play with sysTauEnergyScale (within 0.9-1.1), 
             I suggest you remove events below say 22 GeV *after* the manipulation
             applying doSystTauEnergyScale with sysTauENergyScale=1. does NOT yield identical results as not applyield 
             doSystTauEnergyScale, this is because of rounding error and zero mass approximation.
             doSysTauEnerbyScale impacts PRI_had_pt as well as PRI_met and PRI_met_phi
    - so overall I suggest that when playing with doSystTauEnergyScale, the reference is
          - not using DER_mass_MMC
          - applying *after* this manipulation PRI_had_pt>22
          - run with sysTauENergyScale=1. to have the reference
          
Author D. Rousseau LAL, Nov 2016

Modification Dec 2016 (V. Estrade):
- Wrap everything into separated functions.
- V4 class now handle 1D-vector values (to improve computation efficiency).
- Fix compatibility with both python 2 and 3.
- Use pandas.DataFrame to ease computation along columns
- Loading function for the base HiggsML dataset (fetch it on the internet if needed)

Refactor March 2017 (V. Estrade):
- Split load function (cleaner)

July 06 2017 (V. Estrade):
- Add normalization_weight function

May 2019 (D. Rousseau) :
- Major hack, in preparation for Centralesupelec EI,
python syst/datawarehouse/datawarehouse/higgsml.py -i atlas-higgs-challenge-2014-v2.csv.gz -o atlas-higgs-challenge-2014-v2-s0.csv

python higgsml_syst.py -i atlas-higgs-challenge-2014-v2.csv.gz -o atlas-higgs-challenge-2014-v2-syst1.csv --csv -p --BKGnorm 1. --Wnorm 1. --tes 1. --jes 1. --soft_met 0. --seed 31415926
python higgsml_syst.py --help # for command help
reasonable values for parameters
BKGnorm : 1.05  
Wnorm : 1.5 
tes : 1.03
jes : 1.03
soft_met : 3 GeV
"""
__version__ = "4.0"
__author__ = "David Rousseau, and Victor Estrade "


import pandas as pd
import numpy as np
from numpy import sin, cos, cosh, sinh, sqrt, exp


def calcul_int(data):

    # Definition of the x and y components of the hadron's momentum
    data["had_px"] = data.PRI_had_pt * cos(data.PRI_had_phi)
    data["had_py"] = data.PRI_had_pt * sin(data.PRI_had_phi)
    data["had_pz"] = data.PRI_had_pt * sinh(data.PRI_had_eta)
    data["p_had"] = data.PRI_had_pt * cosh(data.PRI_had_eta)

    # Definition of the x and y components of the lepton's momentum
    data["lep_px"] = data.PRI_lep_pt * cos(data.PRI_lep_phi)
    data["lep_py"] = data.PRI_lep_pt * sin(data.PRI_lep_phi)
    data["lep_pz"] = data.PRI_lep_pt * sinh(data.PRI_lep_eta)
    data["p_lep"] = data.PRI_lep_pt * cosh(data.PRI_lep_eta)

    # Definition of the x and y components of the neutrinos's momentum (MET)
    data["met_x"] = data.PRI_met * cos(data.PRI_met_phi)
    data["met_y"] = data.PRI_met * sin(data.PRI_met_phi)

    # Definition of the x and y components of the leading jet's momentum
    data["jet_leading_px"] = (
        data.PRI_jet_leading_pt * cos(data.PRI_jet_leading_phi) * (data.PRI_n_jets >= 1)
    )  # = 0 if PRI_n_jets == 0
    data["jet_leading_py"] = (
        data.PRI_jet_leading_pt * sin(data.PRI_jet_leading_phi) * (data.PRI_n_jets >= 1)
    )
    data["jet_leading_pz"] = (
        data.PRI_jet_leading_pt
        * sinh(data.PRI_jet_leading_eta)
        * (data.PRI_n_jets >= 1)
    )
    data["p_jet_leading"] = (
        data.PRI_jet_leading_pt
        * cosh(data.PRI_jet_leading_eta)
        * (data.PRI_n_jets >= 1)
    )

    # Definition of the x and y components of the subleading jet's momentum
    data["jet_subleading_px"] = (
        data.PRI_jet_subleading_pt
        * cos(data.PRI_jet_subleading_phi)
        * (data.PRI_n_jets >= 2)
    )  # = 0 if PRI_n_jets <= 1
    data["jet_subleading_py"] = (
        data.PRI_jet_subleading_pt
        * sin(data.PRI_jet_subleading_phi)
        * (data.PRI_n_jets >= 2)
    )
    data["jet_subleading_pz"] = (
        data.PRI_jet_subleading_pt
        * sinh(data.PRI_jet_subleading_eta)
        * (data.PRI_n_jets >= 2)
    )
    data["p_jet_subleading"] = (
        data.PRI_jet_subleading_pt
        * cosh(data.PRI_jet_subleading_eta)
        * (data.PRI_n_jets >= 2)
    )

    return data


def f_DER_mass_transverse_met_lep(data):
    """
    Calculate the transverse mass between the MET and the lepton
    Parameters: data (dataframe)
    """
    data["calcul_int"] = (
        (data.PRI_met + data.PRI_lep_pt) ** 2
        - (data.met_x + data.lep_px) ** 2
        - (data.met_y + data.lep_py) ** 2
    )
    data["DER_mass_transverse_met_lep"] = sqrt(data.calcul_int * (data.calcul_int >= 0))
    del data["calcul_int"]
    return data


def f_DER_mass_vis(data):
    """
    Calculate the invariant mass of the hadron and the lepton
    Parameters: data (dataframe)
    """

    data["DER_mass_vis"] = sqrt(
        (data.p_lep + data.p_had) ** 2
        - (data.lep_px + data.had_px) ** 2
        - (data.lep_py + data.had_py) ** 2
        - (data.lep_pz + data.had_pz) ** 2
    )
    return data


def f_DER_pt_h(data):
    """
    Calculate the transverse momentum of the hadronic system
    Parameters: data (dataframe)
    """

    data["DER_pt_h"] = sqrt(
        (data.had_px + data.lep_px + data.met_x) ** 2
        + (data.had_py + data.lep_py + data.met_y) ** 2
    )
    return data


def f_DER_deltaeta_jet_jet(data):
    """
    Calculate the absolute value of the difference of the pseudorapidity of the two jets
    Parameters: data (dataframe)
    """

    data["DER_deltaeta_jet_jet"] = abs(
        data.PRI_jet_subleading_eta - data.PRI_jet_leading_eta
    ) * (data.PRI_n_jets >= 2) - 7 * (data.PRI_n_jets < 2)
    return data


from numpy import sqrt


# undefined if PRI_n_jets <= 1:
def f_DER_mass_jet_jet(data):
    """
    Calculate the invariant mass of the two jets
    Parameters: data (dataframe)
    """

    data["calcul_int"] = (
        (data.p_jet_leading + data.p_jet_subleading) ** 2
        - (data.jet_leading_px + data.jet_subleading_px) ** 2
        - (data.jet_leading_py + data.jet_subleading_py) ** 2
        - (data.jet_leading_pz + data.jet_subleading_pz) ** 2
    )
    data["DER_mass_jet_jet"] = sqrt(data.calcul_int * (data.calcul_int >= 0)) * (
        data.PRI_n_jets >= 2
    ) - 7 * (data.PRI_n_jets <= 1)

    del data["calcul_int"]
    return data


def f_DER_prodeta_jet_jet(data):
    """
    Calculate the product of the pseudorapidities of the two jets
    Parameters: data (dataframe)
    """

    data["DER_prodeta_jet_jet"] = (
        data.PRI_jet_leading_eta * data.PRI_jet_subleading_eta * (data.PRI_n_jets >= 2)
        - 7 * (data.PRI_n_jets <= 1)
    )
    return data


def f_DER_deltar_had_lep(data):
    data["difference2_eta"] = (data.PRI_lep_eta - data.PRI_had_eta) ** 2
    data["difference2_phi"] = (
        np.abs(
            np.mod(data.PRI_lep_phi - data.PRI_had_phi + 3 * np.pi, 2 * np.pi) - np.pi
        )
    ) ** 2
    data["DER_deltar_had_lep"] = sqrt(data.difference2_eta + data.difference2_phi)

    del data["difference2_eta"]
    del data["difference2_phi"]
    return data


def f_DER_pt_tot(data):
    """
    Calculate the total transverse momentum
    Parameters: data (dataframe)
    """
    data["DER_pt_tot"] = sqrt(
        (
            data.had_px
            + data.lep_px
            + data.met_x
            + data.jet_leading_px
            + data.jet_subleading_px
        )
        ** 2
        + (
            data.had_py
            + data.lep_py
            + data.met_y
            + data.jet_leading_py
            + data.jet_subleading_py
        )
        ** 2
    )
    return data


def f_DER_sum_pt(data):
    """
    Calculate the sum of the transverse momentum of the lepton, the hadron and the jets
    Parameters: data (dataframe)
    """

    data["DER_sum_pt"] = data.PRI_had_pt + data.PRI_lep_pt + data.PRI_jet_all_pt
    return data


def f_DER_pt_ratio_lep_had(data):
    """
    Calculate the ratio of the transverse momentum of the lepton and the hadron
    Parameters: data (dataframe)
    """
    data["DER_pt_ratio_lep_had"] = data.PRI_lep_pt / data.PRI_had_pt
    return data


def f_DER_met_phi_centrality(data):
    """
    Calculate the centrality of the MET
    Parameters: data (dataframe)
    """

    def A(met, lep, had):
        return sin(met - lep) * np.sign(sin(had - lep))

    def B(met, lep, had):
        return sin(had - met) * np.sign(sin(had - lep))

    data["A"] = A(data.PRI_met_phi, data.PRI_lep_phi, data.PRI_had_phi)
    data["B"] = B(data.PRI_met_phi, data.PRI_lep_phi, data.PRI_had_phi)
    data["num"] = data.A + data.B
    data["denum"] = sqrt(data.A**2 + data.B**2)

    data["DER_met_phi_centrality"] = data.num / (data.denum + (data.denum == 0)) * (
        data.denum != 0
    ) - 7 * (data.denum == 0)
    epsilon = 0.0001
    mask = data.denum == 0

    data.loc[mask, "A"] = A(
        data.PRI_met_phi, data.PRI_lep_phi + epsilon, data.PRI_had_phi
    )
    data.loc[mask, "B"] = B(
        data.PRI_met_phi, data.PRI_lep_phi + epsilon, data.PRI_had_phi
    )
    data.loc[mask, "num"] = data.A + data.B
    data.loc[mask, "denum"] = sqrt(data.A**2 + data.B**2)
    data.loc[mask, "DER_met_phi_centrality"] = data.num / (
        data.denum + (data.denum == 0)
    ) * (data.denum != 0) - 7 * (data.denum == 0)

    del data["A"]
    del data["B"]
    del data["num"]
    del data["denum"]
    return data


def f_DER_lep_eta_centrality(data):
    """
    Calculate the centrality of the lepton
    Parameters: data (dataframe)
    """

    data["difference"] = (data.PRI_jet_leading_eta - data.PRI_jet_subleading_eta) ** 2
    data["moyenne"] = (data.PRI_jet_leading_eta + data.PRI_jet_subleading_eta) / 2

    data["DER_lep_eta_centrality"] = exp(
        -4 / (data.difference) * ((data.PRI_lep_eta - data.moyenne) ** 2)
    ) * (data.PRI_n_jets >= 2) - 7 * (data.PRI_n_jets <= 1)

    del data["difference"]
    del data["moyenne"]

    return data


def f_del_DER(data):
    """
    Delete all the unnecessary columns that were used to calculate the DER variables
    Parameters: data (dataframe)
    """
    del data["had_px"]
    del data["had_py"]
    del data["had_pz"]
    del data["p_had"]
    del data["lep_px"]
    del data["lep_py"]
    del data["lep_pz"]
    del data["p_lep"]
    del data["met_x"]
    del data["met_y"]
    del data["jet_leading_px"]
    del data["jet_leading_py"]
    del data["jet_leading_pz"]
    del data["p_jet_leading"]
    del data["jet_subleading_px"]
    del data["jet_subleading_py"]
    del data["jet_subleading_pz"]
    del data["p_jet_subleading"]

    return data


def DER_data(data):
    """
    data is supposed to be clean (no Weight, no eventId etc...)
    This function directly modifies the dataframe data so make sure to make a copy if
    you need to keep data
    """
    data = calcul_int(data)
    data = f_DER_mass_transverse_met_lep(data)
    data = f_DER_mass_vis(data)
    data = f_DER_pt_h(data)
    data = f_DER_deltaeta_jet_jet(data)
    data = f_DER_mass_jet_jet(data)
    data = f_DER_prodeta_jet_jet(data)
    data = f_DER_deltar_had_lep(data)
    data = f_DER_pt_tot(data)
    data = f_DER_sum_pt(data)
    data = f_DER_pt_ratio_lep_had(data)
    data = f_DER_met_phi_centrality(data)
    data = f_DER_lep_eta_centrality(data)
    data = f_del_DER(data)
    return data


