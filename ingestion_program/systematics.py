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


import copy
import pandas as pd
import numpy as np
from derived_quantities import DER_data


# ==================================================================================
#  V4 Class and physic computations
# ==================================================================================


class V4:
    """
    A simple 4-vector class to ease calculation, work easy peasy on numpy vector of 4 vector
    """

    px = 0
    py = 0
    pz = 0
    e = 0

    def __init__(self, apx=0.0, apy=0.0, apz=0.0, ae=0.0):
        """
        Constructor with 4 coordinates
        Parameters: apx (float) - x coordinate
                    apy (float) - y coordinate
                    apz (float) - z coordinate
                    ae (float) - energy coordinate
        Returns: None

        """
        self.px = apx
        self.py = apy
        self.pz = apz
        self.e = ae
        if self.e + 1e-3 < self.p():
            raise ValueError(
                "Energy is too small! Energy: {}, p: {}".format(self.e, self.p())
            )

    def copy(self):
        """
        Copy the current V4 object
        Parameters: None
        Returns: copy (V4) - a copy of the current V4 object
        """
        return copy.deepcopy(self)

    def p2(self):
        """
        Compute the squared norm of the 3D momentum
        Parameters: None
        Returns: p2 (float) - squared norm of the 3D momentum

        """
        return self.px**2 + self.py**2 + self.pz**2

    def p(self):
        """
        Compute the norm of the 3D momentum
        Parameters: None
        Returns: p (float) - norm of the 3D momentum

        """
        return np.sqrt(self.p2())

    def pt2(self):
        """
        Compute the squared norm of the transverse momentum
        Parameters: None
        Returns: pt2 (float) - squared norm of the transverse momentum
        """
        return self.px**2 + self.py**2

    def pt(self):
        """
        Compute the norm of the transverse momentum
        Parameters: None
        Returns: pt (float) - norm of the transverse momentum
        """

        return np.sqrt(self.pt2())

    def m(self):
        """
        Compute the mass
        Parameters: None
        Returns: m (float) - mass
        """

        return np.sqrt(np.abs(self.e**2 - self.p2()))  # abs is needed for protection

    def eta(self):
        """
        Compute the pseudo-rapidity
        Parameters: None
        Returns: eta (float) - pseudo-rapidity
        """

        return np.arcsinh(self.pz / self.pt())

    def phi(self):
        """
        Compute the azimuthal angle
        Parameters: None
        Returns: phi (float) - azimuthal angle
        """

        return np.arctan2(self.py, self.px)

    def deltaPhi(self, v):
        """
        Compute the azimuthal angle difference with another V4 object
        Parameters: v (V4) - the other V4 object
        Returns: deltaPhi (float) - azimuthal angle difference
        """

        return (self.phi() - v.phi() + 3 * np.pi) % (2 * np.pi) - np.pi

    def deltaEta(self, v):
        """
        Compute the pseudo-rapidity difference with another V4 object
        Parameters: v (V4) - the other V4 object
        Returns: deltaEta (float) - pseudo-rapidity difference

        """
        return self.eta() - v.eta()

    def deltaR(self, v):
        """
        Compute the delta R with another V4 object
        Parameters: v (V4) - the other V4 object
        Returns: deltaR (float) - delta R with another V4 object
        """

        return np.sqrt(self.deltaPhi(v) ** 2 + self.deltaEta(v) ** 2)

    def eWithM(self, m=0.0):
        """
        Compute the energy with a given mass
        Parameters: m (float) - mass
        Returns: e (float) - energy with a given mass

        """

        return np.sqrt(self.p2() + m**2)

    # FIXME this gives ugly prints with 1D-arrays
    def __str__(self):

        return "PxPyPzE( %s,%s,%s,%s)<=>PtEtaPhiM( %s,%s,%s,%s) " % (
            self.px,
            self.py,
            self.pz,
            self.e,
            self.pt(),
            self.eta(),
            self.phi(),
            self.m(),
        )

    def scale(self, factor=1.0):  # scale
        """Apply a simple scaling"""
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = np.abs(factor * self.e)

    def scaleFixedM(self, factor=1.0):
        """Scale (keeping mass unchanged)"""
        m = self.m()
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = self.eWithM(m)

    def setPtEtaPhiM(self, pt=0.0, eta=0.0, phi=0.0, m=0):
        """Re-initialize with : pt, eta, phi and m"""
        self.px = pt * np.cos(phi)
        self.py = pt * np.sin(phi)
        self.pz = pt * np.sinh(eta)
        self.e = self.eWithM(m)

    def sum(self, v):
        """Add another V4 into self"""
        self.px += v.px
        self.py += v.py
        self.pz += v.pz
        self.e += v.e

    def __iadd__(self, other):
        """Add another V4 into self"""
        try:
            self.px += other.px
            self.py += other.py
            self.pz += other.pz
            self.e += other.e
        except AttributeError:
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return self

    def __add__(self, other):
        """Add 2 V4 vectors : v3 = v1 + v2 = v1.__add__(v2)"""
        copy = self.copy()
        try:
            copy.px += other.px
            copy.py += other.py
            copy.pz += other.pz
            copy.e += other.e
        except AttributeError:
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy


# ==================================================================================


def w_bkg_weight_norm(weights, detailedlabel, systBkgNorm):
    """
    Apply a scaling to the weight. For W background

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    """
    # scale the weight, arbitrary but reasonable value
    weights = (weights * systBkgNorm).where((detailedlabel == "wjets"), other=weights)
    return weights


def all_bkg_weight_norm(weights, label, systBkgNorm):
    """
    Apply a scaling to the weight.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    """
    # scale the weight, arbitrary but reasonable value
    weights = (weights * systBkgNorm).where(label == 0, other=weights)
    return weights


def all_bkg_crossection_norm(crossection_list, systBkgNorm):
    """
    Apply a scaling to the Crosssection.

    Args
    ----
        crossection_list: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    """
    # scale the weight, arbitrary but reasonable value
    crossection_list["crosssection"] = (
        crossection_list["crosssection"] * systBkgNorm
    ).where(crossection_list["Label"] == 0, other=crossection_list["crosssection"])
    return crossection_list


# ==================================================================================
# Manipulate the 4-momenta
# ==================================================================================
def mom4_manipulate(data, systTauEnergyScale, systJetEnergyScale, soft_met, seed=31415):
    """
    Manipulate primary inputs : the PRI_had_pt PRI_jet_leading_pt PRI_jet_subleading_pt and recompute the others values accordingly.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        systTauEnergyScale : the factor applied : PRI_had_pt <-- PRI_had_pt * systTauEnergyScale
        systJetEnergyScale : the factor applied : all jet pt  * systJetEnergyScale
        recompute MET accordingly
        Add soft MET gaussian random energy


    Notes :
    -------
        Recompute :
            - PRI_had_pt
            - PRI_jet_leading_pt
            - PRI_jet_subleading_pt
            - PRI_met
            - PRI_met_phi
            - PRI_met_sumet
        Round up to 3 decimals.

    """

    vmet = V4()  # met 4-vector
    vmet.setPtEtaPhiM(data["PRI_met"], 0.0, data["PRI_met_phi"], 0.0)  # met mass zero,
    # met_sumet=data["PRI_met_sumet"]

    if systTauEnergyScale != 1.0:
        # scale tau energy scale, arbitrary but reasonable value
        data["PRI_had_pt"] *= systTauEnergyScale

        # build 4-vectors
        vtau = V4()  # tau 4-vector
        vtau.setPtEtaPhiM(
            data["PRI_had_pt"], data["PRI_had_eta"], data["PRI_had_phi"], 0.8
        )  # tau mass 0.8 like in original

        # vlep = V4() # lepton 4-vector
        # vlep.setPtEtaPhiM(data["PRI_lep_pt"], data["PRI_lep_eta"], data["PRI_lep_phi"], 0.) # lep mass 0 (either 0.106 or 0.0005 but info is lost)

        # fix MET according to tau pt change (minus sign since met is minus sum pt of visible particles
        vtauDeltaMinus = vtau.copy()
        vtauDeltaMinus.scaleFixedM((1.0 - systTauEnergyScale) / systTauEnergyScale)
        vmet += vtauDeltaMinus
        vmet.pz = 0.0
        vmet.e = vmet.eWithM(0.0)

        # met_sum_et is increased if energy scale increased
        # tauDeltaMinus = vtau.pt()
        # met_sumet+= (systTauEnergyScale-1)/systTauEnergyScale *tauDeltaMinus

    # scale jet energy scale, arbitrary but reasonable value

    if systJetEnergyScale != 1.0:
        # data["PRI_jet_leading_pt"]    *= systJetEnergyScale
        data["PRI_jet_leading_pt"] = np.where(
            data["PRI_n_jets"] > 0,
            data["PRI_jet_leading_pt"] * systJetEnergyScale,
            data["PRI_jet_leading_pt"],
        )
        # data["PRI_jet_subleading_pt"] *= systJetEnergyScale
        data["PRI_jet_subleading_pt"] = np.where(
            data["PRI_n_jets"] > 1,
            data["PRI_jet_subleading_pt"] * systJetEnergyScale,
            data["PRI_jet_subleading_pt"],
        )

        data["PRI_jet_all_pt"] *= systJetEnergyScale

        # jet_all_pt = data["PRI_jet_all_pt"]
        # # met_sum_et is increased if energy scale increased
        # met_sumet+= (systJetEnergyScale-1)/systJetEnergyScale *jet_all_pt

        # first jet if it exists
        vj1 = V4()
        vj1.setPtEtaPhiM(
            data["PRI_jet_leading_pt"].where(data["PRI_n_jets"] > 0, other=0),
            data["PRI_jet_leading_eta"].where(data["PRI_n_jets"] > 0, other=0),
            data["PRI_jet_leading_phi"].where(data["PRI_n_jets"] > 0, other=0),
            0.0,
        )  # zero mass
        # fix MET according to leading jet pt change
        vj1DeltaMinus = vj1.copy()
        vj1DeltaMinus.scaleFixedM((1.0 - systJetEnergyScale) / systJetEnergyScale)
        vmet += vj1DeltaMinus
        vmet.pz = 0.0
        vmet.e = vmet.eWithM(0.0)

        # second jet if it exists
        vj2 = V4()
        vj2.setPtEtaPhiM(
            data["PRI_jet_subleading_pt"].where(data["PRI_n_jets"] > 1, other=0),
            data["PRI_jet_subleading_eta"].where(data["PRI_n_jets"] > 1, other=0),
            data["PRI_jet_subleading_phi"].where(data["PRI_n_jets"] > 1, other=0),
            0.0,
        )  # zero mass

        # fix MET according to leading jet pt change
        vj2DeltaMinus = vj2.copy()
        vj2DeltaMinus.scaleFixedM((1.0 - systJetEnergyScale) / systJetEnergyScale)
        vmet += vj2DeltaMinus
        vmet.pz = 0.0
        vmet.e = vmet.eWithM(0.0)

    # note that in principle we should also fix MET for the third jet or more but we do not have enough information

    if soft_met > 0:
        # add soft met term
        # Compute the missing v4 vector
        random_state = np.random.RandomState(seed=seed)
        SIZE = data.shape[0]
        v4_soft_term = V4()
        v4_soft_term.px = random_state.normal(0, soft_met, size=SIZE)
        v4_soft_term.py = random_state.normal(0, soft_met, size=SIZE)
        v4_soft_term.pz = np.zeros(SIZE)
        v4_soft_term.e = v4_soft_term.eWithM(0.0)
        # fix MET according to soft term
        vmet = vmet + v4_soft_term

    data["PRI_met"] = vmet.pt()
    data["PRI_met_phi"] = vmet.phi()
    #     data["PRI_met_sumet"] = met_sumet

    # Fix precision to 3 decimals
    DECIMALS = 3

    data["PRI_had_pt"] = data["PRI_had_pt"].round(decimals=DECIMALS)
    data["PRI_had_eta"] = data["PRI_had_eta"].round(decimals=DECIMALS)
    data["PRI_had_phi"] = data["PRI_had_phi"].round(decimals=DECIMALS)
    data["PRI_lep_pt"] = data["PRI_lep_pt"].round(decimals=DECIMALS)
    data["PRI_lep_eta"] = data["PRI_lep_eta"].round(decimals=DECIMALS)
    data["PRI_lep_phi"] = data["PRI_lep_phi"].round(decimals=DECIMALS)
    data["PRI_met"] = data["PRI_met"].round(decimals=DECIMALS)
    data["PRI_met_phi"] = data["PRI_met_phi"].round(decimals=DECIMALS)
    #     data["PRI_met_sumet"] = data["PRI_met_sumet"].round(decimals=DECIMALS)
    data["PRI_jet_leading_pt"] = data["PRI_jet_leading_pt"].round(decimals=DECIMALS)
    data["PRI_jet_leading_eta"] = data["PRI_jet_leading_eta"].round(decimals=DECIMALS)
    data["PRI_jet_leading_phi"] = data["PRI_jet_leading_phi"].round(decimals=DECIMALS)
    data["PRI_jet_subleading_pt"] = data["PRI_jet_subleading_pt"].round(
        decimals=DECIMALS
    )
    data["PRI_jet_subleading_eta"] = data["PRI_jet_subleading_eta"].round(
        decimals=DECIMALS
    )
    data["PRI_jet_subleading_phi"] = data["PRI_jet_subleading_phi"].round(
        decimals=DECIMALS
    )
    data["PRI_jet_all_pt"] = data["PRI_jet_all_pt"].round(decimals=DECIMALS)

    return data


def make_unweighted_set(data_set):
    keys = ["htautau", "ztautau", "wjets", "ttbar", "diboson"]
    unweighted_set = {}
    for key in keys:
        unweighted_set[key] = data_set["data"][data_set["detailedlabel"] == key].sample(
            frac=1, random_state=31415
        )

    return unweighted_set


def postprocess(data):

    data = data.drop(data[data.PRI_had_pt < 26].index)
    data = data.drop(data[data.PRI_lep_pt < 20].index)
    # data = data.drop(data[data.PRI_met>70].index)

    return data


def systematics(
    data_set=None,
    tes=1.0,
    jes=1.0,
    soft_met=1.0,
    seed=31415,
    w_scale=None,
    bkg_scale=None,
    verbose=0,
):
    """
    Params:
    -------
    data:
        dataframe
    tes:
        between 0.9 and 1.1, default: 1.0
        1.0 means no systemtics
    jes:
        default: 1.0
    soft_met:
        default: 1.0
    w_scale:
        default: None
    bkg_scale:
        default: None
    """

    if w_scale is not None:
        if "weights" in data_set.keys():
            print("W bkg weight rescaling :", w_scale)
            data_set["weights"] = w_bkg_weight_norm(
                data_set["weights"], data_set["detailed_labels"], w_scale
            )

    if bkg_scale is not None:
        if "weights" in data_set.keys():
            print("All bkg weight rescaling :", bkg_scale)
            data_set["weights"] = all_bkg_weight_norm(
                data_set["weights"], data_set["label"], bkg_scale
            )

    if verbose > 0:
        print("Tau energy rescaling :", tes)
    data = mom4_manipulate(
        data=data_set["data"].copy(),
        systTauEnergyScale=tes,
        systJetEnergyScale=jes,
        soft_met=soft_met,
        seed=seed,
    )

    df = DER_data(data)
    for key in data_set.keys():
        if key is not "data":
            df[key] = data_set[key]

    data_syst = postprocess(df)

    data_syst_set = {}
    for key in data_set.keys():
        if key is not "data":
            data_syst_set[key] = data_syst.pop(key)
    data_syst_set["data"] = data_syst

    return data_syst_set


LHC_NUMBERS = {
    "ztautau": 7306660,
    "wjets": 3812700,
    "diboson": 2398564,
    "ttbar": 616017,
    "htautau": 285
}

def get_bootstraped_dataset(
    test_set,
    mu=1.0,
    seed=31415,
    w_scale=1.0,
    bkg_scale=1.0,
):

    bkg_norm = LHC_NUMBERS
    if w_scale is not None:
        bkg_norm["wjets"] = int(LHC_NUMBERS["wjets"] * w_scale * bkg_scale)

    if bkg_scale is not None:
        bkg_norm["ztautau"] = int(LHC_NUMBERS["ztautau"] * bkg_scale)
        bkg_norm["diboson"] = int(LHC_NUMBERS["diboson"] * bkg_scale)
        bkg_norm["ttbar"] = int(LHC_NUMBERS["ttbar"] * bkg_scale)

    bkg_norm["htautau"] = int(LHC_NUMBERS["htautau"] * mu)

    pseudo_data = []
    for key in test_set.keys():
        temp = (test_set[key].sample(
            n=bkg_norm[key], replace=True, random_state=seed
        ))
        print(f" for set {key} lenth = {len(temp)}" )
        pseudo_data.append(temp)
              

    pseudo_data = pd.concat(pseudo_data)

    pseudo_data = pseudo_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    return pseudo_data


def get_systematics_dataset(
    data,
    tes=1.0,
    jes=1.0,
    soft_met=1.0,
):
    weights = np.ones(data.shape[0])

    data_syst = systematics(
        data_set={"data": data, "weights": weights},
        tes=tes,
        jes=jes,
        soft_met=soft_met,
    )


    return data_syst
