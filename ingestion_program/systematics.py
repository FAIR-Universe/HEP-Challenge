#!/usr/bin/env python
# -*- coding: utf-8 -*-


__doc__ = """
This module contains the systematics functions for the FAIR Challenge.
Originally written by David Rousseau, and Victor Estrade.
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

        Parameters:
            apx (float): x coordinate
            apy (float): y coordinate
            apz (float): z coordinate
            ae (float): energy coordinate

        Returns:
            None
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

        Parameters:
            None

        Returns:
            copy (V4): a copy of the current V4 object
        """
        return copy.deepcopy(self)

    def p2(self):
        """
        Compute the squared norm of the 3D momentum

        Parameters:
            None

        Returns:
            p2 (float): squared norm of the 3D momentum
        """
        return self.px**2 + self.py**2 + self.pz**2

    def p(self):
        """
        Compute the norm of the 3D momentum

        Parameters:
            None

        Returns:
            p (float): norm of the 3D momentum

        """
        return np.sqrt(self.p2())

    def pt2(self):
        """
        Compute the squared norm of the transverse momentum

        Parameters:
            None

        Returns:
            pt2 (float): squared norm of the transverse momentum
        """
        return self.px**2 + self.py**2

    def pt(self):
        """
        Compute the norm of the transverse momentum

        Parameters:
            None

        Returns:
            pt (float): norm of the transverse momentum
        """

        return np.sqrt(self.pt2())

    def m(self):
        """
        Compute the mass

        Parameters:
            None

        Returns:
            m (float): mass
        """

        return np.sqrt(np.abs(self.e**2 - self.p2()))  # abs is needed for protection

    def eta(self):
        """
        Compute the pseudo-rapidity

        Parameters:
            None

        Returns:
            eta (float): pseudo-rapidity
        """

        return np.arcsinh(self.pz / self.pt())

    def phi(self):
        """
        Compute the azimuthal angle

        Parameters:
            None

        Returns:
            phi (float): azimuthal angle
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

        Parameters:
            v (V4): the other V4 object

        Returns:
            deltaPhi (float): azimuthal angle difference

        """
        return self.eta() - v.eta()

    def deltaR(self, v):
        """
        Compute the delta R with another V4 object

        Parameters:
            v (V4): the other V4 object

        Returns:
            deltaEta (float): pseudo-rapidity difference
        """

        return np.sqrt(self.deltaPhi(v) ** 2 + self.deltaEta(v) ** 2)

    def eWithM(self, m=0.0):
        """
        Compute the energy with a given mass

        Parameters:
            m (float): mass

        Returns:
            e (float): energy with a given mass

        """

        return np.sqrt(self.p2() + m**2)

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

    def scale(self, factor=1.0):
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
            return NotImplemented
        return copy


def ttbar_bkg_weight_norm(weights, detailedlabel, systBkgNorm):
    """
    Apply a scaling to the weight. For ttbar background

    Args:
        * weights (array-like): The weights to be scaled
        * detailedlabel (array-like): The detailed labels
        * systBkgNorm (float): The scaling factor

    Returns:
        array-like: The scaled weights
    """
    weights[detailedlabel == "ttbar"] = weights[detailedlabel == "ttbar"] * systBkgNorm
    return weights


def diboson_bkg_weight_norm(weights, detailedlabel, systBkgNorm):
    """
    Apply a scaling to the weight. For Diboson background

    Args:
        * weights (array-like): The weights to be scaled
        * detailedlabel (array-like): The detailed labels
        * systBkgNorm (float): The scaling factor

    
    Returns:
        array-like: The scaled weights

    """
    weights[detailedlabel == "diboson"] = (
        weights[detailedlabel == "diboson"] * systBkgNorm
    )
    return weights


def all_bkg_weight_norm(weights, label, systBkgNorm):
    """
    Apply a scaling to the weight.

    Args:
        weights (array-like): The weights to be scaled
        label (array-like): The labels
        systBkgNorm (float): The scaling factor

    Returns:
        array-like: The scaled weights

    """
    weights[label == 0] = weights[label == 0] * systBkgNorm
    return weights


# ==================================================================================
# Manipulate the 4-momenta
# ==================================================================================
def mom4_manipulate(data, systTauEnergyScale, systJetEnergyScale, soft_met, seed=31415):
    """
    Manipulate primary inputs : the PRI_had_pt PRI_jet_leading_pt PRI_jet_subleading_pt and recompute the others values accordingly.

    Args:
        * data (pandas.DataFrame): The dataset to be manipulated
        * systTauEnergyScale (float): The factor applied to PRI_had_pt
        * systJetEnergyScale (float): The factor applied to all jet pt
        * soft_met (float): The additional soft MET energy
        * seed (int): The random seed

    Returns:
        pandas.DataFrame: The manipulated dataset

    """

    vmet = V4()
    vmet.setPtEtaPhiM(data["PRI_met"], 0.0, data["PRI_met_phi"], 0.0)
    # met_sumet=data["PRI_met_sumet"]

    if systTauEnergyScale != 1.0:
        data["PRI_had_pt"] *= systTauEnergyScale

        vtau = V4()
        vtau.setPtEtaPhiM(
            data["PRI_had_pt"], data["PRI_had_eta"], data["PRI_had_phi"], 0.8
        )

        vtauDeltaMinus = vtau.copy()
        vtauDeltaMinus.scaleFixedM((1.0 - systTauEnergyScale) / systTauEnergyScale)
        vmet += vtauDeltaMinus
        vmet.pz = 0.0
        vmet.e = vmet.eWithM(0.0)

    if systJetEnergyScale != 1.0:
        data["PRI_jet_leading_pt"] = np.where(
            data["PRI_n_jets"] > 0,
            data["PRI_jet_leading_pt"] * systJetEnergyScale,
            data["PRI_jet_leading_pt"],
        )

        data["PRI_jet_subleading_pt"] = np.where(
            data["PRI_n_jets"] > 1,
            data["PRI_jet_subleading_pt"] * systJetEnergyScale,
            data["PRI_jet_subleading_pt"],
        )

        data["PRI_jet_all_pt"] *= systJetEnergyScale

        vj1 = V4()
        vj1.setPtEtaPhiM(
            data["PRI_jet_leading_pt"].where(data["PRI_n_jets"] > 0, other=0),
            data["PRI_jet_leading_eta"].where(data["PRI_n_jets"] > 0, other=0),
            data["PRI_jet_leading_phi"].where(data["PRI_n_jets"] > 0, other=0),
            0.0,
        )

        vj1DeltaMinus = vj1.copy()
        vj1DeltaMinus.scaleFixedM((1.0 - systJetEnergyScale) / systJetEnergyScale)
        vmet += vj1DeltaMinus
        vmet.pz = 0.0
        vmet.e = vmet.eWithM(0.0)

        vj2 = V4()
        vj2.setPtEtaPhiM(
            data["PRI_jet_subleading_pt"].where(data["PRI_n_jets"] > 1, other=0),
            data["PRI_jet_subleading_eta"].where(data["PRI_n_jets"] > 1, other=0),
            data["PRI_jet_subleading_phi"].where(data["PRI_n_jets"] > 1, other=0),
            0.0,
        )

        vj2DeltaMinus = vj2.copy()
        vj2DeltaMinus.scaleFixedM((1.0 - systJetEnergyScale) / systJetEnergyScale)
        vmet += vj2DeltaMinus
        vmet.pz = 0.0
        vmet.e = vmet.eWithM(0.0)

    if soft_met > 0:
        random_state = np.random.RandomState(seed=seed)
        SIZE = data.shape[0]
        v4_soft_term = V4()
        v4_soft_term.px = random_state.normal(0, soft_met, size=SIZE)
        v4_soft_term.py = random_state.normal(0, soft_met, size=SIZE)
        v4_soft_term.pz = np.zeros(SIZE)
        v4_soft_term.e = v4_soft_term.eWithM(0.0)
        vmet = vmet + v4_soft_term

    data["PRI_met"] = vmet.pt()
    data["PRI_met_phi"] = vmet.phi()

    DECIMALS = 3

    data["PRI_had_pt"] = data["PRI_had_pt"].round(decimals=DECIMALS)
    data["PRI_had_eta"] = data["PRI_had_eta"].round(decimals=DECIMALS)
    data["PRI_had_phi"] = data["PRI_had_phi"].round(decimals=DECIMALS)
    data["PRI_lep_pt"] = data["PRI_lep_pt"].round(decimals=DECIMALS)
    data["PRI_lep_eta"] = data["PRI_lep_eta"].round(decimals=DECIMALS)
    data["PRI_lep_phi"] = data["PRI_lep_phi"].round(decimals=DECIMALS)
    data["PRI_met"] = data["PRI_met"].round(decimals=DECIMALS)
    data["PRI_met_phi"] = data["PRI_met_phi"].round(decimals=DECIMALS)
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
    keys = ["htautau", "ztautau", "ttbar", "diboson"]
    unweighted_set = {}
    for key in keys:
        unweighted_set[key] = data_set["data"][data_set["detailedlabel"] == key].sample(
            frac=1, random_state=31415
        )

    return unweighted_set


def postprocess(data):
    """
    Select the events with the following conditions:
    * PRI_had_pt > 26
    * PRI_jet_leading_pt > 26
    * PRI_jet_subleading_pt > 26
    * PRI_lep_pt > 20

    This is applied to the dataset after the systematics are applied

    Args:
        data (pandas.DataFrame): The manipulated dataset

    Returns:
        pandas.DataFrame: The postprocessed dataset
    """
    # apply higher threshold on had pt (dropping events)
    data = data.drop(data[data.PRI_had_pt < 26].index)
    
    #need to reindex
    data.reset_index(drop=True, inplace=True)

    # apply threshold on leading and subleading jets if they exist
    # note that it is assumed that the systematics transformation is monotonous in pt
    # so that leading and subleading jet should never be swapped

    # if subleading jet pt below high threshold, do so it never existed
    mask = data['PRI_jet_subleading_pt'].between(0, 26)
    data.loc[mask, 'PRI_jet_all_pt'] -= data['PRI_jet_subleading_pt']
    data.loc[mask, 'PRI_jet_subleading_pt'] = -25
    data.loc[mask, 'PRI_jet_subleading_eta'] = -25
    data.loc[mask, 'PRI_jet_subleading_phi'] = -25
    data.loc[mask, 'PRI_n_jets'] -= 1

    # if leading jet pt below high threshold, do so it never existed
    mask = data['PRI_jet_leading_pt'].between(0, 26)
    data.loc[mask, 'PRI_jet_all_pt'] -= data['PRI_jet_leading_pt']
    data.loc[mask, 'PRI_jet_leading_pt'] = -25
    data.loc[mask, 'PRI_jet_leading_eta'] = -25
    data.loc[mask, 'PRI_jet_leading_phi'] = -25
    data.loc[mask, 'PRI_n_jets'] -= 1



    # apply low threshold on lepton pt (does nothing)
    data = data.drop(data[data.PRI_lep_pt < 20].index)

    return data


def systematics(
    data_set=None,
    tes=1.0,
    jes=1.0,
    soft_met=0.0,
    seed=31415,
    ttbar_scale=None,
    diboson_scale=None,
    bkg_scale=None,
    dopostprocess=True,
):
    """
    Apply systematics to the dataset

    Args:
        * data_set (dict)/(df): The dataset to apply systematics to
        * tes (float): The factor applied to PRI_had_pt
        * jes (float): The factor applied to all jet pt
        * soft_met (float): The additional soft MET energy
        * seed (int): The random seed
        * ttbar_scale (float): The scaling factor for ttbar background
        * diboson_scale (float): The scaling factor for diboson background
        * bkg_scale (float): The scaling factor for other backgrounds

    Returns:
        dict: The dataset with applied systematics
    """
    if isinstance(data_set, pd.DataFrame):
        data_df = data_set.copy()
        data_type = "df"
    elif isinstance(data_set, dict):
        if "data" not in data_set.keys():
            raise ValueError("data_set must contain a 'data' key")
        if "weights" not in data_set.keys():
            raise ValueError("data_set must contain a 'weights' key")
        
        data_type = "dict"
        
        data_df = data_set["data"].copy()
        for key in data_set.keys():
            if key not in ["data","settings"]:
                data_df[key] = data_set[key]
    else:
        raise ValueError("data_set must be a pandas DataFrame or a dictionary")
        

    # modify primary features according to tes, jes softmet    
    data_syst = mom4_manipulate(
        data=data_df.copy(),
        systTauEnergyScale=tes,
        systJetEnergyScale=jes,
        soft_met=soft_met,
        seed=seed,
    )

    if dopostprocess:
        # deal with thresholds on had pt and jet pt
        # possibly remove sub leading jet
        # possibly remove events
        data_syst = postprocess(data_syst)

    if "detailedlabel" in data_syst.columns:
        if "weights" in data_syst.columns:
            weights = data_syst["weights"].copy()
        else:
            weights = np.ones(data_syst.shape[0])

        if ttbar_scale is not None:
            weights = ttbar_bkg_weight_norm(
                weights, data_syst["detailed_labels"], ttbar_scale
            )

        if diboson_scale is not None:
            weights = diboson_bkg_weight_norm(
                weights, data_syst["detailed_labels"], diboson_scale
            )

        if bkg_scale is not None:
            weights = all_bkg_weight_norm(weights, data_syst["labels"], bkg_scale)

        data_syst["weights"] = weights

    # build resulting dictionary / dataframe
    
    if data_type == "df":
        return DER_data(data_syst)
    else:
        data_syst_set = {}
        for key in data_set.keys():
            if key not in ["data","settings"]:
                data_syst_set[key] = data_syst.pop(key)
        # compute DERived features        
        data_syst_set["data"] = DER_data(data_syst)
        if "settings" in data_set.keys():
            data_syst_set["settings"] = data_set["settings"]
        
        return data_syst_set


def get_bootstrapped_dataset(
    test_set,
    mu=1.0,
    seed=31415,
    ttbar_scale=None,
    diboson_scale=None,
    bkg_scale=None,
    poisson=True,
):
    """
    Generate a bootstrapped dataset

    Args:
        * test_set (dict): The original test dataset
        * mu (float): The scaling factor for htautau background
        * seed (int): The random seed
        * ttbar_scale (float): The scaling factor for ttbar background
        * diboson_scale (float): The scaling factor for diboson background
        * bkg_scale (float): The scaling factor for other backgrounds

    Returns:
        pandas.DataFrame: The bootstrapped dataset
    """
    bkg_norm = {
        "ztautau": 1.0,
        "diboson": 1.0,
        "ttbar": 1.0,
        "htautau": 1.0,
    }

    if bkg_scale is not None:
        bkg_scale_ = bkg_scale
    else:
        bkg_scale_ = 1.0

    if ttbar_scale is not None:
        bkg_norm["ttbar"] = ttbar_scale * bkg_scale_

    if diboson_scale is not None:
        bkg_norm["diboson"] = diboson_scale * bkg_scale_

    if bkg_scale is not None:
        bkg_norm["ztautau"] = bkg_scale_

    bkg_norm["htautau"] = mu

    pseudo_data = []
    Seed = seed
    for i, key in enumerate(test_set.keys()):
        Seed = Seed + i

        if poisson:
            random_state = np.random.RandomState(seed=Seed)
            new_weights = random_state.poisson(bkg_norm[key] * test_set[key]["weights"])
        else:
            new_weights = bkg_norm[key] * test_set[key]["weights"]

        temp_data = test_set[key][new_weights > 0]

        temp_data.loc[:, "weights"] = new_weights[new_weights > 0]

        pseudo_data.append(temp_data)

    pseudo_data = pd.concat(pseudo_data)

    pseudo_data.reset_index(drop=True, inplace=True)

    unweighted_data = repeat_rows_by_weight(pseudo_data.copy(), seed=seed)

    return unweighted_data


def get_systematics_dataset(
    data,
    tes=1.0,
    jes=1.0,
    soft_met=0.0,
):

    weights = np.ones(data.shape[0])

    data_syst = systematics(
        data_set={"data": data, "weights": weights},
        tes=tes,
        jes=jes,
        soft_met=soft_met,
    )

    return data_syst


# Assuming 'data_set' is a DataFrame with a 'weights' column
def repeat_rows_by_weight(data_set,seed=31415):

    # Ensure 'weights' column is integer, as fractional weights don't make sense for row repetition
    data_set["weights"] = data_set["weights"].astype(int)

    # Repeat rows based on the 'weights' column
    repeated_data_set = data_set.loc[data_set.index.repeat(data_set["weights"])]

    # Reset index to avoid duplicate indices
    repeated_data_set.reset_index(drop=True, inplace=True)

    repeated_data_set = repeated_data_set.sample(frac=1, random_state=seed).reset_index(drop=True)

    repeated_data_set.drop(columns="weights", inplace=True)

    return repeated_data_set