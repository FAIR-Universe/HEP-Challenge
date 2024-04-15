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

import sys
import os
import gzip
import copy
import pandas as pd
import numpy as np
from numpy import sin, cos, cosh, sinh, sqrt, exp


def load_higgs(in_file):
    """
    Load HiggsML dataset from csv file.
    Parameters: in_file (str) - path to the csv file
    Returns: data (pandas.DataFrame) - the dataset

    """

    filename=in_file
    print ("filename=",filename)
    data = pd.read_csv(filename)
    return data

# ==================================================================================
#  V4 Class and physic computations
# ==================================================================================

class V4:

    """
    A simple 4-vector class to ease calculation, work easy peasy on numpy vector of 4 vector
    """
    px=0
    py=0
    pz=0
    e=0
    def __init__(self,apx=0., apy=0., apz=0., ae=0.):
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
            raise ValueError("Energy is too small! Energy: {}, p: {}".format(self.e, self.p()))

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

        return np.sqrt( np.abs( self.e**2 - self.p2() ) ) # abs is needed for protection
    
    def eta(self):
        """
        Compute the pseudo-rapidity
        Parameters: None
        Returns: eta (float) - pseudo-rapidity
        """

        return np.arcsinh( self.pz/self.pt() )
    
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

        return (self.phi() - v.phi() + 3*np.pi) % (2*np.pi) - np.pi
    
    def deltaEta(self,v):
        """
        Compute the pseudo-rapidity difference with another V4 object
        Parameters: v (V4) - the other V4 object
        Returns: deltaEta (float) - pseudo-rapidity difference

        """
        return self.eta()-v.eta()
    
    def deltaR(self,v):
        """
        Compute the delta R with another V4 object
        Parameters: v (V4) - the other V4 object
        Returns: deltaR (float) - delta R with another V4 object    
        """

        return np.sqrt(self.deltaPhi(v)**2+self.deltaEta(v)**2 )

    def eWithM(self,m=0.):
        """
        Compute the energy with a given mass
        Parameters: m (float) - mass
        Returns: e (float) - energy with a given mass

        """

        return np.sqrt(self.p2()+m**2)

    # FIXME this gives ugly prints with 1D-arrays
    def __str__(self):

        return "PxPyPzE( %s,%s,%s,%s)<=>PtEtaPhiM( %s,%s,%s,%s) " % (self.px, self.py,self.pz,self.e,self.pt(),self.eta(),self.phi(),self.m())

    def scale(self,factor=1.): # scale

        """Apply a simple scaling"""
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = np.abs( factor*self.e )
    
    def scaleFixedM(self,factor=1.): 
        """Scale (keeping mass unchanged)"""
        m = self.m()
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = self.eWithM(m)
    
    def setPtEtaPhiM(self, pt=0., eta=0., phi=0., m=0):
        """Re-initialize with : pt, eta, phi and m"""
        self.px = pt*np.cos(phi)
        self.py = pt*np.sin(phi)
        self.pz = pt*np.sinh(eta)
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


def w_bkg_weight_norm(data, systBkgNorm):
    """
    Apply a scaling to the weight. For W background

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    """
    # scale the weight, arbitrary but reasonable value
    data["Weight"] = (data["Weight"]*systBkgNorm).where(data["process_flags"] == 300, other=data["Weight"])
    return data


def all_bkg_weight_norm(data, systBkgNorm):
    """
    Apply a scaling to the weight.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    """
    # scale the weight, arbitrary but reasonable value
    data["Weight"] = (data["Weight"]*systBkgNorm).where(data["Label"] == 0, other=data["Weight"])
    return data

        
# ==================================================================================
# Manipulate the 4-momenta
# ==================================================================================
def mom4_manipulate (data, systTauEnergyScale, systJetEnergyScale,soft_met,seed = 31415):
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


    vmet = V4() # met 4-vector
    vmet.setPtEtaPhiM(data["PRI_met"], 0., data["PRI_met_phi"], 0.) # met mass zero,
#     met_sumet=data["PRI_met_sumet"]
    
    if systTauEnergyScale!=1.:
        # scale tau energy scale, arbitrary but reasonable value
        data["PRI_had_pt"] *= systTauEnergyScale 


        # build 4-vectors
        vtau = V4() # tau 4-vector
        vtau.setPtEtaPhiM(data["PRI_had_pt"], data["PRI_had_eta"], data["PRI_had_phi"], 0.8) # tau mass 0.8 like in original

        #vlep = V4() # lepton 4-vector
        #vlep.setPtEtaPhiM(data["PRI_lep_pt"], data["PRI_lep_eta"], data["PRI_lep_phi"], 0.) # lep mass 0 (either 0.106 or 0.0005 but info is lost)


        # fix MET according to tau pt change (minus sign since met is minus sum pt of visible particles
        vtauDeltaMinus = vtau.copy()
        vtauDeltaMinus.scaleFixedM( (1.-systTauEnergyScale)/systTauEnergyScale )
        vmet += vtauDeltaMinus
        vmet.pz = 0.
        vmet.e = vmet.eWithM(0.)

        #met_sum_et is increased if energy scale increased
        tauDeltaMinus=vtau.pt()
#         met_sumet+= (systTauEnergyScale-1)/systTauEnergyScale *tauDeltaMinus

    # scale jet energy scale, arbitrary but reasonable value

    if systJetEnergyScale!=1. :
        #data["PRI_jet_leading_pt"]    *= systJetEnergyScale
        data["PRI_jet_leading_pt"] = np.where(data["PRI_n_jets"] >0,
                                           data["PRI_jet_leading_pt"]*systJetEnergyScale,
                                           data["PRI_jet_leading_pt"])
        #data["PRI_jet_subleading_pt"] *= systJetEnergyScale
        data["PRI_jet_subleading_pt"] = np.where(data["PRI_n_jets"] >1,
                                           data["PRI_jet_subleading_pt"]*systJetEnergyScale,
                                           data["PRI_jet_subleading_pt"])

        data["PRI_jet_all_pt"] *= systJetEnergyScale 

        jet_all_pt= data["PRI_jet_all_pt"]
    
        #met_sum_et is increased if energy scale increased
#         met_sumet+= (systJetEnergyScale-1)/systJetEnergyScale *jet_all_pt
    

        # first jet if it exists
        vj1 = V4()
        vj1.setPtEtaPhiM(data["PRI_jet_leading_pt"].where( data["PRI_n_jets"] > 0, other=0 ),
                             data["PRI_jet_leading_eta"].where( data["PRI_n_jets"] > 0, other=0 ),
                             data["PRI_jet_leading_phi"].where( data["PRI_n_jets"] > 0, other=0 ),
                             0.) # zero mass
        # fix MET according to leading jet pt change
        vj1DeltaMinus = vj1.copy()
        vj1DeltaMinus.scaleFixedM( (1.-systJetEnergyScale)/systJetEnergyScale )
        vmet += vj1DeltaMinus
        vmet.pz = 0.
        vmet.e = vmet.eWithM(0.)



        # second jet if it exists
        vj2=V4()
        vj2.setPtEtaPhiM(data["PRI_jet_subleading_pt"].where( data["PRI_n_jets"] > 1, other=0 ),
                         data["PRI_jet_subleading_eta"].where( data["PRI_n_jets"] > 1, other=0 ),
                         data["PRI_jet_subleading_phi"].where( data["PRI_n_jets"] > 1, other=0 ),
                         0.) # zero mass

        # fix MET according to leading jet pt change
        vj2DeltaMinus = vj2.copy()
        vj2DeltaMinus.scaleFixedM( (1.-systJetEnergyScale)/systJetEnergyScale )
        vmet += vj2DeltaMinus
        vmet.pz = 0.
        vmet.e = vmet.eWithM(0.)
        
    #note that in principle we should also fix MET for the third jet or more but we do not have enough information

    if soft_met>0:
        # add soft met term
        # Compute the missing v4 vector
        random_state = np.random.RandomState(seed=seed)
        SIZE = data.shape[0]
        v4_soft_term = V4()
        v4_soft_term.px = random_state.normal(0, soft_met, size=SIZE)
        v4_soft_term.py = random_state.normal(0, soft_met, size=SIZE)
        v4_soft_term.pz = np.zeros(SIZE)
        v4_soft_term.e = v4_soft_term.eWithM(0.)
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
    data["PRI_jet_subleading_pt"] = data["PRI_jet_subleading_pt"].round(decimals=DECIMALS)
    data["PRI_jet_subleading_eta"] = data["PRI_jet_subleading_eta"].round(decimals=DECIMALS)
    data["PRI_jet_subleading_phi"] = data["PRI_jet_subleading_phi"].round(decimals=DECIMALS)
    data["PRI_jet_all_pt"] = data["PRI_jet_all_pt"].round(decimals=DECIMALS)

    return data


def calcul_int(data):

    #Definition of the x and y components of the hadron's momentum
    data["had_px"] = data.PRI_had_pt * cos(data.PRI_had_phi)
    data["had_py"] = data.PRI_had_pt * sin(data.PRI_had_phi)
    data["had_pz"] = data.PRI_had_pt * sinh(data.PRI_had_eta)
    data["p_had"] = data.PRI_had_pt * cosh(data.PRI_had_eta)

    #Definition of the x and y components of the lepton's momentum
    data["lep_px"] = data.PRI_lep_pt * cos(data.PRI_lep_phi)
    data["lep_py"] = data.PRI_lep_pt * sin(data.PRI_lep_phi)
    data["lep_pz"] = data.PRI_lep_pt * sinh(data.PRI_lep_eta)
    data["p_lep"] = data.PRI_lep_pt * cosh(data.PRI_lep_eta)

    #Definition of the x and y components of the neutrinos's momentum (MET)
    data["met_x"] = data.PRI_met * cos(data.PRI_met_phi)
    data["met_y"] = data.PRI_met * sin(data.PRI_met_phi)

    #Definition of the x and y components of the leading jet's momentum
    data["jet_leading_px"] = data.PRI_jet_leading_pt * cos(data.PRI_jet_leading_phi) * (data.PRI_n_jets >= 1)  # = 0 if PRI_n_jets == 0
    data["jet_leading_py"] = data.PRI_jet_leading_pt * sin(data.PRI_jet_leading_phi) * (data.PRI_n_jets >= 1)
    data["jet_leading_pz"] = data.PRI_jet_leading_pt * sinh(data.PRI_jet_leading_eta) * (data.PRI_n_jets >= 1)
    data["p_jet_leading"] = data.PRI_jet_leading_pt * cosh(data.PRI_jet_leading_eta) * (data.PRI_n_jets >= 1)

    #Definition of the x and y components of the subleading jet's momentum
    data["jet_subleading_px"] = data.PRI_jet_subleading_pt * cos(data.PRI_jet_subleading_phi) * (data.PRI_n_jets >= 2)  # = 0 if PRI_n_jets <= 1
    data["jet_subleading_py"] = data.PRI_jet_subleading_pt * sin(data.PRI_jet_subleading_phi) * (data.PRI_n_jets >= 2)
    data["jet_subleading_pz"] = data.PRI_jet_subleading_pt * sinh(data.PRI_jet_subleading_eta) * (data.PRI_n_jets >= 2)
    data["p_jet_subleading"] = data.PRI_jet_subleading_pt * cosh(data.PRI_jet_subleading_eta) * (data.PRI_n_jets >= 2)

    return data
#calcul_int(data)
#display(data.head(10))


def f_DER_mass_transverse_met_lep(data):
    """
    Calculate the transverse mass between the MET and the lepton
    Parameters: data (dataframe)
    """
    data["calcul_int"] = (data.PRI_met+data.PRI_lep_pt)**2-(data.met_x+data.lep_px)**2-(data.met_y+data.lep_py)**2
    data["DER_mass_transverse_met_lep"] = sqrt(data.calcul_int * (data.calcul_int >= 0))
    del data["calcul_int"]
    return data

def f_DER_mass_vis(data):
    """
    Calculate the invariant mass of the hadron and the lepton
    Parameters: data (dataframe)
    """

    data["DER_mass_vis"]=sqrt((data.p_lep+data.p_had)**2-(data.lep_px+data.had_px)**2-(data.lep_py+data.had_py)**2-(data.lep_pz+data.had_pz)**2)
    return data

def f_DER_pt_h(data):
    """
    Calculate the transverse momentum of the hadronic system
    Parameters: data (dataframe)
    """

    data["DER_pt_h"] = sqrt((data.had_px + data.lep_px + data.met_x)**2 
                          + (data.had_py + data.lep_py + data.met_y)**2)
    return data

def f_DER_deltaeta_jet_jet(data):
    """
    Calculate the absolute value of the difference of the pseudorapidity of the two jets
    Parameters: data (dataframe)
    """


    data["DER_deltaeta_jet_jet"]= abs(data.PRI_jet_subleading_eta-data.PRI_jet_leading_eta)*(data.PRI_n_jets>=2)-7*(data.PRI_n_jets<2)
    return data

from numpy import sqrt
#undefined if PRI_n_jets <= 1:
def f_DER_mass_jet_jet(data):
    """
    Calculate the invariant mass of the two jets
    Parameters: data (dataframe)
    """

    data["calcul_int"] = ((data.p_jet_leading + data.p_jet_subleading)**2 
                      - (data.jet_leading_px + data.jet_subleading_px)**2 
                      - (data.jet_leading_py + data.jet_subleading_py)**2
                      - (data.jet_leading_pz + data.jet_subleading_pz)**2)
    data["DER_mass_jet_jet"] = (sqrt(data.calcul_int*(data.calcul_int>=0))*(data.PRI_n_jets >= 2)
                            - 7*(data.PRI_n_jets <= 1))

    del data["calcul_int"]
    return data


def f_DER_prodeta_jet_jet(data):
    """
    Calculate the product of the pseudorapidities of the two jets
    Parameters: data (dataframe)
    """

    data["DER_prodeta_jet_jet"] = data.PRI_jet_leading_eta*data.PRI_jet_subleading_eta*(data.PRI_n_jets >= 2) -7*(data.PRI_n_jets<=1)
    return data


def f_DER_deltar_had_lep(data):
    data["difference2_eta"]=(data.PRI_lep_eta-data.PRI_had_eta)**2
    data["difference2_phi"]=  (np.abs(np.mod(data.PRI_lep_phi-data.PRI_had_phi+3*np.pi,2*np.pi)-np.pi))**2    
    data["DER_deltar_had_lep"] = sqrt(data.difference2_eta + data.difference2_phi)

    del data["difference2_eta"]
    del data["difference2_phi"]
    return data


def f_DER_pt_tot(data):
    """
    Calculate the total transverse momentum
    Parameters: data (dataframe)
    """
    data["DER_pt_tot"] = sqrt((data.had_px + data.lep_px + data.met_x + data.jet_leading_px + data.jet_subleading_px)**2 
                          + (data.had_py + data.lep_py + data.met_y + data.jet_leading_py + data.jet_subleading_py)**2)
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
    def A(met,lep,had):
        return sin(met-lep) * np.sign(sin(had-lep))

    def B(met,lep,had):
        return sin(had-met) * np.sign(sin(had-lep))

    data["A"]  = A(data.PRI_met_phi, data.PRI_lep_phi, data.PRI_had_phi)
    data["B"]  = B(data.PRI_met_phi, data.PRI_lep_phi, data.PRI_had_phi)
    data["num"] = (data.A + data.B)
    data["denum"] = sqrt(data.A**2 + data.B**2)

    data["DER_met_phi_centrality"] = data.num/(data.denum + (data.denum == 0))*(data.denum != 0) - 7*(data.denum == 0)
    epsilon = 0.0001
    mask = data.denum == 0
        
    data.loc[mask,"A"]  = A(data.PRI_met_phi, data.PRI_lep_phi + epsilon, data.PRI_had_phi)
    data.loc[mask,"B"]  = B(data.PRI_met_phi, data.PRI_lep_phi + epsilon, data.PRI_had_phi)
    data.loc[mask,"num"] = (data.A + data.B)
    data.loc[mask,"denum"] = sqrt(data.A**2 + data.B**2)
    data.loc[mask,"DER_met_phi_centrality"] = data.num/(data.denum + (data.denum == 0))*(data.denum != 0) - 7*(data.denum == 0)

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

    data["difference"] = (data.PRI_jet_leading_eta - data.PRI_jet_subleading_eta)**2
    data["moyenne"] = (data.PRI_jet_leading_eta + data.PRI_jet_subleading_eta)/2

    data["DER_lep_eta_centrality"]=(exp(-4/(data.difference)*((data.PRI_lep_eta - data.moyenne)**2))*(data.PRI_n_jets >= 2)
                                      -7*(data.PRI_n_jets <= 1))

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

def postprocess(data):

    data = data.drop(data[data.PRI_had_pt<26].index)
    data = data.drop(data[data.PRI_lep_pt<20].index)
    # data = data.drop(data[data.PRI_met>70].index)

    return data


# ==================================================================================
#  MAIN : here is defined the behaviour of this module as a main script
# ==================================================================================
class Systematics:

    def __init__(
        self,
        data=None,
        tes=1.0,
        jes=1.0,
        soft_met=1.0,
        seed=31415,
        w_scale=None,
        bkg_scale=None,
        verbose=0

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

        self.data = data
        self.tes = tes
        self.jes = jes
        self.soft_met = soft_met
        self.w_scale = w_scale
        self.bkg_scale = bkg_scale

        self.columns = [
            "EventId",
            "PRI_had_pt",
            "PRI_had_eta",
            "PRI_had_phi",
            "PRI_lep_pt",
            "PRI_lep_eta",
            "PRI_lep_phi",
            "PRI_met",
            "PRI_met_phi",
            "PRI_met_sumet",
            "PRI_n_jets",
            "PRI_jet_leading_pt",
            "PRI_jet_leading_eta",
            "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt",
            "PRI_jet_subleading_eta",
            "PRI_jet_subleading_phi",
            "PRI_jet_all_pt",
            "weights",
            "Label",
            "detailLabel",
        ] 

        if self.w_scale is not None:
            print("W bkg weight rescaling :", self.w_scale)
            self.data = w_bkg_weight_norm(self.data, self.w_scale)

        if self.bkg_scale is not None:
            print("All bkg weight rescaling :", self.bkg_scale)
            self.data = all_bkg_weight_norm(self.data, self.bkg_scale)
        if verbose > 0:
            print("Tau energy rescaling :", self.tes)
        self.data = mom4_manipulate(data=self.data,systTauEnergyScale = self.tes,systJetEnergyScale = self.jes,soft_met = self.soft_met,seed=seed)
        self.data = postprocess(self.data)
        self.data = DER_data(self.data)
