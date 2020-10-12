"""
Module with scoring functions that take RDKit mol objects as input for scoring.
"""
import warnings
import os
import pandas as pd
import numpy as np
from functools import wraps
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit import DataStructs
from rdkit import RDConfig
import sys
sys.path.append(RDConfig.RDContribDir)
from SA_Score import sascorer


from functools import partial
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem import MACCSkeys


fpDict = {
    'ECFP': partial(GetMorganFingerprintAsBitVect, radius=2, nBits=1024),
    'MACCS': MACCSkeys,
    'Daylight': partial(RDKFingerprint, minPath=1, maxPath=7, fpSize=2048),
    'AtomPair': GetAtomPairFingerprintAsBitVect,
}


def check_valid_mol(func):
    """
    Decorator function that checks if a mol object is None (resulting from a non-processable SMILES string)
    :param func: the function to decorate.
    :return: The decorated function.
    """
    @wraps(func)
    def wrapper(mol, *args, **kwargs):
        if mol is not None:
            return func(mol, *args, **kwargs)
        else:
            return -100
    return wrapper

@check_valid_mol
def qed_score(mol):
    """
    Quantitative Drug Likeness (QED)
    :param mol: input mol
    :return: score
    """
    try:
        score = qed(mol)
    except :
        score = 0
    return score


@check_valid_mol
def substructure_match_score(mol, query, kind="any"):
    """
    :param mol: input molecule
    :param query: A list or a single SMARTS pattern the query is checked against.
    :param kind: "any": input should match one of the queries.  "all": should match all.
    :return: 1 if it matches, 0 if not.
    """
    if not isinstance(query, list):
        query = [query]
    if kind == "any":
        match = np.any([mol.HasSubstructMatch(sub) for sub in query])
    elif kind == "all":
        match = np.all([mol.HasSubstructMatch(sub) for sub in query])
    else:
        raise ValueError("use kind == any or all")

    if match:
        score = 1
    else:
        score = 0
    return score

@check_valid_mol
def sa_score(mol):
    """
    Synthetic acceptability score as proposed by Ertel et al..
    """
    try:
        score = sascorer.calculateScore(mol)
    except:
        score = 0
    return score

@check_valid_mol
def logp_score(mol):
    """
    crippen logP
    """
    score = Chem.Crippen.MolLogP(mol)
    return score

@check_valid_mol
def penalized_logp_score(mol, alpha=1):
    """
    penalized logP score as defined by .
    """
    score = reward_penalized_log_p(mol)
    return alpha * score

@check_valid_mol
def heavy_atom_count(mol):
    """
    Number of heavy atoms in molecule
    """
    hac = Chem.Descriptors.HeavyAtomCount(mol)
    return hac


@check_valid_mol
def molecular_weight(mol):
    """molecular weight"""
    mw = Chem.Descriptors.MolWt(mol)
    return mw


@check_valid_mol
def penalize_long_aliphatic_chains(mol, min_members):
    """
    Score that is 0 for molecules with aliphatic chains longer than min_members.
    """
    query = Chem.MolFromSmarts("[AR0]" + "~[AR0]"*(min_members - 1))
    if mol.HasSubstructMatch(query):
        score = 0
    else:
        score = 1
    return score

@check_valid_mol
def penalize_macrocycles(mol):
    """ 0 for molecules with macrocycles."""
    score = 1
    ri = mol.GetRingInfo()
    for x in ri.AtomRings():
        if len(x) > 8:
            score = 0
            break
    return score

@check_valid_mol
def logP_score(mol):

    try:
        logP = round(Descriptors.MolLogP(mol), 3)
    except Exception:
        logP = 0
    
    return logP

@check_valid_mol
def similarity_score(mol, targetFP, ftype='ECFP', **kwgrs):

    fp = fpDict[ftype](mol, **kwgrs)
    score = DataStructs.FingerprintSimilarity(fp, targetFP)
    return score


@check_valid_mol
def sa_score(mol):

    score = sascorer.calculateScore(mol)
    return score