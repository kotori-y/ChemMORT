"""
Modeule with scoring functions that take molecular CDDD embeddings (positions of the particles in the swarm) as input.
"""
from scipy.spatial.distance import cdist
from rdkit import Chem
from functools import wraps
import joblib
import os
import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
# from tensorflow.keras.layers import Multiply, Concatenate, Add
from tensorflow.keras.models import Model

_dir = os.path.dirname(__file__)

logd_model = joblib.load(os.path.join(_dir, 'models/logD_xgb.pkl'))
ames_model = joblib.load(os.path.join(_dir, 'models/ames_xgb.pkl'))
logs_model = joblib.load(os.path.join(_dir, 'models/logs_xgb.pkl'))
hERG_model = joblib.load(os.path.join(_dir, 'models/herg_xgb.pkl'))
liverTox_model = joblib.load(os.path.join(_dir, 'models/hepatoxicity_xgb.pkl'))
LD50_model = joblib.load(os.path.join(_dir, 'models/ld50_xgb.pkl'))
caco2_model = joblib.load(os.path.join(_dir, 'models/Caco2_xgb.pkl'))
mdck_model = joblib.load(os.path.join(_dir, 'models/MDCK_xgb.pkl'))
ppb_model = joblib.load(os.path.join(_dir, 'models/PPB_xgb.pkl'))

################ DNN Model ################
# caco2_weights = os.path.join(_dir, 'models/Caco-2/Caco-2.ckpt')
# mdck_weights = os.path.join(_dir, 'models/MDCk/MDCK.ckpt')
# ppb_weights = os.path.join(_dir, 'models/PPB/ppb.ckpt')


def check_valid_smiles(swarm):
    def calculate_score(func):
        @wraps(func)
        def wrapper(swarm, *args, **kwargs):
            bo = np.array([Chem.MolFromSmiles(smi) is not None for smi in swarm.smiles])
            score = func(swarm.x, *args, **kwargs)
            score = np.where(bo, score, -100)
            return score
        return wrapper
    return calculate_score


def approximate_res(func):
    """
    Decorator function that accurate to three decimal places
    
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return round(float(func(*args, **kwargs)), 3)
    return wrapper

@check_valid_smiles(swarm=None)
def distance_score(emb, query):
    """
    Function tha calculates the distance between an input molecular embedding and a target molecular embedding.
    :param x: input molecular embedding
    :param target: target molecular embedding
    :param metric: The metric used by scipy.spatial.distance.cdist to compute the distance in space.
    :return: The distance between input and target.
    """
    score = cdist(emb, query, metric="cosine").flatten()
    return score


# reg score
# @approximate_res
@check_valid_smiles(swarm=None)
def logD_score(emb):
    """
    Measuring the logD of molecule
    
    Parameters
    ----------
    emb : numpy.ndarray, size (x,512)
        the cddd descriptor of molecule

    Returns
    -------
    logD : float
        logD score
    """
    logD = logd_model.predict(emb)
    return logD

@check_valid_smiles(swarm=None)
def ames_score(emb):
    """
    Measure the ames positive probability of molecule,
    more close to 1, more "toxic"

    Parameters
    ----------
    emb : numpy.ndarray, size (x,512)
        the cddd descriptor of molecule

    Returns
    -------
    ames : float
        ames score

    """
    ames = ames_model.predict_proba(emb)[:,1]
    return ames

@check_valid_smiles(swarm=None)
def caco2_score(emb):
    """
    """
    caco2 = caco2_model.predict(emb)
    return caco2

@check_valid_smiles(swarm=None)
def mdck_score(emb):
    """
    """
    mdck = mdck_model.predict(emb)
    return mdck

@check_valid_smiles(swarm=None)
def ppb_score(emb):
    """
    """
    ppb = ppb_model.predict(emb)
    return ppb

@check_valid_smiles(swarm=None)
def logS_score(emb):
    """
    """
    logs = logs_model.predict(emb)
    return logs

@check_valid_smiles(swarm=None)
def hERG_score(emb):
    """
    """
    herg = hERG_model.predict_proba(emb)[:,1]
    return herg

@check_valid_smiles(swarm=None)
def hepatoxicity_score(emb):
    """
    """
    liverTox = liverTox_model.predict_proba(emb)[:,1]
    return liverTox

@check_valid_smiles(swarm=None)
def LD50_score(emb):
    """
    """
    ld50 = LD50_model.predict_proba(emb)[:,1]
    return ld50