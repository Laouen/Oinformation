import numpy as np
import pandas as pd

from npeet import entropy_estimators as ee
from Oinfo.oinfo_gc import nplet_tc_dtc

import sys
import os

GCMI_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libraries')
sys.path.append(GCMI_dir)

from gcmi.python import gcmi


def npeet_entropy(X:np.ndarray):
    return ee.entropy(X, base=np.e)

def gcmi_entropy(X:np.ndarray):
    return gcmi.ent_g(gcmi.copnorm(X.T))

def o_information_thoi(X:np.ndarray):
    return nplet_tc_dtc(X)[2]


def generate_flat_system(alpha: float=1.0, beta: float=1.0, gamma: float=1.0, T: int=1000):

    assert 0 <= alpha <= 1.0, 'alpha must be in range [0,1]'
    assert 0 <= beta <= 1.0, 'beta must be in range [0,1]'
    assert 0 <= gamma <= 1.0, 'gamma must be in range [0,1]'

    # Generate random samples
    Z00, Z01, Z1, Z2, Z3, Z4, Z5, Z6 = [np.random.normal(0, 1, T) for _ in range(8)]

    # Define the variables
    X1 = alpha*np.log(np.abs(Z00) + 1) + beta*Z01 + gamma*Z1
    X2 = alpha*Z00                     + beta*Z01 + gamma*Z2
    X3 = alpha*np.power(Z00,2)         + beta*Z01 + gamma*Z3
    X4 = alpha*np.exp(Z00)             + beta*Z01 + gamma*Z4
    X5 = alpha*np.sin(Z00)             + beta*Z01 + gamma*Z5
    X6 = alpha*np.cos(Z00)             + beta*Z01 + gamma*Z6

    return pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6,
        'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'Z4': Z4, 'Z5': Z5, 'Z6': Z6,
        'Z00': Z00, 'Z01': Z01
    })

def ReLU(X, cutoff=0):
    return np.maximum(X,cutoff)

def generate_relu_sistem(alpha: float=1.0, beta: float=1.0, pow_factor: float=0.5, T: float=10000):

    assert 0 <= alpha <= 1.0, 'alpha must be in range [0,1]'
    assert 0 <= beta <= 1.0, 'beta must be in range [0,1]'

    Z_syn, Z_red = np.random.normal(0, 1, (2,T))

    X1 = alpha*np.power(ReLU(Z_syn), pow_factor)    + beta*Z_red
    X2 = -alpha*np.power(ReLU(-Z_syn), pow_factor)  + beta*Z_red

    return pd.DataFrame({'X1': X1, 'X2': X2, 'Z_syn': Z_syn, 'Z_red': Z_red})

def generate_continuos_xor(alpha: float=1.0, T: int=10000):
    X1, X2, Z = np.random.normal(0, 1, (3, T))

    X1_XOR_X2 = np.logical_xor(X1 > 0, X2 > 0).astype(int)
    Zxor = alpha*(Z + 4*X1_XOR_X2) + (1-alpha)*Z

    return pd.DataFrame({'X1': X1, 'X2': X2, 'Zxor': Zxor})