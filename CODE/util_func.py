#!/usr/bin/env python
# ==========================================================
# Author: Pietro Bergamini - pietro.bergamini@phd.unipd.it
# ==========================================================

import numpy as np
from scipy.integrate import quad
import string
import random

from math import atan

def integrandone(r, r_core, r_cut, R_new):
    return( (r_cut * atan(r / r_cut) - r_core * atan(r / r_core)) / ( (1 + r ** 2 / r_core ** 2) * (1 + r ** 2 / r_cut ** 2))) * r
def integrandtwo(r, r_core, r_cut, R_new):
    return(r_cut * atan(r / r_cut) - r_core * atan(r / r_core)) / (  r ** 2 * (1 + r ** 2 / r_core ** 2) * (1 + r ** 2 / r_cut ** 2) ) * (r**3 - (r**2 -R_new**2)**1.5)
def finteg(r_core, r_cut, R_new):
    return  (quad(integrandone, 0., R_new, epsrel=1e-4, args=(r_core, r_cut, R_new)))[0]*0.3333 +(quad(integrandtwo, R_new, np.inf, epsrel=1e-4, args=(r_core, r_cut, R_new)))[0]*0.3333
vinteg = np.vectorize(finteg)

def deprojection_lenstool(R=None, r_core=None, r_cut=None):
    """
    Computes the projection coefficient in equation C.16 of Bergamini et al. (2019) using 1D integrals (Mamon & Lokas 2005, Agnell oet al. 2014) and vectorisation.
    :param R: projected aperture radius within which cluster member stellar velocity dispersion are measured
    :param r_core: cluster member (dPIE) core radius
    :param r_cut: cluster member (dPIE) cut radius
    :return: projection coefficient in equation C.16 of Bergamini et al. 2019
    """
    R_new = np.asarray(R, dtype=float)
    r_core = np.asarray(r_core, dtype=float)
    r_cut = np.asarray(r_cut, dtype=float)

    c = (4 / np.pi) * ((r_core + r_cut) / (r_core ** 2 * r_cut)) * (
                1 / (np.sqrt(r_core ** 2 + R_new ** 2) - r_core - np.sqrt(r_cut ** 2 + R_new ** 2) + r_cut))

    tvi=vinteg(r_core, r_cut, R_new)
    projection_coeff = np.sqrt(c * tvi)

    return projection_coeff * np.sqrt(3 / 2)

def find_nearest(array, value):
    """
    Return the index of the *array entry closest to *value

    :param array: array
    :param value: single value
    :return: index of the array entry closest to *value
    """
    value_matrix = (np.ones((len(array[:][0]), 1)) * value).transpose()
    idx = np.argmin((np.abs(array - value_matrix)), axis=1)
    return idx


def scaling_func(mag, ref, slope, ref_mag):
    """
    Genaral form for cluster member scaling relations in equations 3 and 4 of Bergamini et al. 2019

    :param mag: cluster member magnitude
    :param ref: reference value (reference velocity dispersion or reference core radius or reference truncation radius) corrisponding to the reference magnitude *ref_mag
    :param slope: scaling relation slope
    :param ref_mag: scaling relation referece magnitude
    :return: scaled value (velocity dispersion or core radius or truncation radius)
    """
    return ref * 10 ** (((ref_mag - mag) * slope) / 2.5)


def randomstring(size=8, chars=string.ascii_lowercase + string.digits):
    """
    Generates a random string of *size characters

    :param size: number of characters. Default value = 8
    :param chars: Type of characters. Default letters and numbers
    :return: Random string of *size characters
    """
    return ''.join(random.choice(chars) for _ in range(size))

