# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 12:27:14 2021

@author: jan
"""

import numpy as np
import sympy as sp
from numba import jit

sp.init_printing(use_latex=True)

p, q = -10, -25

def exact(p, q):
    return sp.log(sp.exp(p) + sp.exp(q))


@jit(nopython=True)
def naive(p, q):
    return np.log(np.exp(p) + np.exp(q))


@jit(nopython=True)
def new_formula(p:np.double, q:np.double) -> np.double:
    P, Q = max(p, q), min(p, q)
    return P + np.log(1+np.exp(Q-P))


@jit(nopython=True)
def new_formula2(p:np.double, q:np.double) -> np.double:
    return max(p,q) + np.log(1+np.exp(-abs(p-q)))


def test(p, q):
    return (sp.N(naive(p,q) - exact(p,q)), sp.N(new_formula(p,q) - exact(p, q)))


pp = -np.arange(50)
qq = pp

