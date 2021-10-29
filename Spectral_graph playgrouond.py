# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 17:53:34 2021

@author: Janko
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

N = 3

global_structure = np.array([
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1]]) * 5

n = N * global_structure.shape[0]

G = np.kron(global_structure, np.ones((N, N)))
G += np.random.exponential(.1, (n, n))

plt.matshow(G)

L = sparse.csgraph.laplacian(G)
L = (L+L.T)/2

plt.matshow(L)

lam, S = np.linalg.eigh(L)

eps = 1e-5
compoment_count = np.sum(lam < eps)

order = np.argsort(S[:, compoment_count])
diffs = np.diff(S[order, compoment_count])
orderdiffs = np.argsort(diffs)[::-1]
