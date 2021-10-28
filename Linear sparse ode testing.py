# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:16:39 2021

@author: jan
"""

import numpy as np
from scipy.sparse import csc_matrix, random
from scipy.sparse.linalg import expm
from scipy.integrate import solve_ivp


N = 5000

A = random(N, N, density=2/N, format="csc")
eA = expm(A)


def dt(t, y):
    return A @ y

t_max = 100

y0 = np.random.randn(N)

sols = []

for m in ("RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"):
    sols.append(solve_ivp(dt, (0, t_max), y0, jac=A, method=m, t_eval=(t_max,)))
    
