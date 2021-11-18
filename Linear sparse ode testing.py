# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:16:39 2021

@author: jan
"""

import numpy as np
from scipy.sparse import random, diags
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp

t_max = 200

N = 30000

A = random(N, N, density=2/N, format="csc")
A = A - diags(A.sum(1).A1)

x0 = np.random.rand(N)
x0 /= x0.sum()

eA = expm_multiply(A, x0, 0, t_max, num=2, endpoint=True)[-1]


def dt(t, y):
    return A @ y


def dt_log(t, y):
    return None # TODO



sols = []
"""
for m in ("RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"):
    sols.append(solve_ivp(dt, (0, t_max), x0, jac=A, method=m, t_eval=(t_max,)))
"""
  
for m in ("RK23", "RK45", "DOP853"):
    sols.append(solve_ivp(dt, (0, t_max), x0, jac=A, method=m, t_eval=(t_max,)))
    
"""
print("expm function:")
%timeit eA = expm_multiply(A, x0, 0, t_max, num=2, endpoint=True)

for m in ("RK23", "RK45", "DOP853"):
    print(m + ":")
    %timeit solve_ivp(dt, (0, t_max), x0, method=m, t_eval=(t_max,), vectorized=True)
"""