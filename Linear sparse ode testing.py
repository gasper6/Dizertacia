# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:16:39 2021

@author: jan
"""

import numpy as np
from scipy.sparse import random, diags
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp

t_max = 100

N = 1000

A = random(N, N, density=2/N, format="csc")
A = A - diags(A.sum(1).A1)

x0 = np.random.rand(N)
x0 /= x0.sum()


def dt(t, y):
    return A @ y


def dt_log(t, y):
    x = np.exp(y)
    return  (A @ x)/x


eA = expm_multiply(A, x0, 0, t_max, num=2, endpoint=True)[-1]


sols = []

# Implicit methods are exluded, they are way too slow
for m in ("RK23", "RK45", "DOP853"):
    sols.append(solve_ivp(dt, (0, t_max), x0, method=m, t_eval=(t_max,), vectorized=False, rtol=1e-12))

x0_log = np.log(x0)
x0_log = np.nan_to_num(x0_log, neginf=-1e200)

sols_log = []
  
for m in ("RK23", "RK45", "DOP853"):
    sols_log.append(solve_ivp(dt_log, (0, t_max), x0_log, method=m, t_eval=(t_max,), atol=1e-12))


if N <= 5:
    print("expm solution:")
    print(eA)
    print()
    
    print("ODE solutions:")
    print(*[s.y.ravel() for s in sols], sep="\n")
    print()
    
    print("ODE log-solutions:")
    print(*[np.exp(s.y.ravel()) for s in sols_log], sep="\n")
    print()


"""
print("expm function:")
%timeit eA = expm_multiply(A, x0, 0, t_max, num=2, endpoint=True)

print("ODE")
for m in ("RK23", "RK45", "DOP853"):
    print(m + ":")
    %timeit solve_ivp(dt, (0, t_max), x0, method=m, t_eval=(t_max,), vectorized=True, rtol=1e-12)

print("ODE log-transformed")
for m in ("RK23", "RK45", "DOP853"):
    print(m + ":")
    %timeit sols_log.append(solve_ivp(dt_log, (0, t_max), x0_log, method=m, t_eval=(t_max,), atol=1e-12))

"""