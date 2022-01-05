# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:41:27 2022

@author: jan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

S_init = 9991
I_init = 1
R_init = 0


R0 = 20 # basic reproduction number
# Units in years:
M = 2  # mean time of protective immunity
SD = 0.5   # standard deviation of protective immunity
infectious_period = 10 # mean time of infectious period


γ = 365/infectious_period
β = R0 * γ
β_ = β * 1
V = SD*SD
k = round(M*M/V)
ω = k / M


t_max = 10
N_eval = 10000

def SIRRS(t, y, β, γ, ω, β_=None):
    β_ = β_ or β
    S = y[0]
    I = y[1]
    R = y[2:]
    N = y.sum()
    
    k = len(R)
    
    dS = -β*S*I/N + ω*R[-1]
    dI = β*S*I/N - γ*I
    # dR = np.zeros(k)
    
    # We add fictive transition R_1 → R_1 to simplify calculations
    stage_progression = np.convolve(R, (-ω, ω))[:-1]
    boosting = β_ * R*I/N
    
    dR = stage_progression - boosting
    dR[0] += boosting.sum() + γ*I
    
    return dS, dI, *dR

y0 = np.zeros(k+2)
y0[0] = S_init
y0[1] = I_init
y0[2] = R_init

t_eval = np.arange(N_eval+1) * (t_max/N_eval)

sol = solve_ivp(SIRRS, (0, t_max), y0, method="RK45", args=(β, γ, ω, β_),
                t_eval=t_eval, dense_output=True)

S = sol.y[0]
I = sol.y[1]
R = sol.y[2:].sum(0)
t = sol.t

plt.figure()
plt.plot(t, S, "b", label="Susceptible")
plt.plot(t, I, "r", label="Infectious")
plt.plot(t, R, "g", label="Recovered")
plt.title("")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# Structure of the recovered compartment
#plt.figure()
#plt.plot(t, sol.y[2:].T)

