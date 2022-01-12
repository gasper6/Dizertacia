# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:41:27 2022

@author: jan
"""

import os  # for saving figures
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#########
# Setup #
#########

save_figures = True
save_path = "./SIRRS results/Fig1/"

S_init = 9999
I_init = 1
R_init = 0


R0 = 3 # basic reproduction number
M = 2  # mean time of protective immunity
SD = 2   # standard deviation of protective immunity
infectious_period = 10/365 # mean time of infectious period
β_multiplier = 0 # How much is β_ bigger than β

t_max = 10  # stopping time

N_eval = 10000  # fineness of the time resolution


# calculation of model parameters
γ = 1/infectious_period
β = R0 * γ
β_ = β * β_multiplier
V = SD*SD
k = round(M*M/V)
ω = k / M


def SIRRS(t, y, β, γ, ω, β_=None):
    if β_ is None:
        β_ = β

    S = y[0]
    I = y[1]
    R = y[2:]
    N = y.sum()
    
    # k = len(R)
    
    dS = -β*S*I/N + ω*R[-1]
    dI = β*S*I/N - γ*I
    # dR = np.zeros(k)
    
    # We add fictive transition R_1 → R_1 to simplify calculations
    stage_progression = np.convolve(R, (-ω, ω))[:-1]
    boosting = β_ * R*I/N
    
    dR = stage_progression - boosting
    dR[0] += boosting.sum() + γ*I
    
    return dS, dI, *dR






######################
# Numerical solution #
######################

# Setting initial conditon and timespan

y0 = np.zeros(k+2)
y0[0] = S_init
y0[1] = I_init
y0[2] = R_init

t_eval = np.arange(N_eval+1) * (t_max/N_eval)


# Solving
sol = solve_ivp(SIRRS, (0, t_max), y0, method="RK45", args=(β, γ, ω, β_),
                t_eval=t_eval, dense_output=True)

S = sol.y[0]
I = sol.y[1]
R = sol.y[2:].sum(0)
t = sol.t



############
# Plotting #
############

plt.rcParams.update({'font.size': 14,
                     'font.family': 'serif',
                     'figure.figsize': (9, 5),
                     'text.usetex' : True
                     })


if not os.path.isdir(save_path):
    os.makedirs(save_path)

plt.figure()
plt.plot(t, S, "b", label="Susceptible")
plt.plot(t, I, "r", label="Infectious")
plt.plot(t, R, "g", label="Recovered")
plt.title("All comaprtments")
plt.grid()
plt.legend()
plt.tight_layout()
if save_figures:
    plt.savefig(save_path+"All.png")


plt.figure()
plt.plot(t, S, "b", label="Susceptible")
plt.title("Susceptible")
plt.grid()
plt.tight_layout()
if save_figures:
    plt.savefig(save_path+"S.png")

plt.figure()
plt.plot(t, I, "r", label="Infectious")
plt.title("Infectious")
plt.grid()
plt.tight_layout()
if save_figures:
    plt.savefig(save_path+"I.png")


plt.figure()
if k > 1:
    plt.plot(t, sol.y[2].T, "k", alpha=0.1, label="$R_1 \\dots R_{%d}$"%k)
    plt.plot(t, sol.y[3:].T, "k", alpha=0.1)
    plt.legend()
plt.plot(t, R, "g", label="Recovered")
plt.title("Recovered")
plt.grid()
plt.tight_layout()
if save_figures:
    plt.savefig(save_path+"R.png")

if save_figures:
    if os.path.exists(save_path + "Parameters.txt"):
        os.remove(save_path + "Parameters.txt")
    with open(save_path + "Parameters.txt", "x") as f:
        f.write("Initial = %f, %f, %f\n"%(S_init, I_init, R_init))
        f.write("R0 = %.2g\n"%R0)
        f.write("M = %.2g\n"%M)
        f.write("SD = %.2g\n"%(k**.5 * ω))
        f.write("infectious_period = %.2g\n"%infectious_period)
        f.write("gamma = %.4g\n"%γ)
        f.write("beta = %.4g\n"%β)
        f.write("beta_tilde = %.4g\n"%β_)
        f.write("k = %d\n"%k)
        f.write("omega = %.4g\n"%ω)

plt.show()


# Structure of the recovered compartment
#plt.figure()
#plt.plot(t, sol.y[2:].T)

