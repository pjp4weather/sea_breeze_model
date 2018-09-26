#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:23:35 2018

@author: paul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import SeaBreezeModel as sbm
from scipy.optimize import curve_fit 

plt.close("all")
# =============================================================================
# Observations
# =============================================================================
data = pd.read_csv('data.csv',sep=',',header=0)

u_vel = data['u0[m/s]']
v_vel = data['v0[m/s]']
dpdx  = data['dpdx[Pa/km]']

hours = np.arange(len(u_vel))
dpdx_mean = np.mean(dpdx)

def linfit(x,a,b):
    return a*x +b

poptlin,pcovlin = curve_fit(linfit,hours, dpdx - dpdx_mean)

dpdx_synoptic = linfit(hours,poptlin[0],poptlin[1]) + dpdx_mean
dpdx_detrend = dpdx - dpdx_synoptic

#%% =============================================================================
# Estimate A and phi
# =============================================================================
plt.close("all")
plt.subplots()
plt.xlabel("time [hours]")
plt.ylabel("dpdx [Pa/km]")
plt.title("pressure gradient x-direction")
plt.plot(hours,dpdx,label="measurements")
plt.plot(hours,dpdx_detrend, label="detrended ")
plt.plot(hours,dpdx_synoptic,label="synoptic (mean + trend)")
def func(hours,A,phase):
    return A*np.cos(2*np.pi*hours/24.+ phase)

A_guess = 1.
phase_guess = 90.*2*np.pi/180.


popt,pcov = curve_fit(func,hours, dpdx_detrend, p0 = [A_guess,phase_guess]) 

A_estimate = popt[0] #1.5
phase_estimate = popt[1]- 2*np.pi 


plt.plot(hours,func(hours,A_estimate,phase_estimate),label="fit, A="+ str(round(A_estimate,2)) + ", $\phi$ ="+str(round(phase_estimate*180/np.pi,2))+"$^{\circ}$")

A_estimate = 1.8
phase_estimate = -50. /180 * np.pi
 
plt.plot(hours,func(hours,A_estimate,phase_estimate),label="choice, A="+ str(round(A_estimate,2)) + ", $\phi$ ="+str(round(phase_estimate*180/np.pi,2))+"$^{\circ}$")
plt.legend()
plt.grid()
 

#%% =============================================================================
# Model
# =============================================================================
# define integration method
dt = 100.
tmax_h = 48 
corrector_step = False
method = "rk4"

integrator = sbm.integration(dt,tmax_h,corrector_step,method)

# define model
lat = 52
A = A_estimate/1000.   # 0.0015
phase = phase_estimate  #-85./(2*np.pi)
gamma = 0.00015

hours_arr = np.linspace(0.,47.,integrator.iters,endpoint=True)
dpdx_synoptic = linfit(hours_arr,poptlin[0],poptlin[1]) + dpdx_mean

sb = sbm.seaBreezeModel(lat,A,phase,gamma,dpdx_synoptic/1000.)


# integrate the model
u_init = u_vel[0]
v_init = v_vel[0]

integrator.integrate2D(sb.f,sb.vg, u_init,v_init)

# store variabels
u_model,v_model,time_model = integrator.u,integrator.v, integrator.time    

# =============================================================================
# Plots
# =============================================================================

fig, ax = plt.subplots(1,2)
ax[0].set_title("u-velocity")
ax[0].plot(hours,u_vel,label="measurements")
ax[0].plot(time_model/3600.,u_model,label="model")
ax[0].set_xlabel('time [hours]', fontsize=14) # label along x-axes
ax[0].set_ylabel('velocity [m/s]', fontsize=14) # label along x-axes
ax[0].legend()
ax[0].grid()

ax[1].set_title("v-velocity")
ax[1].plot(hours,v_vel,label="measurements")
ax[1].plot(time_model/3600.,v_model,label="model")
ax[1].set_xlabel('time [hours]', fontsize=14) # label along x-axes
ax[1].set_ylabel('velocity [m/s]', fontsize=14) # label along x-axes
ax[1].legend()
ax[1].grid()

plt.savefig("img/task2")