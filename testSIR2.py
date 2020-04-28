#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:03:18 2020

@author: bkotzen
Using testSIR_BI as inspiration, trying to code up MCMC for SIR on my own.
"""
import pymc3 as pm
import numpy as np
from pymc3.ode import DifferentialEquation
import matplotlib.pyplot as plt
import analyze_metros #
#from scipy.integrate import odeint     # this is the numerical scheme to integrate if needed


# Create functions 
def SIR_diffeq(x,t,theta):
    """ Code up SIR as a differential equation. 
            INPUTS:
                x - 2D vector with compartments [S,I] at a single time
                t - time (unused, but needed for integration later)
                theta - parameters [beta,lambda] where beta is transition prob. to I, lambda is recovery prob
            OUTPUTS:
                dx - 2D vector of derivative at current time
    """
    dx=[None,None]
    dx[0] = -theta[0]*x[0]*x[1]                   # dS/dt = -beta*S*I
    dx[1] =  theta[0]*x[0]*x[1] - theta[1]*x[1]   # dI/dt = beta*S*I - lambda*I
    return dx

def SIR_wDR_diffeq(x,t,theta):
    """ Incorporating detection rate, code up SIR as a differential equation.
        Detection rate is the percent of individuals in I that are detected in the data.
            INPUTS:
                x - 2D vector with compartments [S,I] at a single time
                t - time (unused, but needed for integration later)
                theta - parameters [beta,lambda,dr] where 
                    beta is transition prob. to I
                    lambda is recovery prob
                    dr is detection rate
            OUTPUTS:
                dx - 2D vector of derivative at current time
    """
    dx=[None,None]
    S = x[0]
    I = x[1]/theta[2]    # incorporate detection rate to our estimate of I
    dx[0] = -theta[0]*S*I                # dS/dt = -beta*S*I
    dx[1] =  theta[0]*S*I - theta[1]*I   # dI/dt = beta*S*I - lambda*I
    return dx
# %% NEED TO COME BACK TO THIS SECTION TO ACTUALLY USE REAL DATA
# Set time range
time_range = np.arange(1,11)     # I'm just using 10 steps
# Load actual data
data=analyze_metros.load_MSA('New York')

#X= [np.array([999,997,996,994,993,992,990,989,986,984]),
 #   np.array([1,2,5,6,7,8,9,11,13,15])]     # Fictional data, S and I over time
 
X=[np.array(data['s']), np.array(data['i'])]
print (X)
#%%
# Create differential equation models
    
model1 = DifferentialEquation(
         func = SIR_diffeq,   # what is the differential equation? 
         times = time_range,  # what is the time grid for numerical integration?
         n_states = 2,        # what is the dimensionality of the system?
         n_theta = 2,         # how many parameters are there?
#         t0 = 0               # are we starting at the beginning?
         )

model2 = DifferentialEquation(
         func = SIR_wDR_diffeq,   # what is the differential equation? 
         times = time_range,      # what is the time grid for numerical integration?
         n_states = 2,            # what is the dimensionality of the system?
         n_theta = 3,             # how many parameters are there?
#         t0 = 0                   # are we starting at the beginning?
         )

# First, fit for model w/o detection rate
mod = pm.Model()
with mod:
    # Set priors
    beta = pm.Uniform('beta', lower=1E-10, upper=1E-5)  # from literature
    lam = pm.Uniform('lam', lower=0.01, upper = 1)      # from literature
    tau = pm.Uniform('tau', lower=1E-8, upper=100)      # from literature... 1/tau is var = noise in data
    
    # Implement numerical integration
    solution = model1(y0=[X[0][0],X[1][0]], theta=[beta,lam])
    
    # Likelihood
    # May want to come back and use Negative Binomial distribution
    # May need to adjust the prior distribution for the mean...
        # Since Poisson distribution has mean=var, can't explore var of data with Bayesian Inf
    x_obs = pm.Poisson('x_obs', mu=tau, observed = X)
    
    # Trace and MC
    step1 = pm.HamiltonianMC([beta,lam,tau], target_accept=0.9)
    trace = pm.sample(step=[step1], draws=5000, tune=1000, chains=2)
    
beta_samples = trace.get_values('beta',burn=400,thin=5)
lam_samples = trace.get_values('lam',burn=400,thin=5)
tau_samples = trace.get_values('tau',burn=400,thin=5)

fig, axes = plt.subplots(3,1,sharex=True)
axes[0].plot(beta_samples)
axes[0].set_ylabel(r'$\beta$')
axes[1].plot(lam_samples)
axes[1].set_ylabel(r'$\lambda$')
axes[2].plot(tau_samples)
axes[2].set_ylabel(r'$\tau$')
axes[2].set_xlabel('Sample #')
    
fig, axes = plt.subplots(3,1)
axes[0].hist(beta_samples)   # I don't see a clear mean here...
axes[0].set_xlabel(r'$\beta$')
axes[1].hist(lam_samples)   # I don't see a clear mean here...
axes[1].set_xlabel(r'$\lambda$')
axes[2].hist(tau_samples)   # tau is stacked up at 100... I think this is because our particular data has a mean very close to 100
axes[2].set_xlabel(r'$\tau$')

# Now, fit for model w/ detection rate
mod = pm.Model()
with mod:
    # Set priors
    beta = pm.Uniform('beta', lower=1E-10, upper=1E-5)  # from literature
    lam = pm.Uniform('lam', lower=0.01, upper = 1)      # from literature
    dr = pm.Uniform('dr', lower = 0, upper = 1)   # I think this should be between 0 and 1
    tau = pm.Uniform('tau', lower=1E-8, upper=100)      # from literature... 1/tau is var = noise in data
    
    # Implement numerical integration
    solution = model2(y0=[X[0][0],X[1][0]], theta=[beta,lam, dr])
    
    # Likelihood
    # May want to come back and use Negative Binomial distribution
    # May need to adjust the prior distribution for the mean...
        # Since Poisson distribution has mean=var, can't explore var of data with Bayesian Inf
    x_obs = pm.Poisson('x_obs', mu=tau, observed = X)   # note: mu MUST be a pm distribution
    
    # Trace and MC
    step1 = pm.HamiltonianMC([beta,lam,tau], target_accept=0.9)
    trace = pm.sample(step=[step1], draws=5000, tune=1000, chains=2)

beta_samples = trace.get_values('beta',burn=400,thin=5)
lam_samples = trace.get_values('lam',burn=400,thin=5)
dr_samples = trace.get_values('dr',burn=400,thin=5)
tau_samples = trace.get_values('tau',burn=400,thin=5)

fig, axes = plt.subplots(4,1,sharex=True)
axes[0].plot(beta_samples)
axes[0].set_ylabel(r'$\beta$')
axes[1].plot(lam_samples)
axes[1].set_ylabel(r'$\lambda$')
axes[2].plot(dr_samples)
axes[2].set_ylabel('dr')
axes[3].plot(tau_samples)
axes[3].set_ylabel(r'$\tau$')
axes[3].set_xlabel('Sample #')
    
fig, axes = plt.subplots(4,1)
axes[0].hist(beta_samples)   # I don't see a clear mean here...
axes[0].set_xlabel(r'$\beta$')
axes[1].hist(lam_samples)   # I don't see a clear mean here...
axes[1].set_xlabel(r'$\lambda$')
axes[2].hist(dr_samples)   # I don't see a clear mean here...
axes[2].set_xlabel('dr')
axes[3].hist(tau_samples)   # tau is stacked up at 100... I think this is because our particular data has a mean very close to 100
axes[3].set_xlabel(r'$\tau$')


#%% Notes
"""
I don't see strong findings in the beta / lambda / dr traces. This may be
because beta and lambda are linked (nonidentifiability), and maybe dr is also 
nonidentifiable, but in testSIR_BI, there are good results for beta and lambda. 

Additionally, I don't know that I have the likelihood set up right. Some 
literature suggests using a Poisson posterior, others suggest a Negative 
Binomial. Either way, I'm not sure that I have this implemented correctly, and 
I am certainly having trouble using it to estimate uncertainty in the data 
(this is impossible to do independent of finding the mean for the Poisson 
distribution).

Literature also suggests using the Affine-Invariant MCMC, though I believe this
requires an external package (emceev3 or something like this). I don't know 
what parameters I should be using for burn, tune, chains, etc. I played a 
little with draws and found that 5000 gave me results fast without taking too
much time.

As far as further tuning goes, it may be worthwhile to let the Bayesian 
Inference also detect the initial populations in each compartment (this is done
in the literature). 

If we have success finding beta, lambda, and dr from real-world data, we may 
also want to start using Bayesian Inference to find a time at which these 
parameters are shifting. Such a temporal analysis though would require that we 
also interpret and explain this time shift. This would require delving back 
into the time series data, looking at implementation of stay-at-home orders, 
etc., and I'm not sure we will have the time or resources to finish all of that
and summarize and present by May 1.


CONCLUDING REMARKS:
(ordered by priority)

(1) Figure out which distribution observations are sampled from, and figure out
parameters (mean, sd, etc.) accordingly
(2) Start using real data
(3) Understand why there is no clear picture for beta, lambda, or dr, and fix
the problem if possible
(4) Implement finishing touches (Affine-Invariant MCMC, initial conditions, 
temporal analysis, etc.)
"""
