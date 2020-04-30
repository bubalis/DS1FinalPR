# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:47:45 2020

@author: benja
"""


import pymc3 as pm
import numpy as np
from pymc3.ode import DifferentialEquation
import matplotlib.pyplot as plt
import analyze_metros #
from pymc3.backends.base import merge_traces
#from scipy.integrate import odeint     # this is the numerical scheme to integrate if needed


# Create functions 

def SI_diffeq(x,t,theta):
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
    dx[1] =  theta[0]*x[0]*x[1] #- theta[1]*x[1] # dI/dt = beta*S*I - lambda*I
    #dx[2]=  theta[1]*x[1]
    return dx

# %% NEED TO COME BACK TO THIS SECTION TO ACTUALLY USE REAL DATA
# Set time range
 # I'm just using 10 steps
# Load actual data
 
 
data_sets=analyze_metros.SI_chunks('New Orleans')


print(data_sets)

print(len(data_sets))

 


#X=X/pop #code to normalize data to 1

#%%
# Create differential equation models

results=[]
for data in data_sets:
    
    X=np.array([np.array(data['s']), 
            np.array(data['i'])] 
           )
    pop=X[0][0]+X[1][0]
    time_range = np.arange(0,len(X[0]))   
    model=DifferentialEquation(
             func = SI_diffeq,   # what is the differential equation? 
             times = time_range,  # what is the time grid for numerical integration?
             n_states = 2,        # what is the dimensionality of the system?
             n_theta = 1,         # how many parameters are there?
    #         t0 = 0               # are we starting at the beginning?
             )
           
    # First, fit for model w/o detection rate
    with pm.Model() as mod:
        # Set priors
        
        beta = pm.Uniform('beta', 0, upper=.000001)  # from literature
        solution = model(y0=X.T[0], theta=[beta])
        
        # Likelihood
        # May want to come back and use Negative Binomial distribution
        # May need to adjust the prior distribution for the mean...
            # Since Poisson distribution has mean=var, can't explore var of data with Bayesian Inf
        #x_obs =pm.Lognormal('x_obs', mu=solution), sd=sigma, observed = X.T)
        x_obs=pm.Poisson('x_obs', mu=solution,   observed = X.T)
        # Trace and MC
        #step1 = pm.HamiltonianMC([beta,lam,tau], target_accept=0.9)
        step2=pm.Metropolis([beta])
        trace = pm.sample(
                step=[step2], 
                draws=1000, 
                tune=500, 
                cores=1,
                #chain=i
                ) 
    
        
       # for i in range(2)]
     
    beta_samples = trace.get_values('beta',burn=500,
                                    thin=5
                                    )

    
    
    plt.plot(beta_samples)
    plt.set_ylabel(r'$\beta$')
    
    plt.set_xlabel('Sample #')
    plt.show()
        
    plt.hist(beta_samples)   # I don't see a clear mean here...
    plt.set_xlabel(r'$\beta$')
    plt.show()
    
    print(f'Mean value of Beta: {np.mean(beta_samples)}')

    
    print(f'Estimate of R0:{np.mean(beta_samples)*11*pop}')
    results.append({'date range': (data['date'].min(), data['date'].max()),
                    'beta':np.mean(beta_samples), 
                    '95% CI': (np.quantile(beta_samples, .025 ), np.quantile(beta_samples, .975))
                    })
    
#%%