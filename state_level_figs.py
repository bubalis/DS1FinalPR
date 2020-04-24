# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:28:08 2020

@author: benja
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
from data_mgmt import  getMostCurrentRecords



df=pd.read_csv(os.path.join('data', 'covid_tracker_modified.csv'))

swd=pd.read_csv(os.path.join('data',"state_weighted_densities.csv"))

cur=getMostCurrentRecords(df, 'Date', 'state')


cur=cur.merge(swd, left_on='state_fips', right_on='statefips')

x=np.log(cur['weighted density'])
y=np.log(cur['Cases Per Capita']*1000000)

slope, intercept, r_value, p_value, std_err=linregress(x,y)

print(r_value,'   ', p_value)

plt.scatter(x,y)
plt.plot(x, x*slope+intercept, '-k')
plt.ylabel("Log Cases Per Million People")
plt.xlabel("Log Weighted Population Density")
plt.savefig(os.path.join('figures', 'densityvcases.png'))
plt.show()

x=np.log(cur['weighted density'])
y=np.log(cur['Deaths Per Capita']*1000000)

slope, intercept, r_value, p_value, std_err=linregress(x,y)

print(r_value,'   ', p_value)

plt.scatter(x,y)
plt.plot(x, x*slope+intercept, '-k')
plt.ylabel("Log Deaths Per Million People")
plt.xlabel("Log Weighted Population Density")
plt.savefig(os.path.join('figures', 'densityvDeaths.png'))
plt.show()


x, y= cur['Cases Per Capita']*1000, cur['Deaths Per Capita']*1000

slope, intercept, r_value, p_value, std_err=linregress(x,y)

print(r_value,'   ', p_value)

plt.scatter(x,y)
plt.plot(x, x*slope+intercept, '-k')
plt.ylabel("Deaths per 1000")
plt.xlabel("Cases per Thousand")
plt.savefig(os.path.join('figures', 'deathvsCase.png'))
plt.show()


x=np.log10(cur['Positive Rate'])
y=np.log10(cur['Deaths']/cur['Cases'])

slope, intercept, r_value, p_value, std_err=linregress(x,y)

print(r_value,'   ', p_value)

plt.scatter(x,y)
plt.plot(x, x*slope+intercept, '-k')
plt.ylabel("Log Crude Infection Fatality Rate")
plt.xlabel("Log Positive Test Rate")
plt.savefig(os.path.join('figures', 'posRateVifr.png'))
plt.show()


x, y= np.log(cur['Cases Per Capita']*1000), np.log10(cur['Deaths Per Capita']*1000)

slope, intercept, r_value, p_value, std_err=linregress(x,y)

print(r_value,'   ', p_value)

plt.scatter(x,y)
plt.plot(x, x*slope+intercept, '-k')
plt.ylabel("Log Deaths per 1000")
plt.xlabel("Log Cases per Thousand")
