# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:36:41 2020

@author: benja
"""


#%%import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
from data_mgmt import  getMostCurrentRecords

df=pd.read_csv(os.path.join('data', 'covid_by_county.csv'))

df.dropna(subset=['date'], inplace=True)

cur=getMostCurrentRecords(df, 'date', 'county_fips')
cur['crude_ifr']=cur['deaths']/cur['cases']
cur['cases per million']=cur['cases_per_cap']*1000000
cur['deaths per million']=cur['deaths_per_cap']*1000000


def printResults(slope,  r_value, std_err):
    print("Slope:   ", slope)
    print("R value:   ", r_value)
    print("Standard Error:   ", std_err )

def logTransform(df,xcol, ycol):
    '''Drop 0 values and return x and y log transformed.'''
    non_0=df[(df[xcol]!=0) & (df[ycol]!=0)]
    return np.log(non_0[xcol]), np.log(non_0[ycol])

def plotter(df, xcol, ycol, loglog=False, **kwargs):
    '''Plot selected data from the df, with linear Trend line.
    if loglog is true, make a log-log plot.
    Print the results of a simple linear fit. 
    Save figure'''
    if loglog:
        x, y = logTransform(df,xcol, ycol)
    else:
        x,y= df[xcol], df[ycol]
    slope, intercept, r_value, p_value, std_err=linregress(x,y)
    plt.scatter(x, y, **kwargs)
    plt.plot(x, x*slope+intercept, '-k')
    if loglog:
        xlabel, ylabel=f'log {xcol}', f'log {ycol}'
    else:
        xlabel, ylabel= xcol, ycol
    plt.xlabel(xlabel.title())
    plt.ylabel(ylabel.title())
    printResults(slope, r_value, std_err)
    if loglog:
        savename=f'loglog_{xcol}_v_{ycol}.png'
    else:
        savename=f'{xcol}_v_{ycol}.png'
    plt.savefig(os.path.join('figures', 'bycounty', savename))
    plt.show()
    
    return slope, intercept, r_value, std_err




if __name__=='__main__':
    results={}
    
    
    comparisons=[('population', 'deaths'), 
                 ('DENSITY', 'crude_ifr'),
                 ('DENSITY', 'cases per million'),
                 ('DENSITY', 'deaths per million'),
                 ('cases per million', 'deaths per million')]
                 
                 
                 
    for xcol, ycol in comparisons: 
        slope, intercept, r_value, std_err = plotter(cur, xcol, ycol,  False)
        slope, intercept, r_value, std_err = plotter(cur, xcol, ycol, True)









#%%
