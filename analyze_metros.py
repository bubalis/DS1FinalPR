# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:21:39 2020

@author: benja
"""

import pandas as pd
import os
import data_mgmt
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt



#%%

def plot_by_censusArea(df, x, y, kind, loglog=True):
    slope, intercept=data_mgmt.plot(df,  x,  y, loglog)
    plt.xlabel(f'Log {kind} {x.title()}')
    plt.ylabel(f'Log Total Confirmed {y.title()}')
    plt.annotate(f'Trendline= {round(slope, 2)}X+{round(intercept, 2)}', (9,9))
    plt.savefig(os.path.join('figures', f'{kind}_{x}_v_{y}.png'))
    plt.show()
    return slope, intercept


if __name__=='__main__':
    countyData=pd.read_csv(os.path.join('data', 'covid_by_county.csv'))
    
    for kind in ["CBSA", "CSA", "PSA"]: #for 3 statistical area designations: Core-Based, Combined and Primary
        
        df=data_mgmt.df_by_CBSA(countyData, kind)
        cur=data_mgmt.getMostCurrentRecords(df, 'date', f'{kind} Title')
        slope, intercept=plot_by_censusArea(cur, 'population', 'cases', kind, loglog=True)
        slope, intercept=plot_by_censusArea(cur, 'population', 'deaths', kind, loglog=True)