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
import datetime


#%%

def plot_by_censusArea(df, x, y, kind, loglog=True):
    slope, intercept=data_mgmt.plot(df,  x,  y, loglog)
    plt.xlabel(f'Log {kind} {x.title()}')
    plt.ylabel(f'Log Total Confirmed {y.title()}')
    plt.annotate(f'Trendline= {round(slope, 2)}X+{round(intercept, 2)}', (9,9))
    plt.savefig(os.path.join('figures', f'{kind}_{x}_v_{y}.png'))
    plt.show()
    return slope, intercept

def date_to_datetime(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d')

df=pd.read_csv(os.path.join('data', 'data_by_csa.csv'))
df=df.dropna(subset=['date'])
df['date']=df['date'].apply(date_to_datetime)


def load_MSA_for_SIR(name, nameCol='CSA Title', infect_time=11):
    '''Pass a string for name: partial string of Census statistical Area.
    e.g. 'New York' yields the New York-Newark, NY-NJ-CT-PA CSA 
    Returns a df with dates, and columns for s, i and r'''
    
    subset=df[df[nameCol].str.contains(name)]
    subset['s']=subset['population']-subset['cases'] #Number of people who haven't had it yet. 
    subset['r']=subset['cases'].shift(infect_time).fillna(0) #assumes that all infections last exactly the length that is passed
    subset['i']=subset['population']-subset['r']-subset['s']
    return subset[['date', 's', 'i', 'r']]

def loadMSA_by_week(name, nameCol='CSA Title'):
    '''Return a list of week-long dataframes for the given area. '''
    
    data=load_MSA_for_SIR(name)
    data_sets=[]
    start_date=data['date'].min()
    end_date=data['date'].max()
    while start_date<end_date:
        data_sets.append(data[(data['date']>=start_date) & (data['date']<(start_date+7))])
        start_date+=7
    return data_sets
    
if __name__=='__main__':
    countyData=pd.read_csv(os.path.join('data', 'covid_by_county.csv'))
    
    for kind in ["CBSA", "PSA", "CSA"]: #for 3 statistical area designations: Core-Based, Combined and Primary
        
        df=data_mgmt.df_by_CBSA(countyData, kind)
        cur=data_mgmt.getMostCurrentRecords(df, 'date', f'{kind} Title')
        slope, intercept=plot_by_censusArea(cur, 'population', 'cases', kind, loglog=True)
        slope, intercept=plot_by_censusArea(cur, 'population', 'deaths', kind, loglog=True)
    df.to_csv(os.path.join('data', 'data_by_csa.csv'))