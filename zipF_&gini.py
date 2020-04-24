# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:40:26 2020

@author: benja
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import data_mgmt
import from_covid_tracker
import os
#%%



def pop_lorenz(s_df, valcol, popcol, sort_col, **kwargs):
    '''Plot the cumulative distribution function of a variable.
    s_df: the subseted dataframe.
    valcol: the variable to plot.
    popcol: the population column.
    column to sort on (should be valcol per capita.
    pass any plotting **kwargs desired to plot function.'''
    ax=plt.gca()
    s_df=s_df.dropna(subset=[valcol]).sort_values(sort_col)
    Y=(s_df[valcol].cumsum()/s_df[valcol].sum()).to_numpy()
    pops=s_df[popcol].tolist()
    N=s_df[popcol].sum()
    X=np.array([np.sum(pops[:i+1])/N for i in range(len(Y))])
    Xdiff=np.insert([X[i]-X[i-1] for i in range(1, len(X))], 0, X[0])
    print(round(np.sum((X*Xdiff)-(Y*Xdiff))/.5, 2))
    ax.plot(X, Y, '-', **kwargs)


def CDFpop(s_df, valcol, popcol, **kwargs):
    '''Plot the population-adjusted cumulative distribution function of a variable.'''
    ax=plt.gca()
    s_df=s_df.sort_values(valcol).dropna(subset=[valcol])
    X=s_df[valcol].tolist()
    X=X/np.max(X)
    pops=s_df[popcol].tolist()
    N=s_df[popcol].sum()
    Y=[np.sum(pops[:i+1])/N for i in range(len(X))]
    ax.plot(X, Y, '-', **kwargs)
    

def all_CDF_pop(df, states, savename='counties_CDF.png'):
    '''Make a multiple-CDF of the pop-adjusted CDF.'''
    fig=plt.figure()
    for state in states: 
        sdf=df[df['state']==state]
        CDFpop(sdf, 'cases_per_cap', 'population')
    CDFpop(df, 'cases_per_cap', 'population' )
    plt.legend(states+['US'])
    plt.xlabel("Relative rate of infection: 1=max")
    plt.ylabel("% of People living in counties w/ less than")
    plt.savefig('counties.png')
    plt.show()
    
def all_lorenzpop(df, subsets, subset_col, savename='counties_by_date', subset_labels=[]):
    '''Make a population adjusted lorenz curve of data from df.
    subset_col: the column with the subset labels.
    subsets: a list of labels to extract and plot.
    If different legend lables are desired, pass them as list subset_labels.'''
    
    fig=plt.figure()
    for sub in subsets: 
        sdf=df[df[subset_col]==sub]
        pop_lorenz(sdf, 'cases', 'population', 'cases_per_cap')
    if not subset_labels:
        plt.legend(subsets)
    else:
        plt.legend(subset_labels)
    plt.plot(np.linspace(0,1, 500), np.linspace(0,1, 500), '-k' )
    plt.xlabel('Proportion of people')
    plt.ylabel('Proportion of Cases')
    plt.savefig(os.path.join('figures', '{savename}.png'))
    
    plt.show()

def state_gini_overtime(state):
    '''Show the change in the gini of infection rates over a time-series.'''
    
    subset=df[df['state_y']==state]
    start_date=subset[subset['cases']>0]['date'].min()
    days_elapsed=(subset['date'].max()-start_date).days
    interval=int(days_elapsed/6) #set interval so that 
    
    days=[start_date+datetime.timedelta(days=i) for i in range(0, days_elapsed, interval)]
    subset_labels=[datetime.datetime.strftime(day, '%m/%d') for day in days]
    all_lorenzpop(subset, subsets=days, subset_col='date', savename=f'{state}_by_date',subset_labels=subset_labels) 

def dateFormatter(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d')

if __name__=='__main__':
    states=['Massachusetts', 
              'Maine', 'South Dakota', 
              'Georgia', 'Idaho', 'California',
              'Illinois',
              'Vermont', 'New York']
    df=pd.read_csv(os.path.join('data', 'covid_by_county.csv'))
    df=df.dropna(subset=['date'])
    df['date']=df['date'].apply(dateFormatter)
    print('filling in blank data''')
    df=data_mgmt.fill_blank_dates_counties(df)
    
    all_lorenzpop(df[df['date']==df['date'].max()], states, 'state_y')   
    state_gini_overtime('Maine')  
    state_gini_overtime('North Dakota')





