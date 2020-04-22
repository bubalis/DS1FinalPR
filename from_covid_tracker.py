# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:52:44 2020

@author: benja
"""

import requests
import json
import pandas as pd
from datetime import datetime, date
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from matplotlib.dates import DayLocator, DateFormatter
#%%

today=datetime.combine(date.today(), datetime.min.time())


def retrieve_data():
    '''Retrieve Data from the covid tracker website. Return it as a pandas dataframe.'''
    response=requests.get('https://covidtracking.com/api/states/daily')
    return pd.DataFrame(json.loads(response.text))

def avgGrowthrate(df, column, window=5):
    '''Return average growth rate for last (window) days'''
    return (df[column].diff(window)/(df[column]-df[column].diff(window))+1)**(1/window)-1

def doublingTime(perc_growth):
    '''Return a doubling time given a % growth rate'''
    try:
        return 1/math.log(perc_growth+1, 2)
    except (ValueError, ZeroDivisionError):
        return np.nan
                     
def parse_num(string):
    '''Read integer written as string with commas, 
    return as integer'''
    while ',' in string:
        string=string.replace(",", '')
    return int(string)
    

def read_pops():
    '''Get state population Data from csv'''
    with open('state_pops.csv') as csv_file:
        reader=csv.reader(csv_file)
        pop_dic={}
        for line in [line  for line in reader][1:]:
            pop_dic[line[0].strip('.')]=parse_num(line[1])
    return pop_dic     



def zeroToNaN(number):
    '''Convert 0 to np.nan.
    Otherwise return number.'''
    
    if number==0:
        return np.nan
    else:
        return number




def state_subset(state_abbrv, df, pop, interval=7):
    '''Grab data from 1 state and perform analyses on it.
    Return a dataframe with these analyses.'''
    
    df=df[df['state']==state_abbrv]
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df=df.sort_index()
    df.dropna(axis=1, how='all', inplace=True)
    for column in ['Cases', 'Deaths', 'Tests Performed']:
        df[f'{column} % change']=df[column].pct_change()
        df[f'{column} Rolling % change']=avgGrowthrate(df, column, interval).replace([np.inf, -np.inf], np.nan)    
        df[f'{column} Doubling time']=df[f'{column} Rolling % change'].apply(doublingTime)
    df['Positive Rate']=df['Cases']/df['Tests Performed']
    df['7day Rolling Avg Positive Rate']=df['Cases'].diff(7)/df['Tests Performed'].diff(7)
    
    df=df.replace([np.inf, -np.inf], np.nan)    
    df=df.fillna(0)
    for name in ['Cases', 'Deaths', 'Tests Performed']:
        df[f'{name} Rolling % change']=df[f'{name} Rolling % change'].apply(zeroToNaN)
        df[f'{name} Doubling time']=df[f'{name} Doubling time'].apply(zeroToNaN)
    df['Positive Rate'].apply(zeroToNaN)
    df['7day Rolling Avg Positive Rate'].apply(zeroToNaN)
    
    return df


def loadNames():
    '''Load statenames as dictionary.'''
    
    with open('state_dict.txt') as file:
        return json.loads(file.read())


def formatDate(string):
    '''Turn dates as formatted by covid Tracker into datetime objs.'''
    string=str(string)
    return datetime.strptime(string, '%Y%m%d')

def reformatdf(df):
    '''Make date columns into a datetime object and rename columns'''
    
    df['date']=df['date'].apply(formatDate)
    df= df.rename(
            columns={'total': 'Tests Performed', 
                     'positive': 'Cases', 
                     'death':'Deaths', 
                     'date': 'Date',
                     'positiveIncrease': 'New Cases', 
                     'totalTestResultsIncrease': 'New Tests Performed',
                     'deathIncrease': 'New Deaths'
                     }
                )
    return df


def getCurrentData():
    '''Get all data from covid tracker website. Make modifications and save it.'''
    
    df=retrieve_data()
    df.to_csv(os.path.join('data', 'covid_tracker_orig.csv'))
    df=reformatdf(df)
    state_dict=loadNames()
    pop_dict=read_pops()
    new_df=pd.DataFrame()
    for state in state_dict:
        pop=pop_dict[state_dict[state]]
        subset=state_subset(state,df, pop)
        new_df=new_df.append(subset)
    new_df.to_csv(os.path.join('data', 'covid_tracker_modified.csv'))


if __name__=='__main__':
    getCurrentData()
