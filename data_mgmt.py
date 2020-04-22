# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:37:39 2020

@author: benja
"""

import pandas as pd
import requests
import json
import numpy as np
import os

def load_NYTCOVID():
    '''Load current county data from NYT covid as dataframe'''
    return pd.read_csv('https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv')


def retrieve_covidTracker():
    '''Load current state-level data from Covid-Tracker Website and return as dataframe'''
    response=requests.get('https://covidtracking.com/api/states/daily')
    return pd.DataFrame(json.loads(response.text))



def loadCensus():
    '''Load and format census data for: Poulation and population density from Census.
    Return as a dataframe'''
    response=requests.get('https://api.census.gov/data/2019/pep/population?get=DENSITY&POP&for=county:*')
    data=json.loads(response.text)
    df= pd.DataFrame(data[1:], columns=data[0])
    df['county_fips']=(df['state']+df['county']).astype(int)
    return df
    
def makeCountyDF():
    '''Get data from NYT covid and from census. 
    Merge Data Together
    Add per capita columns.'''
    countyCOVID=load_NYTCOVID()
    countyPops = loadCensus()
    df=countyPops.merge(countyCOVID, left_on='county+fips', right_on='fips', how='left')
    df['date']=pd.to_datetime(df['date'])
    df=df.rename(columns={'POP': 'population'})
    df['cases_per_cap']=df['cases']/df['population']
    df['deaths_per_cap']=df['deaths']/df['population']
    return df



def fill_blank_dates_counties(df):
    '''Fill in dates where county is not present with 0s 
    for cases and deaths in absolute and per-capita terms.'''
    counties=df['fips'].unique()
    to_add=[]
    for county in counties:
        county_data={column: df[df['fips']==county][column].min() for column in 
                         ['population', 'COUNTY', 'STATE', 'fips', 'county', 'state']}
        for date in df['date'].unique():
            that_date=df[df['date']==date]
            if county not in that_date['fips'].unique():
                to_add.append({**county_data, **{'deaths':0, 'cases':0, 'date': date, 
                                                 'cases_per_cap':0, 'deaths_per_cap':0}})
    df=df.append(to_add)
    return df


if __name__=='__main__':
    df=makeCountyDF()
    df.to_csv(os.path.join('data', 'covid_by_county.csv'))