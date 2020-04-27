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
import matplotlib.pyplot as plt
from scipy.stats import linregress

#define state_abbrvs
with open(os.path.join('data', 'state_abbrvs.txt')) as f:
    state_abbrvs=json.loads(f.read())

#define state_fips 
with open(os.path.join('data', 'state_fips.txt')) as f:
    state_fips=json.loads(f.read())
    
def abbrv_to_fips(abbrv):
    'Give a state abbreviation.''' 
    return state_fips[state_abbrvs[abbrv]]

def NYT_fixes(df):
    '''Fix NYTimes dataset for two issues.
    NYC and Kansas City'''
    #fix issue with New York City: assign County Fips of 999 to whole city
    df['fips']=np.where((df['county']=="New York City"), 36999, df['fips']) 
    #fix issue with Kansas City assign County Fips of 999 to whole city
    df['fips']=np.where((df['county']=="Kansas City"), 29999, df['fips']) #fix issue with Kansas City
    df.dropna(subset=['fips'], inplace=True)
    df['fips']=df['fips'].astype(int)
    return df

def load_NYTCOVID():
    '''Load current county data from NYT covid as dataframe'''
    return NYT_fixes(pd.read_csv('https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv'))


def retrieve_covidTracker():
    '''Load current state-level data from Covid-Tracker Website and return as dataframe'''
    response=requests.get('https://covidtracking.com/api/states/daily')
    return pd.DataFrame(json.loads(response.text))



def censusfixes(df):
    '''Make fixes to census data to fit with anamolies in NYT data.'''
    
    nyboroughs=df[df['county_fips'].isin([36005, 36047, 36061, 36081, 36085])]
    nycpop=nyboroughs['POP'].sum()
    nyc_density=((nyboroughs['POP']*nyboroughs['DENSITY']).sum())/nycpop
    df=df.append({'state': 36, 
                  'county':999, 
                  'county_fips': 36999, 
                  "DENSITY":nyc_density, 
                  'POP': nycpop}, 
    ignore_index=True)
    #create separate county for kansas city
    df=df.append({'state': 29, 
                  'county':999, 
                  'county_fips': 29999,
                  "DENSITY": 1400,  #est from wikipedia
                  'POP': 491918 }, #est from wikipedia
    ignore_index=True)
    return df
    



def loadCensus():
    '''Load and format census data for: Population and population density from Census.
    Return as a dataframe'''
    response=requests.get('https://api.census.gov/data/2019/pep/population?get=DENSITY&POP&for=county:*')
    data=json.loads(response.text)
    df= pd.DataFrame(data[1:], columns=data[0])
    df['county_fips']=(df['state']+df['county']).astype(int) #make combined county fips code
    df['POP']=df['POP'].astype(int)
    df.dropna(subset=['DENSITY'])
    df["DENSITY"]=df["DENSITY"].astype(float)
    return censusfixes(df)
 
def getMostCurrentRecords(df, datecol, geo_col):
    '''Make a dataFrame of most current records for each geography 
    Input df: dataframe
    datecol: name of column with dates.
    geo_col: column with geography names'''
    newDF=pd.DataFrame()
    for name in df[geo_col].unique():
        subset=df[df[geo_col]==name]
        newDF=newDF.append(
                subset[subset[datecol]==subset[datecol].max()])
    return newDF

    

def makeCountyDF():
    '''Get data from NYT covid and from census. 
    Merge Data Together
    Add per capita columns.'''
    countyCOVID=load_NYTCOVID()
    countyPops = loadCensus()
    df=countyPops.merge(countyCOVID, left_on='county_fips', right_on='fips', how='left')
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
                         ['population', 'county_x', 'state_x', 'fips', 'county_y', 'state_y']}
        for date in df['date'].unique():
            that_date=df[df['date']==date]
            if county not in that_date['fips'].unique():
                to_add.append({**county_data, **{'deaths':0, 'cases':0, 'date': date, 
                                                 'cases_per_cap':0, 'deaths_per_cap':0}})
    df=df.append(to_add)
    return df

def logTransform(df,xcol, ycol):
    '''Drop 0 values and return x and y log transformed.'''
    non_0=df[(df[xcol]>0) & (df[ycol]>0)].dropna()
    return np.log(non_0[xcol]), np.log(non_0[ycol])


def plot(df, x, y, loglog=False):
    '''Plot x and y from a dataframe, excluding any 0 or NA values.
    Plot a trendline, print its slope and r_value. 
    Return slope and intercept.
    If loglog=True, plot log(x) log(y)'''
    
    if loglog:
        x, y=logTransform(df, x,y)
    else:
        df=df[(df[x]>0) & (df[y]>0)].dropna()
        x=df[x]
        y=df[y]
    
    slope, intercept, r_value, p_value, std_err=linregress(x,y)
    print(slope, r_value)
    plt.scatter(x, y)
    plt.plot(x, x*slope+intercept, '-k')
    return slope, intercept




def prepCBSAs(cbsa_df):
    def primary_code(row):
        '''Assign PSA code to a row in the df'''
        if not (str(row['CSA Code'])=='nan') or str(row['CSA Code'])=='':
            return row['CSA Code']
        else:
            return row['CBSA Code']
    
    def primary_name(row):
        '''Assign PSA Name to a row in the df'''
        if not (str(row['CSA Title'])=='nan') or str(row['CSA Title'])=='':
            return row['CSA Title']
        else:
            return row['CBSA Title']
        
        
        
    cbsa_df=cbsa_df.dropna(subset=['FIPS State Code'])
    cbsa_df['full_fips']=(cbsa_df['FIPS County Code']+cbsa_df['FIPS State Code']*1000).astype(int)
    
    #correction for NYC data: In NYT county Data, all boroughs merged into
    #one county called "New York City"
    cbsa_df=cbsa_df.append({'CBSA Code': 35620, 
                  'Metropolitan/Micropolitan Statistical Area': 'Metropolitan Statistical Area',
                 'full_fips':36999, 
                 'Central/Outlying County': 'Central', 
                 'CBSA Title': 'New York-Newark-Jersey City, NY-NJ-PA',
                 'CSA Title':   'New York-Newark, NY-NJ-CT-PA',
                 'CSA Code': 408}, ignore_index=True)
    
    #correction for KC data: all cases/deaths in any County in Kansas City MO
    #Are recorded as occuring in "Kansas City"
    #But all of these counties have areas outside of KC
    cbsa_df=cbsa_df.append({'CBSA Code': 28140, 
                  'Metropolitan/Micropolitan Statistical Area': 'Metropolitan Statistical Area',
                 'full_fips':20999, 
                 'Central/Outlying County': 'Central', 
                 'CBSA Title': 'Kansas City, MO-KS',
                  'CSA Title':   'Kansas City-Overland Park-Kansas City, MO-KS',
                 'CSA Code':312
                 }, ignore_index=True
            )
    
    #assign PSA codes and titles
    cbsa_df['PSA Code']=cbsa_df.apply(primary_code, axis=1)
    cbsa_df['PSA Title']=cbsa_df.apply(primary_name, axis=1)
    return cbsa_df


def df_by_CBSA(county_df, kind='CBSA'):
    '''Make a dataframe that has data organized by Census Bureau Statistical Areas.
    Pass the dataframe that has the county data.
    For kind: pass CBSA, CSA or PSA. 
    CBSA: Aggregate by core-based statistical area (Metropolitan or Micropolitan)
    CSA: Aggregate by Combined Statistical Area.
    PSA: Aggregate by Primary statistical Area: CSA if applicable, CBSA if not.
    '''    
    
    df=pd.read_csv(os.path.join('data', 'metro_areas.csv'), encoding='latin-1')
    df=prepCBSAs(df)
    
    groupby_column=f'{kind} Title'
        
    county_df=county_df.merge(df, left_on='fips', right_on='full_fips', how='left')
    
    #make dataframe with area totals by date
    groups=county_df.groupby([groupby_column, 'date'])
    gdf=pd.DataFrame([groups['population'].sum(), groups['cases'].sum(), groups['deaths'].sum()]).T
    gdf['cases per thousand']=gdf['cases']/gdf['population']*1000
    gdf=gdf.reset_index()
    
    return gdf

if __name__=='__main__':
    df=makeCountyDF()
    df.to_csv(os.path.join('data', 'covid_by_county.csv'))
    CBSA_df=df_by_CBSA(df)
    CBSA_df.to_csv(os.path.join('data', 'covid_by_CBSA.csv'))