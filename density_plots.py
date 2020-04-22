# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:36:41 2020

@author: benja
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

df=pd.read_csv(os.path.join('data', 'covid_by_county.csv'))

df.dropna(subset=['date'], inplace=True)

cur=df[df['date']==df['date'].max()]

plt.scatter(cur['DENSITY'], cur['cases_per_cap']*1000000)
#%%