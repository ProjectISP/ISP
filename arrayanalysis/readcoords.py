#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:25:28 2019

@author: robertocabieces
"""

##read coords

import pandas as pd

path="/Users/robertocabieces/Documents/obs_array/coords.txt"

df=pd.read_csv(path,sep='\t')
n=df.Name.count()

for i in range(n):
    


    Lat = df.loc[0].Lat
    Lon = df.loc[0].Lon
    
    