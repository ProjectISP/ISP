#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:41:30 2019

@author: robertocabieces
"""

from obspy.io.xseed import Parser
from os import listdir
import os
from os.path import isfile, join
import pandas as pd
path=os.getcwd()
stalat0=[]
stalon0=[]
staelev=[]
stacall=[]
pathdataless=path+"/dataless"
outstations_path=path+"/stations"
dataless = [f for f in listdir(pathdataless) if isfile(join(pathdataless, f))]
dataless.sort()

for f in dataless:
    parser = Parser(pathdataless + "/"+ f) 
    blk = parser.blockettes
    try: 
        print(f)                      
#       coord=parser.get_coordinates(seed_id="SHZ", datetime=start1)
#        #coordinates=[coord["longitude"],coord["latitude"]]
#        stalat0.append(coordinates[1])
#        stalon0.append(coordinates[0])  
        stacall.append(blk[50][0].station_call_letters)
        stalat0.append(blk[50][0].latitude)
        stalon0.append(blk[50][0].longitude)
        staelev.append((blk[50][0].elevation)/1000)

    except:        
        pass

df = pd.DataFrame({'Code':'GTSRCE','Name':stacall,'Type':'LATLON','Lon':stalon0,'Lat':stalat0,'Z':'0.000','Depth':staelev},columns=['Code','Name','Type','Lat','Lon','Z','Depth'])
df.to_csv(outstations_path+'/stations.txt',sep=' ',header=False,index = False)
