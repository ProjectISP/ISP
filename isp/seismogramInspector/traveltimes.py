#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:37:28 2019

@author: robertocabieces
"""

from obspy.clients.iris import Client
import pandas as pd
import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics.base import gps2dist_azimuth
def arrivals(latevent,lonevent,depth,sta):
    
    station=pd.read_csv('stations.txt',sep='\t',index_col='Name')
    latsta=station.loc[sta].Lat
    lonsta=station.loc[sta].Lon
    
    #model = TauPyModel(model="iasp91")
    client = Client()
    #result = client.traveltime(evloc=(-36.122,-72.898),staloc=[(-33.45,-70.67),(47.61,-122.33),(35.69,139.69)],evdepth=22.9)
    result = client.traveltime(model='iasp91',evloc=(latsta,lonsta),staloc=(latevent,lonevent),evdepth=depth,traveltimeonly=False)
    #result = client.distaz(stalat=1.1, stalon=1.2, evtlat=3.2,evtlon=1.4)
    
    L=result.decode()
    
    file = open ('TravelTimes.txt', 'w') 
    file.writelines(L)
    file.close()
    
    df = pd.read_csv("TravelTimes.txt",sep='\t',skiprows=1)
    n=len(df)
    #print(result['distance'])
    #print(result['distancemeters'])
    #print(result['backazimuth'])
    #print(result['azimuth'])
    #print(result['ellipsoidname'])
    
    Phases=[]
    Arrivals=[]
    ###Example 1 choice
    for i in np.arange(2,n):
        
        K=df.loc[i]
        
        Phase=K[0][19:24]
        Phase=str(Phase)
        
        TT=K[0][27:34]
        TT=str(TT)
        TT=float(TT)
        
        Phases.append(Phase)
        Arrivals.append(TT)
        
    return [Phases,Arrivals]

def arrivals2(latevent,lonevent,depth,sta):   
    station=pd.read_csv('stations.txt',sep='\t',index_col='Name')
    latsta=station.loc[sta].Lat
    lonsta=station.loc[sta].Lon
    
    coord=gps2dist_azimuth(latevent, lonevent, latsta, lonsta, a=6378137.0, f=0.0033528106647474805)    
    dist=coord[0]/111180
    model = TauPyModel(model="iasp91")
    
    Phases=[]
    Arrivals=[]
    
    arrivals2 = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist)
    n=len(arrivals2)
    
    for i in range(n):
        
        arr = arrivals2[i]
        name=arr.name
        time=arr.time
        Phases.append(name)
        Arrivals.append(time)
        
    return [Phases,Arrivals]
    