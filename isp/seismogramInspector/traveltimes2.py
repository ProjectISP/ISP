#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:05:49 2019

@author: robertocabieces
"""

from obspy.clients.iris import Client
import pandas as pd
import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics.base import gps2dist_azimuth

def arrivals2(latevent,lonevent,depth,sta):   
    station=pd.read_csv('stations.txt',sep='\t',index_col='Name')
    latsta=station.loc[sta].Lat
    lonsta=station.loc[sta].Lon
    
    #coord=gps2dist_azimuth(latevent, lonevent, latsta, lonsta, a=6378137.0, f=0.0033528106647474805)
    coord=gps2dist_azimuth(36.0, -5, latsta, lonsta, a=6378137.0, f=0.0033528106647474805)
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
    
    #[Phases,Time]=arrivals(38.00,-10.00,36.000,-14.000,30.0,'OBS06')   

#arrivals(36.0,-5,15,'OBS06')