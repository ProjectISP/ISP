#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:28:38 2019

@author: robertocabieces
"""

from obspy import read_events


def getNLLinfo(pathfile):
    cat = read_events(pathfile)
    print(cat)
    event = cat[0]
    print(event)
    origin = event.origins[0]
    print(origin)
    time=origin.time
    latitude=origin.latitude
    longitude=origin.longitude
    depth=(origin.depth)/1000 ##km
    return time,latitude,longitude,depth


#time,latitude,longitude,depth=getNLLinfotime("/Users/robertocabieces/Documents/ISP/260/Locations/2015-09-17.hyp")