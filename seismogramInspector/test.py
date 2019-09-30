#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 00:14:03 2019

@author: robertocabieces
"""
from obspy import read
from obspy.core import UTCDateTime
#t1="2015-09-17T15:12:00.00"
#t1=UTCDateTime(t1)
#dt=240
#st=read()
#t1=st1[0].stats.starttime
#t2=t1+1200
#st2=read(path_file+"/*OBS01*",starttime=t1,endtime=t2)
#
#
#
#st=read("/Users/robertocabieces/Documents/ISP/seismogramInspector/OBS01/WM.OBS01..BHZ.D.2015.259",starttime=t1,endtime=t1+dt)
#st.plot()


st=read("/Users/robertocabieces/Documents/ISP/seismogramInspector/OBS01/WM.OBS01..BHZ.D.2015.259")
t1=st[0].stats.starttime