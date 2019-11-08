#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:08:07 2019

@author: robertocabieces
"""
from isp.seismogramInspector.plotpm import plotpmNEZ

"""
Created on Tue Sep 24 17:29:03 2019

@author: robertocabieces
"""
import numpy as np
import matplotlib.pyplot as plt
# import mplcursors
import os
from obspy import read
from obspy.core import UTCDateTime
from obspy.signal.polarization import flinn
from obspy.signal.polarization import polarization_analysis,particle_motion_odr
#from plotpm import *
#from autoscale import *
import matplotlib.dates as mdates

def AP(t1,dt,trz,trn,tre,fmin,fmax,tw):
    t1=UTCDateTime(t1)
    st1=read(tre,starttime=t1,endtime=t1+dt)
    st2=read(trn,starttime=t1,endtime=t1+dt)
    st3=read(trz,starttime=t1,endtime=t1+dt)
    st=st1
    st+=st2
    st+=st3
    print(st)
    fs=st[0].stats.sampling_rate

    
    st.detrend(type='demean')
    st.plot(handle=True)
    
    k=tw
    out=polarization_analysis(st, k, 1/(fs*k), fmin, fmax, t1, t1+dt, verbose=False, method='flinn', var_noise=0.0)
    
    t=out["timestamp"]
    
    
    azimuth=out["azimuth"]+180
    incident_angle=out["incidence"]
    Planarity=out["planarity"]
    rectilinearity=out["rectilinearity"]
    #TT=trace.times("matplotlib")
    N=len(t)
    
    tt=range(0,N)
    
    Z1=st[2].data
    Z1=Z1[0:N]
    N1=st[0].data
    N1=N1[0:N]
    E1=st[0].data
    E1=E1[0:N]


    TT=[]
    for i in t:
        TT.append(UTCDateTime(i))
    
    place=np.arange(0, N, int(N/4))
    
    
    
    
    
    fig, axs = plt.subplots(nrows=7, ncols=1, sharex=True, figsize=(8,8))
    
    def onclick(event):
        
        if event.dblclick:
            if event.button == 1: 
                x1, y1 = event.xdata, event.ydata            
                print(x1)            
                a=t[int(x1)]
                            
                a=UTCDateTime(a)
                b=a+k
                st4=read(tre,starttime=a,endtime=b)
                st5=read(trn,starttime=a,endtime=b)
                st6=read(trz,starttime=a,endtime=b)
                st7=st4
                st7+=st5
                st7+=st6
                st7.detrend(type='demean')
                st7.plot(handle=True)
            
            plotpmNEZ(st7[0].data,st7[1].data,st7[2].data)
    
    
    
    
    ax = axs[0]
    ax.plot(tt, Z1)
    ax.set_ylabel('Comp Z')
    mplcursors.cursor()
    
    ax = axs[1]
    ax.plot(tt, N1)
    ax.set_ylabel('Comp N')
    mplcursors.cursor()
    
    ax = axs[2]
    ax.plot(tt, E1)
    ax.set_ylabel('Comp E')
    mplcursors.cursor()
    
    
    ax = axs[3]
    ax.plot(tt, azimuth)
    ax.set_ylabel('Back.Az')
    mplcursors.cursor()
    ax = axs[4]
    ax.plot(tt,incident_angle)
    ax.set_ylabel('Inc. Angle')
    mplcursors.cursor()
    ax = axs[5]
    ax.plot(tt,Planarity)
    ax.set_ylabel('Planarity')
    mplcursors.cursor()
    ax = axs[6]
    ax.plot(tt,rectilinearity)
    ax.set_ylabel('Rectilinearity')
    fig.suptitle('Polarization analysis')
    mplcursors.cursor()
    
    
    
    plt.xticks(place,TT,rotation=30)
    
    
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.autoscale(ax, 'y', margin=0.1)
    plt.autoscale(ax, 'x', margin=0.1)
    #autoscale(ax0, 'y', margin=0.1)
    #autoscale(ax1, 'y', margin=0.1)
    #autoscale(ax2, 'y', margin=0.1)
    #autoscale()
    #plt.autoscale(enable=True, axis='both', tight=None)
    
    plt.tight_layout()
    plt.show()    
