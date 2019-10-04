#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:12:14 2019

@author: robertocabieces
"""
from obspy import read
from obspy.core import UTCDateTime
import obspy.signal
from obspy.signal.trigger import plot_trigger
from obspy.signal.trigger import classic_sta_lta
from obspy.io.xseed import Parser
from obspy.geodetics.base import gps2dist_azimuth
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from mtspec import mtspec
import math

def rotate(tr1,tr2,deg1,deg2,start,dt):
    pi=3.1416
    ##Read
    start=UTCDateTime(start)
    st=read(tr1,starttime=start,endtime=start+dt) 
    st+=read(tr2)
    st.detrend()
#            ##Synchronize
    maxstart = np.max([tr.stats.starttime for tr in st])
    minend =  np.min([tr.stats.endtime for tr in st])
    st.trim(maxstart, minend)
    Y=st[0].data-np.mean(st[0].data)
#    
    X=st[1].data-np.mean(st[1].data)
    
    
    for i in range(deg2+1):
        
        deg=i+deg1

        if deg < deg2:
          
            fig, axs = plt.subplots(2, 1,figsize=[20, 5])
            
            rad = deg*pi/180
            
            ##Rotate
            N = X*np.sin(rad)+Y*np.cos(rad);
            E = X*np.cos(rad)-Y*np.sin(rad);

            ##Plot
            axs[0].plot(N,linewidth=0.5,color='k')
            axs[1].plot(E,linewidth=0.5,color='k')
            axs[0].set_xlim(0, len(N))
            
            axs[1].set_xlim(0, len(E))
                
            deg=str(deg)
            axs[0].set_title("Rotated "+deg+" Degrees")
            fig.tight_layout()                           
            plt.pause(0.01)
            plt.clf()
            plt.close()
         
        if deg == deg2:
            
            fig, axs = plt.subplots(2, 1,figsize=[20, 5])
            
            rad = deg*pi/180
            ##Read
                
            ##Rotate
            N = X*np.sin(rad)+Y*np.cos(rad);
            E = X*np.cos(rad)-Y*np.sin(rad);
            ##Plot
            axs[0].plot(N,linewidth=0.5,color='k')
            axs[1].plot(E,linewidth=0.5,color='k')
            axs[0].set_xlim(0, len(N))
            
            axs[1].set_xlim(0, len(E))
                
            deg=str(deg)
            axs[0].set_title("Rotated "+deg+" Degrees")
            fig.tight_layout()                           
            plt.show()
                  
             