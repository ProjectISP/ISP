#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:03:39 2018

@author: robertocabieces
"""

#Respuesta Array

import numpy as np
import matplotlib.pyplot as plt
from obspy.imaging.cm import obspy_sequential
from obspy.signal.array_analysis import array_transff_freqslowness


def arf(path,fmin,flim,slim):
    
    data=np.loadtxt(path,skiprows=1,usecols = (1,2,3))
    
    # Generate array coordinates
    n=len(data)
    coords=np.zeros([n,3])
    for i in range(n):
        coords[i]=data[i]
    
    
    #coords /= 1000.
    
    #Slowness units are in s/m
    
    #Frecuency units are in Hz, limit sample frecuency/2
    
    sxmin = -slim
    sxmax = slim
    symin = -slim
    symax = slim
    sstep = slim / 100.

    fstep= flim / 100.
    fmax=flim

    transff=array_transff_freqslowness(coords, slim, sstep, fmin, fmax, fstep, coordsys='lonlat')

    return transff
