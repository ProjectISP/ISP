#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 06:37:45 2017

@author: robertocabieces
"""

#F-K sin guardadp
import os, shutil, pathlib, fnmatch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import obspy
import numpy as np
from obspy.core import read
from obspy.core import UTCDateTime
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.invsim import corn_freq_2_paz
from obspy.signal.array_analysis import array_processing,dump
from obspy.geodetics.base import gps2dist_azimuth
from scipy.signal import argrelextrema as pks
from obspy.signal.array_analysis import get_geometry
from errores import *

#from array_analysisMTSPEC import array_processingMTSPEC
# Load data
def move_dir(src: str, dst: str, pattern: str = '*.npz'):
    if not os.path.isdir(dst):
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
    for f in fnmatch.filter(os.listdir(src), pattern):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))


path="/Users/robertocabieces/Documents/obs_array/260/SCTF"
pathpower="/Users/robertocabieces/Documents/obs_array/260/power"
fmin=0.1
fmax=0.15
slim=0.3
sres=0.001
win_len=24
win_frac=0.01
def FK(path,pathpower,fmin,fmax,slim,sres,win_len,win_frac):
    path=path+"/*.*"
    currentpath=os.getcwd()
    st = read(path)
    maxstart = np.max([tr.stats.starttime for tr in st])
    minend =  np.min([tr.stats.endtime for tr in st])
    st.trim(maxstart, minend)
    #dt=maxstart
    #nr = st.count() #count number of channels

    print("Reading Seismograms")
    #win_len=25.0
    #win_frac=0.01
    
    st[0].stats.coordinates = AttribDict({
        'latitude': +35.909685,
        'elevation': 0.0,
        'longitude': -010.373608})
    
    
    st[1].stats.coordinates = AttribDict({
        'latitude': +35.594688,
        'elevation': 0.0,
        'longitude': -010.554655})
    
    
    st[2].stats.coordinates = AttribDict({
        'latitude': +35.595025,
        'elevation': 0.0,
        'longitude': -010.988251})
    
    
    st[3].stats.coordinates = AttribDict({
        'latitude': +36.220166,
        'elevation': 0.0,
        'longitude': -010.988500})
    
    
    st[4].stats.coordinates = AttribDict({
        'latitude': +36.220166,
        'elevation': 0.0,
        'longitude': -010.555266})

    coord =get_geometry(st, coordsys='lonlat', return_center=True, verbose=True) 
    tr = st[0]
    delta = tr.stats.delta
    fs=1/delta
    stime=tr.stats.starttime+20
    etime=tr.stats.starttime+(4*60-20)
    print("Computing FK")
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=-1*slim, slm_x=slim, sll_y=-1*slim, slm_y=slim, sl_s=sres,
        # sliding window properties
        win_len=win_len, win_frac=win_frac,
        # frequency properties
        frqlow=0.05, frqhigh=0.1, prewhiten=0,
        # restrict output
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
        stime=stime, etime=etime, store=dump
    )

    #store=dump
    nsamp = int(win_len * fs)
    nstep = int(nsamp * win_frac)
    out = array_processing(st, **kwargs)
    
    # Plot
    print("Ploting")
    labels = ['Rel.Power', 'Semblance', 'BAZ', 'slow']
    
    xlocator = mdates.AutoDateLocator()
    fig = plt.figure()
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(4, 1, i + 1)
        ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,edgecolors='none', cmap=obspy_sequential)
        
        ax.set_ylabel(lab)
        ax.set_xlim(out[0, 0], out[-1, 0])
        ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
        ax.xaxis.set_major_locator(xlocator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))
        
    fig.suptitle('Earthquake %s' % (
        stime.strftime('%Y-%m-%d'), ))
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
    
    #plt.show()
    #fig.savefig("TimeWavelet")
    
    # Plot
    
    cmap = obspy_sequential
    
    # make output human readable, adjust backazimuth to values between 0 and 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360
    
    # choose number of fractions in plot (desirably 360 degree/N is an integer!)
    N = 36
    N2 = 300
    abins = np.arange(N + 1) * 360. / N
    sbins = np.linspace(0, 3, N2 + 1)
    
    # sum rel power in bins given by abins and sbins
    hist, baz_edges, sl_edges = \
        np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)
    
    # transform to radian
    baz_edges = np.radians(baz_edges)
    
    # add polar and colorbar axes
    fig = plt.figure(figsize=(8, 8))
    cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
    ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    
    dh = abs(sl_edges[1] - sl_edges[0])
    dw = abs(baz_edges[1] - baz_edges[0])
    
    # circle through backazimuth
    for i, row in enumerate(hist):
        bars = ax.bar(left=(i * dw) * np.ones(N2),
                      height=dh * np.ones(N2),
                      width=dw, bottom=dh * np.arange(N2),
                      color=cmap(row / hist.max()))
    
    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels(['N', 'E', 'sp', 'W'])
    
    # set slowness limits
    ax.set_ylim(0, 0.2)
    [i.set_color('white') for i in ax.get_yticklabels()]
    ColorbarBase(cax, cmap=cmap,
                 norm=Normalize(vmin=hist.min(), vmax=hist.max()))
    
    plt.show()
    #fig.savefig("PolarWavelet")
    
    T = out[:,0]
    relpower =  out[:,1]
    AZ=out[:,3]
    Slowness=out[:,4]
    
    thresh = 0.75
    idx = pks(relpower,np.greater, order=3)
    jdx = np.where((relpower[idx]> thresh))
    kdx = idx[0][jdx[0]]
    kdx = np.array(kdx)
    side = 2
    S=[]
    A=[]
    ErrorAngle=[]
    ErrorSlowness=[]
    move_dir(currentpath,pathpower)
    for n in range(len(kdx)):    
        try:
            Time = T[kdx[n]]
            Time = Time-int(Time)
            H = Time*24
            minutes = (H-int(H))*60
            seconds = (minutes - int(minutes))*60 
            Slow=np.mean(Slowness[kdx[n]-side:kdx[n]+side])
            Az=np.mean(AZ[kdx[n]-side:kdx[n]+side])
            S.append(Slow)  
            A.append(Az)        
            value =kdx[n]*nstep
            [EA,ES]=errores(pathpower,value)
            ErrorAngle.append(EA)
            ErrorSlowness.append(ES)
            Slow="%.2f" % Slow
            Az="%.2f" % Az
            relpower="%.2f" % relpower[kdx[n]]
            print("Slownes and Az Estimated ", int(H), int(minutes), int(seconds), Slow, Az, relpower)
        except:
            print("Somethin is wrong....")
    ##encontrar azimuth para rotar
    AZmaxindex=np.where(relpower == np.max(relpower))
    #AZmaxindex = T.index(max(T))
    AZmax = AZ[AZmaxindex]
    
    return (S, A, ErrorSlowness,ErrorAngle, AZmax, coord)  
        
        
        
        
        