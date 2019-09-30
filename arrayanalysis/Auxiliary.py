#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:04:02 2019

@author: robertocabieces
"""

####Functions Signal Analysis#######


##Single Plot###
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



def plot(path,t1,t2):
    obsfiles = [f for f in listdir(path) if isfile(join(path, f))]
    obsfiles.sort()
    t1=UTCDateTime(t1)
    t2=t1+t2
    path1=path+"/"+"*.*"
    st=read(path1,starttime=t1,endtime=t2,outfile="seismogram")
    st.plot()
    
    return


def get_info(path,t1,t2):
    obsfiles = [f for f in listdir(path) if isfile(join(path, f))]
    obsfiles.sort()
    t1=UTCDateTime(t1)
    t2=t1+t2
    info=""
    for f in obsfiles:        
        st1=read(path+"/"+f,starttime=t1,endtime=t2)
        tr =st1[0]
        network= tr.stats.network
        station= tr.stats.station
        channel= tr.stats.channel
        starttime = str(tr.stats.starttime)
        endtime = str(tr.stats.endtime)
        sampling_rate= str(tr.stats.sampling_rate)
        npts=str(tr.stats.npts)

        info+=network+"   "+station+"   "+channel+"    "+starttime+"         "+endtime+"    "+sampling_rate+"   "+npts+ "\n"
        
    print(info)
    return info

def get_envelope(path,t1,t2,fmin,fmax):
    obsfiles = [f for f in listdir(path) if isfile(join(path, f))]
    obsfiles.sort()
    nfilas=len(obsfiles)
    t1=UTCDateTime(t1)
    t2=t1+t2
    k=1
    for f in obsfiles:
        
        st=read(path+"/"+f,starttime=t1,endtime=t2)
        #st_filt.filter('highpass', freq=0.5, corners=3, zerophase=False)
        tr=st[0]
        samprate=tr.stats.sampling_rate
        tr.detrend()
        tr.taper(max_percentage=0.05)
        #tr.filter('highpass', freq=1, corners=3, zerophase=True)
        tr.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
        data_envelope = obspy.signal.filter.envelope(tr.data)
        data_envelope=np.array(data_envelope)    
        npts=len(data_envelope)
        t = np.arange(0, npts / samprate, 1 / samprate)
        ##Plot##
        plt.ion()
        fig=plt.subplot(nfilas,1,k)
        fig.size=(20,20)
        plt.plot(t, tr.data, color='silver',linewidth=0.25)
        plt.plot(t, data_envelope,'r-')
        plt.title(tr.stats.station)
        plt.ylabel('Amplitude [m]')
        plt.xlabel('Time [s]')        
        plt.show()
        plt.legend()
        k=1+k
    return
        
def Sta_Lta(path,t1,t2,fmin,fmax,sta,lta):
    obsfiles = [f for f in listdir(path) if isfile(join(path, f))]
    obsfiles.sort()
    t1=UTCDateTime(t1)
    t2=t1+t2
    for f in obsfiles:
        
        st=read(path+"/"+f,starttime=t1,endtime=t2)
        tr=st[0]
        tr.detrend()
        tr.taper(max_percentage=0.05)
        tr.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
        df = tr.stats.sampling_rate
        cft = classic_sta_lta(tr.data, int(sta * df), int(lta * df))
        plot_trigger(tr, cft, 10, 2)
        
def spectrum(path,t1,t2):    
    obsfiles = [f for f in listdir(path) if isfile(join(path, f))]
    obsfiles.sort()
    t1=UTCDateTime(t1)
    t2=t1+t2
    for f in obsfiles:
        st=read(path+"/"+f,starttime=t1,endtime=t2)
        tr=st[0]
        tr.detrend()
        sta=tr.stats.station
        x=len(tr.data)
        D=2**math.ceil(math.log2(x))
        #D=int(pow(2, math.ceil(math.log(x, 2))))
        if D>x:
            D=2**(math.ceil(math.log2(x)-1))
       
        data=tr.data[0:D]

        delta = tr.stats.delta
        spec, freq, jackknife_errors, _, _ = mtspec(data,delta=delta , time_bandwidth=3.5, statistics=True)
            
        spec = np.sqrt(spec) #mtspec Amplitude spectrum
        jackknife_errors = np.sqrt(jackknife_errors)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)  
        ax1.loglog(freq, spec, '0.1', linewidth=1.0,color='steelblue',label=sta)
        ax1.frequencies = freq
        ax1.spectrum = spec
        ax1.fill_between(freq, jackknife_errors[:, 0], jackknife_errors[:, 1], facecolor="0.75", alpha=0.5, edgecolor="0.5")
        ax1.set_ylim(spec.min() / 10.0, spec.max() * 100.0)
        ax1.set_xlim(freq[0], 1/(2*delta))
        plt.ylabel('Amplitude [m]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, which="both", ls="-", color='grey')
        plt.legend()
        plt.show()
        
def spectrumelement(data,delta,sta):
        spec, freq, jackknife_errors, _, _ = mtspec(data,delta=delta , time_bandwidth=3.5, statistics=True)
        spec = np.sqrt(spec) #mtspec Amplitude spectrum
        jackknife_errors = np.sqrt(jackknife_errors)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)  
        ax1.loglog(freq, spec, '0.1', linewidth=1.0,color='steelblue',label=sta)
        ax1.frequencies = freq
        ax1.spectrum = spec
        ax1.fill_between(freq, jackknife_errors[:, 0], jackknife_errors[:, 1], facecolor="0.75", alpha=0.5, edgecolor="0.5")
        ax1.set_ylim(spec.min() / 10.0, spec.max() * 100.0)
        #ax1.set_xlim(freq[0], 1/(2*delta))
        plt.ylabel('Amplitude [m]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, which="both", ls="-", color='grey')
        plt.legend()
        plt.show()    

def deconv(sismogramas_path, dataless_path, procesados_path, physical_units):
    corner_freq_minus50 = (0.005, 0.006, 15.0, 18.0)
    corner_freq_plus50 = (0.005, 0.006, 30.0, 35.0)
    # Obtenemos el listado de ficheros a procesar
    obsfiles = [f for f in listdir(sismogramas_path) if isfile(join(sismogramas_path, f))]
    obsfiles.sort()
    for f in obsfiles:
        st = read(sismogramas_path + "/" + f)
        tr = st[0]
        station = tr.stats.station
        delta = tr.stats.delta
        fs = 1 / delta
        try:
            parser = Parser(dataless_path + "/" + "dataless" + station + ".dlsv")
            if fs <= 50:
                pre_filt = corner_freq_minus50
                st.simulate(seedresp={'filename': parser, 'units': physical_units}, pre_filt=pre_filt)
                st.write(procesados_path+"/"+f, format='MSEED')
            else:
                pre_filt = corner_freq_plus50
                st.simulate(seedresp={'filename': parser, 'units': physical_units}, pre_filt=pre_filt)
                st.write(procesados_path + "/" + f, format='MSEED')
                
        except:
            print("An exeption ocurred")
    return

def plrsection(path,path2,lat,lon):
    st =read(path+"/*.*")
    number=len(st[0].data)
    cutofflength=100*20*60 ##20 min  with 100 samp/sec
    if number<cutofflength:
        n=len(st)
        maxstart = np.max([tr.stats.starttime for tr in st])
        minend =  np.min([tr.stats.endtime for tr in st])
        st.trim(maxstart, minend)
        
        data=np.loadtxt(path2,skiprows=1,usecols = (1,2,3))
        for i in range(n):
            coord=gps2dist_azimuth(data[i][1], data[i][0], lat, lon, a=6378137.0, f=0.0033528106647474805)
            dist=coord[0]
            st[i].stats.distance=dist
        st.plot(type='section',color='red',time_down=True,alpha=0.6,linewidth=0.5,station=True,grid_color='white')
    return 

def scan1(path):
    import obspy
    import subprocess as S
    #S.check_call(["obspy-scan",path])
    S.call(["obspy-scan",path])
    print("Done")
    
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
           
####Classic_sta_lta Obspy version 
def classic_sta_lta_py(a, nsta, nlta):
    """
    Computes the standard STA/LTA from a given input array a. The length of
    the STA is given by nsta in samples, respectively is the length of the
    LTA given by nlta in samples. Written in Python.

    .. note::

        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.classic_sta_lta` in this module!

    :type a: NumPy :class:`~numpy.ndarray`
    :param a: Seismic Trace
    :type nsta: int
    :param nsta: Length of short time average window in samples
    :type nlta: int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy :class:`~numpy.ndarray`
    :return: Characteristic function of classic STA/LTA
    """
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(a ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta
            