#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:28:17 2019

@author: robertocabieces
"""

#FKcoherence
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 19 16:54:26 2019

@author: robertocabieces
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from subroutinesmod import *
from obspy.core import read,UTCDateTime
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import get_geometry
import nitime.algorithms as alg
from nitime import utils
from scipy import fftpack
#import util as ut
import numpy as np
import math
import nitime.timeseries as ts
from mtspec import *
from nitime.analysis import CoherenceAnalyzer, MTCoherenceAnalyzer
import nitime
import pandas as pd
from datetime import date

# ==== USER INPUT PARAMETER ===

def FKCoherence(path,path_coords,start,linf,lsup,slim,win_len,sinc,method): 
    path_coords=path_coords+"/"+"coords.txt"
    sides = 'onesided'
    pi= math.pi
    rad=180/pi
        
    smax=slim
    smin=-1*smax
        
    
    Sx=np.arange(smin,smax,sinc)[np.newaxis]
    Sy=np.arange(smin,smax,sinc)[np.newaxis]
    
    nx=ny=len(Sx[0])
    Sy=np.fliplr(Sy)
    
    ###Convert start from Greogorian to actual date
    
    #Time = d.timetuple()
    Time=start
    Time = Time-int(Time)
    d = date.fromordinal(int(start))
    date1=d.isoformat()
    
    
    H = (Time*24)
    H1=int(H) #Horas
    minutes = (H-int(H))*60
    minutes1=int(minutes)
    seconds = (minutes - int(minutes))*60 
    
    H1=str(H1).zfill(2)
    minutes1=str(minutes1).zfill(2)
    seconds="%.2f" % seconds
    seconds=str(seconds).zfill(2)
    
    ##
    #time.struct_time(tm_year=2002, tm_mon=3, tm_mday=11, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=0, tm_yday=70, tm_isdst=-1)
    ##Build the initial date
    
    DATE=date1+"T"+str(H1)+minutes1+seconds
    print(DATE)
    t1=UTCDateTime(DATE)
     
    path=path+"/"+"*.*"
    st=read(path,starttime=t1,endtime=t1+win_len)
    st.sort()
    st.plot()
    df=pd.read_csv(path_coords,sep='\t')
    n=df.Name.count()

    for i in range(n):

        st[i].stats.coordinates = AttribDict({'latitude': df.loc[i].Lat,'elevation': 0.0,'longitude': df.loc[i].Lon})
    
    coord = get_geometry(st, coordsys='lonlat', return_center=True)     
    # =============================
    tr=st[0]
    win=len(tr.data)
    if (win % 2) == 0:
       nfft = win/2 + 1
    else:
       nfft = (win+1)/2
       
    #win=1001
    nr = st.count()   #number of stations            
    delta = st[0].stats.delta
    fs=1/delta
    #nsamp = win*fs
    fn=fs/2
    nwin=(win+1)/2
    fs=1/delta        
    freq=np.arange(0,fn,fn/nfft)
    
    value1,freq1=find_nearest(freq,linf)
    value2,freq2=find_nearest(freq,lsup)
    
    df=value2-value1
    #Extraes los datos
    m=np.zeros((win,nr))
    #WW=np.hamming(int(win))
    #WW=np.transpose(WW)
    for i in range(nr):
        tr=st[i]
        #m[:,i]=(trace.data-np.mean(trace.data))*WW
        m[:,i]=(tr.data-np.mean(tr.data))
    pdata= np.transpose(m)   
    #####Coherence######
    NW = 2  #the time-bandwidth product##Buena seleccion de 2-3
    K = 2 * NW - 1
    tapers, eigs = alg.dpss_windows(win, NW, K)
    tdata = tapers[None, :, :] * pdata[:, None, :] #filas estaciones, 
    #columnas por tapers, profundiadad data
    tspectra = fftpack.fft(tdata)
    
    w = np.empty((nr, int(K), int(nfft)))
    for i in range(nr):
        w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides=sides)
    
    nseq=nr
    L=int(nfft)
    csd_mat = np.zeros((nseq, nseq, L), 'D')
    psd_mat = np.zeros((2, nseq, nseq, L), 'd')
    coh_mat = np.zeros((nseq, nseq, L), 'd')
    coh_var = np.zeros_like(coh_mat)
    Cx=np.ones((nr,nr,df),dtype=np.complex128) 
    
    if method=="MTP.COHERENCE":
        for i in range(nr):
            for j in range(nr):
        
                sxy = alg.mtm_cross_spectrum(tspectra[i],(tspectra[j]), (w[i], w[j]), sides='onesided')
                sxx = alg.mtm_cross_spectrum(tspectra[i], tspectra[i], w[i], sides='onesided')
                syy = alg.mtm_cross_spectrum(tspectra[j], tspectra[j], w[j], sides='onesided')
                s=sxy/np.sqrt((sxx*syy))
                cxcohe=s[value1:value2]        
                Cx[i,j,:]=cxcohe
    
    ####Calculates Conventional FK-power  ##without normalization
    if method=="FK":       
        for i in range(nr):
            for j in range(nr):
                A=np.fft.rfft(m[:,i])
                B=np.fft.rfft(m[:,j])
                out=A*np.conjugate(B)
        
                cxcohe=out[value1:value2]
                Cx[i,j,:]=cxcohe
       
    r=np.zeros((nr,2))
    A=np.zeros((nr,1),dtype=np.complex128)
    S=np.zeros((1,2))
    K=np.zeros((1,5))
    Pow=np.zeros((len(Sx[0]),len(Sy[0])))
    for n in range(nr):
        r[n,:]=coord[n][0:2]
    
    freq=freq[value1:value2] 
    
    for i in range(ny):
        for j in range(nx):
            S[0,0]=Sx[0][j]
            S[0,1]=Sy[0][i]
            k=(S*r)
            K = np.sum(k,axis=1)
            n=0
            for f in freq:
                    A=np.exp(-1j*2*pi*f)**K
                    B=np.conjugate(np.transpose(A))
                    D=np.matmul(B,Cx[:,:,n])/nr
                    P=np.matmul(D,A)/nr                
                    Pow[i,j]+=np.abs(P)
                    n=n+1
    
    Pow=Pow/len(freq)
    Pow1=Pow ##Azimuth
    
    
    #Plotting part
    Pow=np.fliplr(Pow)
    Pow=np.flipud(Pow)
        
    fig, ax = plt.subplots()   
    x=y=np.linspace(smin,smax,nx)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, Pow, 50,cmap =plt.cm.jet)
    cs = ax.contourf(X, Y, Pow,50,cmap=plt.cm.jet)
    plt.ylabel('Sy [s/km]')
    plt.xlabel('Sx [s/km]')
    cbar = fig.colorbar(cs)
    plt.show()
    return Pow
    #fig.savefig("Coherence.png",dpi=400)                   



##Prueba
    
#path_coords="/Users/robertocabieces/Documents/obs_array/coords.txt"
#path="/Users/robertocabieces/Desktop/GUIPYTHON/ArrayProcessing/260/SCTF"
#
#start=735858.63442909
#win_len=24
#f_min=0.1
#fmax=0.15
#slim=0.3
#sinc = 0.005
#
#FKCoherence(path,path_coords,start,f_min,fmax,slim,win_len,sinc)


