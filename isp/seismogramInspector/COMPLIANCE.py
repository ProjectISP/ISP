#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:11:20 2019

@author: robertocabieces
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:02:06 2019

@author: robertocabieces
"""
import warnings
warnings.filterwarnings("ignore")
import os
#import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from obspy import UTCDateTime as UDT, read, Trace, Stream
from obspy.core import UTCDateTime
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift
from scipy import signal
from subroutinesmod import *
import numpy as np
import math
from scipy.interpolate import interp1d
pi=np.pi
path=os.getcwd()
obsfiles = [f for f in listdir(path) if isfile(join(path, f))]
linf=0.02
lsup=0.1

def butter_bandpass(x,lowcut, highcut, fs, order):
    from scipy.signal import butter,lfilter
    x=x-np.mean(x)
    n=len(x)
    w = signal.blackman(n)
    x=x*w
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut /nyq
    b, a = butter(order, [low, high], btype='band')
    y=lfilter(b,a,x)
    return y
def power_log(x):
    n=math.ceil(math.log(x, 2))
    return n


#dt = UTCDateTime("2015-11-29T00:05:00")
#endtime=UTCDateTime("2015-11-29T23:55:00")
#st=read(path+"/"+"*WM*",starttime=dt,endtime=endtime)

##Read and Synchronization
st=read(path+"/"+"*WM*")
maxstart = np.max([tr.stats.starttime for tr in st])
minend =  np.min([tr.stats.endtime for tr in st])
st.trim(maxstart, minend)
dt=maxstart
nr = st.count() #count number of channels
#trY=st[2]
#trY=st[1]
trZ=st[3]
#Time=trZ.times("utcdatetime")
win=len(trZ.data)
if (win % 2) == 0:
   nfft1 = win/2 + 1
else:
   nfft1 = (win+1)/2

nfft=2**16

noverlap=int(nfft*0.5)
delta = st[0].stats.delta
fs=1/delta
fn=fs/2

Time_window=nfft/fs
print(Time_window)
freq=np.arange(0,fn,fn/nfft)
m=np.zeros((win,nr))
for i in range(4):
        tr=st[i]
        m[:,i]=(tr.data-np.mean(tr.data))

pdata= np.transpose(m)
#####Coherence######
Phh=signal.welch(pdata[0,:], fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
Pzz=signal.welch(pdata[3,:], fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
Pzh=signal.csd(pdata[3,:], pdata[0,:], fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
Phz=signal.csd(pdata[0,:], pdata[3,:], fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
##Calculate Complex Cohe
f=Pzz[0]

value1,freq1=find_nearest(f,linf)
value2,freq2=find_nearest(f,lsup)

num=Phz[1]
den=np.sqrt((Phh[1])*(Pzz[1]))
cohe=num/den
A=(np.abs(np.array(cohe[:])))

fig = plt.figure(figsize=(8,8))
ax1=fig.add_subplot(211)
plt.loglog(f,A)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()

print("Calculating Transfer Function")
##  Both are equal (be careful Pzh is not equal to Phz, must be conjugate)
Thz=cohe*np.sqrt(Pzz[1]/Phh[1]) #Crawford and Webb 2000
#Thz=np.conj(Pzh[1]/Phh[1]) #Bell et al., 2016

print("Calculating new Trace in Frequency Domain")
H=pdata[0,:]-np.mean(pdata[0,:])
Z=pdata[3,:]-np.mean(pdata[3,:])
t=np.array(range(len(Z)))/fs
Hf=np.fft.rfft(H,2**power_log(len(Z)))
Zf=np.fft.rfft(Z,2**power_log(len(Z)))

##Interpolate Thz to Hf

nfft1=len(Hf)

freq1=np.arange(0,fn,fn/nfft1)

set_interp = interp1d(f, Thz, kind='cubic')
Thzf = set_interp(freq1)
#plt.plot(f, np.abs(Thz), 'o', freq1, np.abs(Thzf), '--')

Zff=(Zf)-(Thzf*Hf)
Znew=np.fft.irfft(Zff)
Znew=Znew[0:len(Z)]
##
print("Calculating new PSDs")
##
Zpow=signal.welch(Z, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
Zpownew=signal.welch(Znew, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
##
fig = plt.figure(figsize=(8,8))
plt.loglog(Zpow[0],Zpow[1])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [counts/s]^2')
plt.loglog(Zpownew[0],Zpownew[1])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [counts/s]^2')
#Plot time Series
stats = {'network': 'WM', 'station': 'OBS06', 'location': '',
    	  'channel': 'SHZ', 'npts': win, 'sampling_rate': fs,
    	  'mseed': {'dataquality': 'M'}}

stats['starttime'] = dt
st2 = Stream([Trace(data=Znew, header=stats)])

#st2.write(path+"/WM.OBS06..SHZ.M.2015.333",format="MSEED")

st.filter('bandpass', freqmin=0.01,freqmax=0.1, corners=4, zerophase=True)
st2.filter('bandpass', freqmin=0.01,freqmax=0.1, corners=4, zerophase=True)
trZ=st[3]
trZnew=st2[0]

fig = plt.figure(figsize=(8,8))
ax1=fig.add_subplot(211)
plt.plot(t,trZ.data)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [counts/s]')
ax2=fig.add_subplot(212)
plt.plot(t,trZnew.data)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [counts/s]')
plt.show()