#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 02:01:51 2019

@author: robertocabieces
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:44:06 2019

@author: robertocabieces
"""

#####

##########CWT with convolution of bamk of atoms######


import numpy as np
from obspy.signal.filter import lowpass
import numpy as np

def cwt_fast(data,ba,nConv,frex,half_wave, fs):
    npts=len(data)
    ##FFT data
    cf = []
    tf = []
    dataX = np.fft.fft(data,nConv)
##loop over frequencies
    for fi in range(len(frex)):
            cmwX=ba[fi,:]
            #cwt = np.fft.irfft(np.multiply(cmwX,dataX))
            cwt = np.fft.ifft(np.multiply(cmwX,dataX))
            #end =len(cwt)
            cwt = cwt[int(half_wave+1):npts+int(half_wave+1)]
            d = np.diff(np.log10(np.abs(cwt)**2))
            cf.append(d)
            tf.append(cwt)

    tf=np.asarray(tf)
    sc=np.mean(tf, axis=0)
    cf = np.mean(cf, axis = 0)
    cf = cf - np.mean(cf)
    n = len(cf)
    window = np.hamming(n)
    cf = window * cf
    cf = lowpass(cf, 0.2, fs, corners=4, zerophase=True)
    cf = window * cf
    return cf,sc,tf
   
