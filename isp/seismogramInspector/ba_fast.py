#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:47:47 2019

@author: robertocabieces
"""

##ba fast
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:28:44 2019

@author: robertocabieces
"""

####Generation of bank of atoms

import numpy as np
import math


def ccwt_ba_fast(npts, srate,fmin, fmax, wmin, wmax,tt, nf):
    """
    Continuous Wavelet Transformation, using Morlet Wavelet

    :param data: time dependent signal.
    :param srate: sampling frequency
     Tradeoff between time and frequency resolution (number of cycles, wmin, wmax)
    :param wmin: number of cycles minumum
    :param wmin: number of cycles maximum, 
    :Central frequency of the Morlet Wavelet
    :param fmin: minimum frequency (in Hz)
    :param fmax: maximum frequency (in Hz)
    :param nf: number of logarithmically spaced frequencies between fmin and

    :return: time frequency representation of st, type numpy.ndarray of complex values, shape = (nf, len(st)).
    """
    
    ##Wavelet parameters
    dt = 1/srate
    
    #time = (np.arange(0, npts-1, 1))/srate
    #frex = np.logspace(np.log2(fmin), np.log2(fmax), nf,base=2) #Logarithmically space central frequencies
    frex = np.logspace(np.log10(fmin), np.log10(fmax), nf)
    wtime =  np.arange(-tt,tt+dt, dt) #Kernel of the Mother Morlet Wavelet
    half_wave = (len(wtime)-1)/2
    nCycles=np.logspace(np.log10(wmin),np.log10(wmax),nf)

    ###FFT parameters
    nKern = len(wtime)
    nConv = npts+nKern
    nConv=2**math.ceil(math.log2(nConv))
    #####Proyecto,realMorlet#####
    if (nConv % 2) == 0:
       nConv2 = nConv/2 + 1
    else:
       nConv2 = (nConv+1)/2

    tf = np.zeros((len(frex), nConv), dtype=np.complex)
    tf = np.transpose(tf)
    
    ##loop over frequencies
   
    for fi in range(0, len(frex)):
        ##Create the morlet wavelet and get its fft
        s = nCycles[fi]/(2*np.pi*frex[fi])
        #Normalize Factor
        A = 1/(np.pi*s**2)**0.25 
        #complexsine = np.multiply(1j*2*(np.pi)*frex[fi],wtime))
        #Gaussian = np.exp(-1*np.divide(np.power(wtime,2),2*s**2))
        cmw = np.multiply(np.exp(np.multiply(1j*2*(np.pi)*frex[fi],wtime)),np.exp(-1*np.divide(np.power(wtime,2),2*s**2)))
        cmw = cmw.conjugate()
        #Normalizing.The square root term causes the wavelet to be normalized to have an energy (squared integral) of 1.
        cmw = A*cmw
        #Calculate the fft of the "atom"
        cmwX = np.fft.fft(cmw,nConv)
        
        #Convolution
        tf[:,fi] = cmwX
        
        
    return tf.T,nConv,frex,half_wave
