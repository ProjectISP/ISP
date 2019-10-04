#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:26:10 2018

@author: Roberto Cabieces Diaz

Script based on the paper 

Cabieces, R., , García-Yeguas, A., Villaseñor, A., Buforn, E., Pazos, A.,Krüger, F., Olivar, A.,Barco, J. (2018) Slowness vector estimation over large-aperture sparse arrays with the Continuous Wavelet Transform: Application to Ocean Bottom Seismometers. Geophys. J. Int. (submitted, september 2018 editor accepted).
"""


import numpy as np

def ccwt(data, srate,fmin, fmax, wmin, wmax,tt, nf):
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
    npts = len(data)
    #time = (np.arange(0, npts-1, 1))/srate
    frex = np.logspace(np.log10(fmin), np.log10(fmax), nf) #Logarithmically space central frequencies

    wtime =  np.arange(-tt,tt+dt, dt) #Kernel of the Mother Morlet Wavelet
    half_wave = (len(wtime)-1)/2
    nCycles=np.logspace(np.log10(wmin),np.log10(wmax),nf)

    ###FFT parameters
    nKern = len(wtime)
    nConv = nKern+npts-1
    
    ##Initialize output time-frequency data
    tf = np.zeros((len(frex), npts-1), dtype=np.complex)
    tf = np.transpose(tf)
     ##FFT data
    dataX=np.fft.fft(data,nConv)
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
        cwt = np.fft.ifft(np.multiply(cmwX,dataX))
        end =len(cwt)
        cwt = cwt[int(half_wave+1):int(end-half_wave)]
        tf[:,fi] = cwt
    #Power db conversion   this is opcional, it is already done in the script              
    #tf=np.abs(tf)**2
    #maxtf = np.max(tf)
    #tf = 10*(np.log10(tf/maxtf))
    return tf.T
