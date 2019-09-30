#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:15:54 2019

@author: robertocabieces
"""
##correlate
import numpy as np
import matplotlib.pyplot as plt
def plot_correlation_function(ccf,N,K):
    #ccf = ccf[N-K-1:N+K-1]
    kappa = np.arange(-len(ccf)//2,len(ccf)//2)
    plt.stem(kappa, ccf)
    plt.xlabel(r'$\kappa$')
    plt.axis([-K, K, -0.2, 1.1*max(ccf)])







x=np.array([1, 2, 3])
N=len(x)
h=np.array([3, 4, 5])
K=len(h)
y=np.convolve(h, x, mode='full')

ccf = 1/len(y) * np.correlate(y, x, mode='full')

plot_correlation_function(ccf,N,K)