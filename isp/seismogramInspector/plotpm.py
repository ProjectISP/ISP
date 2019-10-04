#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:44:38 2019

@author: robertocabieces
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotpmNEZ(E,N,Z):
     
     fig2, axs = plt.subplots(nrows=2, ncols=2,figsize=(12,8))
    # plot time signal:
     axs[0, 0].set_title("E-Z")
     axs[0, 0].plot(E, Z, 'o', ls='-', ms=0.25)
     axs[0, 0].set_xlabel("Amplitude N")
     axs[0, 0].set_ylabel("Amplitude E")
    
     axs[0, 1].set_title("E-N")
     axs[0, 1].plot(E, N, 'o', ls='-', ms=0.25)
     axs[0, 1].set_xlabel("Amplitude N")
     axs[0, 1].set_ylabel("Amplitude E")
     
     axs[1, 0].set_title("N-Z")
     axs[1, 0].plot(N, Z, 'o', ls='-', ms=0.25)
     axs[1, 0].set_xlabel("Amplitude N")
     axs[1, 0].set_ylabel("Amplitude E")
     
     
     
     plt.rcParams['legend.fontsize'] = 10

     fig3 = plt.figure(figsize=(8,8))
     ax = fig3.gca(projection='3d')

    # Prepare arrays x, y, z


     ax.plot(E, N, Z, label='Particle motion')
     ax.legend()

     plt.show()
     