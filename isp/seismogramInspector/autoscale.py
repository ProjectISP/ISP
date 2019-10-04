#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:42:05 2019

@author: robertocabieces
"""
import numpy as np
import matplotlib.pyplot as plt
def autoscale(ax=None, axis='y', margin=0.1):
    '''Autoscales the x or y axis of a given matplotlib ax object
    to fit the margins set by manually limits of the other axis,
    with margins in fraction of the width of the plot

    Defaults to current axes object if not specified.
    '''

    if ax is None:
        ax = plt.gca()
    newlow, newhigh = np.inf, -np.inf

    for artist in ax.collections + ax.lines:
        x,y = get_xy(artist)
        if axis == 'y':
            setlim = ax.set_ylim
            lim = ax.get_xlim()
            fixed, dependent = x, y
        else:
            setlim = ax.set_xlim
            lim = ax.get_ylim()
            fixed, dependent = y, x

        low, high = calculate_new_limit(fixed, dependent, lim)
        newlow = low if low < newlow else newlow
        newhigh = high if high > newhigh else newhigh

    margin = margin*(newhigh - newlow)

    setlim(newlow-margin, newhigh+margin)

def calculate_new_limit(fixed, dependent, limit):
    '''Calculates the min/max of the dependent axis given 
    a fixed axis with limits
    '''
    if len(fixed) > 2:
        mask = (fixed>limit[0]) & (fixed < limit[1])
        window = dependent[mask]
        low, high = window.min(), window.max()
    else:
        low = dependent[0]
        high = dependent[-1]
        if low == 0.0 and high == 1.0:
            # This is a axhline in the autoscale direction
            low = np.inf
            high = -np.inf
    return low, high

def get_xy(artist):
    '''Gets the xy coordinates of a given artist
    '''
    if "Collection" in str(artist):
        x, y = artist.get_offsets().T
    elif "Line" in str(artist):
        x, y = artist.get_xdata(), artist.get_ydata()
    else:
        raise ValueError("This type of object isn't implemented yet")
    return x, y


#fig, axes = plt.subplots(ncols = 4, figsize=(12,3))
#(ax1, ax2, ax3, ax4) = axes
#
#x = np.linspace(0,100,300)
#noise = np.random.normal(scale=0.1, size=x.shape)
#y = 2*x + 3 + noise
#
#for ax in axes:
#    ax.plot(x, y)
#    ax.scatter(x,y, color='red')
#    ax.axhline(50., ls='--', color='green')
#for ax in axes[1:]:
#    ax.set_xlim(20,21)
#    ax.set_ylim(40,45)
#
#autoscale(ax3, 'y', margin=0.1)
#autoscale(ax4, 'x', margin=0.1)
#
#ax1.set_title('Raw data')
#ax2.set_title('Specificed limits')
#ax3.set_title('Autoscale y')
#ax4.set_title('Autoscale x')
#plt.tight_layout()
#plt.show()