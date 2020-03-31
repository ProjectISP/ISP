#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 18:15:16 2019

@author: robertocabieces
"""
from mtspec import mtspec
import numpy as np
from obspy import read
import math
import scipy.signal
import matplotlib.pyplot as plt

from isp.seismogramInspector.entropy import spectral_entropy


def find_nearest(array, value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx,val

def MTspectrum(data,win,dt,tbp,ntapers,linf,lsup):

    if (win % 2) == 0:
       nfft = win/2 + 1
    else:
       nfft = (win+1)/2
    
    
    lim=len(data)-win
    S=np.zeros([int(nfft),int(lim)])
    data2 = np.zeros(2 ** math.ceil(math.log2(win)))

    for n in range(lim):
        data1=data[n:win+n]
        data1=data1-np.mean(data1)
        data2[0:win]=data1
        spec,freq = mtspec(data2,delta=dt ,time_bandwidth=tbp,number_of_tapers=ntapers)
        spec=spec[0:int(nfft)]
        S[:,n]=spec

    value1,freq1=find_nearest(freq,linf)
    value2,freq2=find_nearest(freq,lsup)
    S=S[value1:value2]
    
    return S

def Entropydetect(data,win,dt):           
    
    lim=len(data)-win
    Entropy=np.zeros([1,int(lim)]) 
     
    for n in range(lim):
        data1=data[n:win+n]
        data1=data1-np.mean(data1)
        Entropy1 = spectral_entropy(data1, sf=1/dt, method='welch', nperseg=win, normalize=True)
        Entropy[:,n]=Entropy1 
    
    return Entropy


#######
def _pad_zeros(a, num, num2=None):
    """Pad num zeros at both sides of array a"""
    if num2 is None:
        num2 = num
    hstack = [np.zeros(num, dtype=a.dtype), a, np.zeros(num2, dtype=a.dtype)]
    return np.hstack(hstack)


def _xcorr_padzeros(a, b, shift, method):
    """
    Cross-correlation using SciPy with mode='valid' and precedent zero padding
    """
    if shift is None:
        shift = (len(a) + len(b) - 1) // 2
    dif = len(a) - len(b) - 2 * shift
    if dif > 0:
        b = _pad_zeros(b, dif // 2)
    else:
        a = _pad_zeros(a, -dif // 2)
    return scipy.signal.correlate(a, b, 'valid', method)


def _xcorr_slice(a, b, shift, method):
    """
    Cross-correlation using SciPy with mode='full' and subsequent slicing
    """
    mid = (len(a) + len(b) - 1) // 2
    if shift is None:
        shift = mid
    if shift > mid:
        # Such a large shift is not possible without zero padding
        return _xcorr_padzeros(a, b, shift, method)
    cc = scipy.signal.correlate(a, b, 'full', method)
    return cc[mid - shift:mid + shift + len(cc) % 2]


def get_lags(cc):
    """
    Return array with lags
    :param cc: Cross-correlation returned by correlate_maxlag.
    :return: lags
    """
    mid = (len(cc) - 1) / 2
    if len(cc) % 2 == 1:
        mid = int(mid)
    return np.arange(len(cc)) - mid

def correlate_maxlag(a, b, maxlag, demean=True, normalize='naive',
                     method='auto'):
    """
    Cross-correlation of two signals up to a specified maximal lag.
    This function only allows 'naive' normalization with the overall
    standard deviations. This is a reasonable approximation for signals of
    similar length and a relatively small maxlag parameter.
    :func:`correlate_template` provides correct normalization.
    :param a,b: signals to correlate
    :param int maxlag: Number of samples to shift for cross correlation.
        The cross-correlation will consist of ``2*maxlag+1`` or
        ``2*maxlag`` samples. The sample with zero shift will be in the middle.
    :param bool demean: Demean data beforehand.
    :param normalize: Method for normalization of cross-correlation.
        One of ``'naive'`` or ``None``
        ``'naive'`` normalizes by the overall standard deviation.
        ``None`` does not normalize.
    :param method: correlation method to use.
        See :func:`scipy.signal.correlate`.
    :return: cross-correlation function.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if demean:
        a = a - np.mean(a)
        b = b - np.mean(b)
    # choose the usually faster xcorr function for each method
    _xcorr = _xcorr_padzeros if method == 'direct' else _xcorr_slice
    cc = _xcorr(a, b, maxlag, method)
    if normalize == 'naive':
        norm = (np.sum(a ** 2) * np.sum(b ** 2)) ** 0.5
        if norm <= np.finfo(float).eps:
            # norm is zero
            # => cross-correlation function will have only zeros
            cc[:] = 0
        elif cc.dtype == float:
            cc /= norm
        else:
            cc = cc / norm
    elif normalize is not None:
        raise ValueError("normalize has to be one of (None, 'naive'))")
    return cc

def _window_sum(data, window_len):
    """Rolling sum of data"""
    window_sum = np.cumsum(data)
    # in-place equivalent of
    # window_sum = window_sum[window_len:] - window_sum[:-window_len]
    # return window_sum
    np.subtract(window_sum[window_len:], window_sum[:-window_len],
                out=window_sum[:-window_len])
    return window_sum[:-window_len]

####

def correlate_template(data, template, mode='valid', demean=True,
                       normalize='full', method='auto'):
    """
    Normalized cross-correlation of two signals with specified mode.
    If you are interested only in a part of the cross-correlation function
    around zero shift use :func:`correlate_maxlag` which allows to
    explicetly specify the maximum lag.
    :param data,template: signals to correlate. Template array must be shorter
        than data array.
    :param normalize:
        One of ``'naive'``, ``'full'`` or ``None``.
        ``'full'`` normalizes every correlation properly,
        whereas ``'naive'`` normalizes by the overall standard deviations.
        ``None`` does not normalize.
    :param mode: correlation mode to use.
        See :func:`scipy.signal.correlate`.
    :param bool demean: Demean data beforehand.
        For ``normalize='full'`` data is demeaned in different windows
        for each correlation value.
    :param method: correlation method to use.
        See :func:`scipy.signal.correlate`.
    :return: cross-correlation function.
    .. note::
        Calling the function with ``demean=True, normalize='full'`` (default)
        returns the zero-normalized cross-correlation function.
        Calling the function with ``demean=False, normalize='full'``
        returns the normalized cross-correlation function.
    """
    data = np.asarray(data)
    template = np.asarray(template)
    lent = len(template)
    if len(data) < lent:
        raise ValueError('Data must not be shorter than template.')
    if demean:
        template = template - np.mean(template)
        if normalize != 'full':
            data = data - np.mean(data)
    cc = scipy.signal.correlate(data, template, mode, method)
    if normalize is not None:
        tnorm = np.sum(template ** 2)
        if normalize == 'naive':
            norm = (tnorm * np.sum(data ** 2)) ** 0.5
            if norm <= np.finfo(float).eps:
                cc[:] = 0
            elif cc.dtype == float:
                cc /= norm
            else:
                cc = cc / norm
        elif normalize == 'full':
            pad = len(cc) - len(data) + lent
            if mode == 'same':
                pad1, pad2 = (pad + 2) // 2, (pad - 1) // 2
            else:
                pad1, pad2 = (pad + 1) // 2, pad // 2
            data = _pad_zeros(data, pad1, pad2)
            # in-place equivalent of
            # if demean:
            #     norm = ((_window_sum(data ** 2, lent) -
            #              _window_sum(data, lent) ** 2 / lent) * tnorm) ** 0.5
            # else:
            #      norm = (_window_sum(data ** 2, lent) * tnorm) ** 0.5
            # cc = cc / norm
            if demean:
                norm = _window_sum(data, lent) ** 2
                if norm.dtype == float:
                    norm /= lent
                else:
                    norm = norm / lent
                np.subtract(_window_sum(data ** 2, lent), norm, out=norm)
            else:
                norm = _window_sum(data ** 2, lent)
            norm *= tnorm
            if norm.dtype == float:
                np.sqrt(norm, out=norm)
            else:
                norm = np.sqrt(norm)
            mask = norm <= np.finfo(float).eps
            if cc.dtype == float:
                cc[~mask] /= norm[~mask]
            else:
                cc = cc / norm
            cc[mask] = 0
        else:
            msg = "normalize has to be one of (None, 'naive', 'full')"
            raise ValueError(msg)
    return cc

def cohe(tr1,tr2,fs,nfft):
    
    noverlap=int(nfft*0.5)
    
    Phh=scipy.signal.welch(tr1, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
    Pzz=scipy.signal.welch(tr1, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
    #Pzh=scipy.signal.csd(tr2, tr1, fs=fs, open_main_window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
    Phz=scipy.signal.csd(tr1, tr2, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
    f=Pzz[0]
    num=Phz[1]
    den=np.sqrt((Phh[1])*(Pzz[1]))
    cohe=num/den
    A=(np.abs(np.array(cohe[:])))
    fig = plt.figure(figsize=(8,8))
    #ax1=fig.add_subplot(211)
    plt.loglog(f,A)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.show()

###
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
