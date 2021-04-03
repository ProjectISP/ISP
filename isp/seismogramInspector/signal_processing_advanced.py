#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 18:15:16 2019

@author: robertocabieces
"""
from signal import signal
from mtspec import mtspec
import numpy as np
import math
import scipy.signal
import pywt
from isp.seismogramInspector.entropy import spectral_entropy
import copy
from obspy.signal.trigger import classic_sta_lta
import obspy.signal

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

    if win > 0:
        N = int(win/dt)
    else:
        N=1024
    win = 2 ** math.ceil(math.log2(N))

    lim=len(data)-win
    Entropy=np.zeros([1,int(lim)])

    for n in range(lim):
        data1=data[n:win+n]
        data1=data1-np.mean(data1)
        Entropy1 = spectral_entropy(data1, sf=1/dt, method='welch', nperseg=win, normalize=True)
        Entropy[:,n]=Entropy1 
    
    return Entropy[0]


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

def cohe(tr1, tr2, fs, nfft, overlap):
    """
    Estimate Coherence throug Welch method

    """

    nfft = 2 ** math.ceil(math.log2(nfft)) # counts
    overlap = overlap / 100
    noverlap = int(nfft * overlap)
    Phh=scipy.signal.welch(tr1, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
    Pzz=scipy.signal.welch(tr2, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
    #Pzh=scipy.signal.csd(tr2, tr1, fs=fs, open_main_window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
    Phz=scipy.signal.csd(tr1, tr2, fs=fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='density', axis=-1)
    f=Pzz[0]
    num=Phz[1]
    den=np.sqrt((Phh[1])*(Pzz[1]))
    cohe=num/den
    phase = np.angle(cohe*180/np.pi)
    A=(np.abs(np.array(cohe[:])))
    return A, f, phase

###
def spectrumelement(data,delta,sta):
    """

    Return the amplitude spectrum using multitaper aproach

    """

    spec, freq, jackknife_errors, _, _ = mtspec(data, delta=delta , time_bandwidth=3.5, statistics=True)
    spec = np.sqrt(spec) #mtspec Amplitude spectrum
    jackknife_errors = np.sqrt(jackknife_errors)
    return spec, freq, jackknife_errors


def sta_lta(data, sampling_rate, STA = 1, LTA = 40):
    #from obspy.signal.filter import lowpass
    cft = classic_sta_lta(data, int(STA * sampling_rate), int(LTA * sampling_rate))
    #cft=cft-np.mean(cft)
    ##
    #cft = np.diff(cft)
    ##
    #window = np.hanning(len(cft))
    #cft = window*cft
    #cf1= lowpass(cft, 0.15, sampling_rate, corners=3, zerophase=True)
    return cft


def envelope(data,sampling_rate):

    #from obspy.signal.filter import lowpass
    N = len(data)
    D = 2 ** math.ceil(math.log2(N))
    z = np.zeros(D - N)
    data = np.concatenate((data, z), axis=0)
    ###Necesary padding with zeros
    data_envelope = obspy.signal.filter.envelope(data)
    data_envelope = data_envelope[0:N]
    #window = np.hanning(len(data_envelope))
    #data_envelope = window * data_envelope
    #data_envelope1 = lowpass(data_envelope, 0.15, sampling_rate, corners=3, zerophase=True)
    return data_envelope


def add_white_noise(tr, SNR_dB):
    L = len(tr.data)
    SNR = 10**(SNR_dB/10)
    Esym = np.sum(np.abs(tr.data)**2)/L
    N0 = Esym / SNR
    noiseSigma = np.sqrt(N0)
    n = noiseSigma * np.random.normal(size=L)
    tr.data = tr.data+ n
    return tr


def whiten(tr, freqmin, freqmax):
    nsamp = tr.stats.sampling_rate

    n = len(tr.data)
    if n == 1:
        return tr
    else:
        frange = float(freqmax) - float(freqmin)
        nsmo = int(np.fix(min(0.01, 0.5 * (frange)) * float(n) / nsamp))
        f = np.arange(n) * nsamp / (n - 1.)
        JJ = ((f > float(freqmin)) & (f < float(freqmax))).nonzero()[0]

        # signal FFT
        FFTs = np.fft.fft(tr.data)
        FFTsW = np.zeros(n) + 1j * np.zeros(n)

        # Apodization to the left with cos^2 (to smooth the discontinuities)
        smo1 = (np.cos(np.linspace(np.pi / 2, np.pi, nsmo + 1)) ** 2)
        FFTsW[JJ[0]:JJ[0] + nsmo + 1] = smo1 * np.exp(1j * np.angle(FFTs[JJ[0]:JJ[0] + nsmo + 1]))

        # boxcar
        FFTsW[JJ[0] + nsmo + 1:JJ[-1] - nsmo] = np.ones(len(JJ) - 2 * (nsmo + 1)) \
                                                * np.exp(1j * np.angle(FFTs[JJ[0] + nsmo + 1:JJ[-1] - nsmo]))

        # Apodization to the right with cos^2 (to smooth the discontinuities)
        smo2 = (np.cos(np.linspace(0, np.pi / 2, nsmo + 1)) ** 2)
        espo = np.exp(1j * np.angle(FFTs[JJ[-1] - nsmo:JJ[-1] + 1]))
        FFTsW[JJ[-1] - nsmo:JJ[-1] + 1] = smo2 * espo

        whitedata = 2. * np.fft.ifft(FFTsW).real

        tr.data = np.require(whitedata, dtype="float32")

        return tr


# Functions Noise Processing

def get_window(N, alpha=0.2):

    window = np.ones(N)
    x = np.linspace(-1., 1., N)
    ind1 = (abs(x) > 1 - alpha) * (x < 0)
    ind2 = (abs(x) > 1 - alpha) * (x > 0)
    window[ind1] = 0.5 * (1 - np.cos(np.pi * (x[ind1] + 1) / alpha))
    window[ind2] = 0.5 * (1 - np.cos(np.pi * (x[ind2] - 1) / alpha))
    return window

def normalize(tr, clip_factor=6, clip_weight=10, norm_win=None, norm_method="1bit"):
    if norm_method == 'clipping':
        lim = clip_factor * np.std(tr.data)
        tr.data[tr.data > lim] = lim
        tr.data[tr.data < -lim] = -lim

    elif norm_method == "clipping iteration":
        lim = clip_factor * np.std(np.abs(tr.data))

        # as long as still values left above the waterlevel, clip_weight
        while tr.data[np.abs(tr.data) > lim] != []:
            tr.data[tr.data > lim] /= clip_weight
            tr.data[tr.data < -lim] /= clip_weight

    elif norm_method == 'time normalization':
        lwin = int(tr.stats.sampling_rate * norm_win)
        st = 0  # starting point
        N = lwin  # ending point

        while N < tr.stats.npts:
            win = tr.data[st:N]

            w = np.mean(np.abs(win)) / (2. * lwin + 1)

            # weight center of window
            tr.data[st + int(lwin / 2)] /= w

            # shift window
            st += 1
            N += 1

        # taper edges
        #taper = get_window(tr.stats.npts)
        #tr.data *= taper

    elif norm_method == "1bit":
        tr.data = np.sign(tr.data)
        tr.data = np.float32(tr.data)

    return tr

# Denoise Section

def smoothing(tr, type='gaussian', k=5, fwhm=0.05):
    # k window size in seconds

    n = len(tr.data)

    if type == 'mean':
        k = int(k * tr.stats.sampling_rate)

        # initialize filtered signal vector
        filtsig = np.zeros(n)
        for i in range(k, n - k - 1):
            # each point is the average of k surrounding points
            # print(i - k,i + k)
            filtsig[i] = np.mean(tr.data[i - k:i + k])

        tr.data = filtsig

    if type == 'gaussian':
        ## create Gaussian kernel
        # full-width half-maximum: the key Gaussian parameter in seconds
        # normalized time vector in seconds
        k = int(k * tr.stats.sampling_rate)
        fwhm = int(fwhm * tr.stats.sampling_rate)
        gtime = np.arange(-k, k)
        # create Gaussian window
        gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)
        # compute empirical FWHM

        pstPeakHalf = k + np.argmin((gauswin[k:] - .5) ** 2)
        prePeakHalf = np.argmin((gauswin - .5) ** 2)
        # empFWHM = gtime[pstPeakHalf] - gtime[prePeakHalf]
        # show the Gaussian
        # plt.plot(gtime/tr.stats.sampling_rate,gauswin)
        # plt.plot([gtime[prePeakHalf],gtime[pstPeakHalf]],[gauswin[prePeakHalf],gauswin[pstPeakHalf]],'m')
        # then normalize Gaussian to unit energy
        gauswin = gauswin / np.sum(gauswin)
        # implement the filter
        # initialize filtered signal vector
        filtsigG = copy.deepcopy(tr.data)
        # implement the running mean filter
        for i in range(k + 1, n - k - 1):
            # each point is the weighted average of k surrounding points
            filtsigG[i] = np.sum(tr.data[i - k:i + k] * gauswin)

        tr.data = filtsigG

    if type == 'tkeo':
        # extract needed variables

        emg = tr.data

        # initialize filtered signal
        emgf = copy.deepcopy(emg)

        # the loop version for interpretability
        # for i in range(1, len(emgf) - 1):
        #    emgf[i] = emg[i] ** 2 - emg[i - 1] * emg[i + 1]

        # the vectorized version for speed and elegance

        emgf[1:-1] = emg[1:-1] ** 2 - emg[0:-2] * emg[2:]

        ## convert both signals to zscore

        # find timepoint zero
        # time0 = np.argmin(emgtime ** 2)

        # convert original EMG to z-score from time-zero
        # emgZ = (emg - np.mean(emg[0:time0])) / np.std(emg[0:time0])

        # same for filtered EMG energy
        # emgZf = (emgf - np.mean(emgf[0:time0])) / np.std(emgf[0:time0])
        # tr.data = emgZf
        tr.data = emgf

    return tr

def wavelet_denoise(tr, threshold = 0.04, dwt = 'sym4' ):
    # Threshold for filtering
    # Create wavelet object and define parameters
    w = pywt.Wavelet(dwt)
    maxlev = pywt.dwt_max_level(len(tr.data), w.dec_len)
    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(tr.data, dwt, level=maxlev)
    # cA = pywt.threshold(cA, threshold*max(cA))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
    datarec = pywt.waverec(coeffs, dwt)
    tr.data = datarec
    return tr


