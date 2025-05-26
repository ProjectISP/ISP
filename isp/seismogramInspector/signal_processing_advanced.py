#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 18:15:16 2019

@author: robertocabieces
"""
from obspy import Trace
import nitime.algorithms as tsa
import numpy as np
import math
import scipy.signal
import pywt  # this should be added on requirements.txt if is a necessary package
from scipy import ndimage
from scipy.signal import savgol_filter

from isp.seismogramInspector.entropy import spectral_entropy
import copy
from obspy.signal.trigger import classic_sta_lta
import obspy.signal
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

try:
    # use numba as optional for improve performance
    from numba import jit, njit

    __use_numba = True
except ImportError:
    __use_numba = False

def median_absolute_deviation(x):
    """
    Returns the median absolute deviation from the window's median
    :param x: Values in the window
    :return: MAD
    """
    return np.median(np.abs(x - np.median(x)))

def find_nearest(array, value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx,val


# def MTspectrum(data, win, dt, tbp, ntapers, linf, lsup):
#     if (win % 2) == 0:
#         nfft = win / 2 + 1
#     else:
#         nfft = (win + 1) / 2
#
#     lim = len(data) - win
#     S = np.zeros([int(nfft), int(lim)])
#     data2 = np.zeros(2 ** math.ceil(math.log2(win)))
#
#     for n in range(lim):
#         data1 = data[n:win + n]
#         data1 = data1 - np.mean(data1)
#         data2[0:win] = data1
#         freq, spec, _ = tsa.multi_taper_psd(data2, 1/dt, adaptive=False, jackknife=False, low_bias=False)
#         spec = spec[0:int(nfft)]
#         S[:, n] = spec
#
#     value1, freq1 = find_nearest(freq, linf)
#     value2, freq2 = find_nearest(freq, lsup)
#     S = S[value1:value2]
#
#     return S

def MTspectrum(data, win, dt, linf, lsup, step_percentage=0.1, res=1):

    # win -- samples
    # Ensure nfft is a power of 2
    nfft = 2 ** math.ceil(math.log2(win)) # Next power to 2

    # Step size as a percentage of window size
    step_size = max(1, int(nfft * step_percentage))  # Ensure step size is at least 1
    lim = len(data) - nfft  # Define sliding window limit
    num_steps = (lim // step_size) + 1  # Total number of steps
    S = np.zeros([nfft // 2 + 1, num_steps])  # Adjust output size for reduced steps

    # Precompute frequency indices for slicing spectrum
    fs = 1 / dt  # Sampling frequency
    #freq, _, _ = tsa.multi_taper_psd(np.zeros(nfft), fs, adaptive=True, jackknife=False, low_bias=False)

    for idx, n in enumerate(range(0, lim, step_size)):
        #print(f"{(n + 1) * 100 / lim:.2f}% done")
        data1 = data[n:nfft + n]
        data1 = data1 - np.mean(data1)
        freq, spec, _ = tsa.multi_taper_psd(data1, fs, adaptive=True, jackknife=False, low_bias=True)

        S[:,idx] = spec

    value1, freq1 = find_nearest(freq, linf)
    value2, freq2 = find_nearest(freq, lsup)

    spectrum = S[value1:value2,:]

    if res > 1:
          spectrum = ndimage.zoom(spectrum, (1.0, 1 / spectrum))
          t = np.linspace(0, res * dt * spectrum.shape[1], spectrum.shape[1])
          f = np.linspace(linf, lsup, spectrum.shape[0])
    else:
        t = np.linspace(0, len(data)*dt, spectrum.shape[1])
        f = np.linspace(linf, lsup, spectrum.shape[0])

    return spectrum, num_steps, t, f




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

def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)
###
def spectrumelement(data, delta, sta=None):
    """
    Return the amplitude spectrum using multitaper and compare with FFT.

    Parameters:
    - data: array-like, time-domain signal
    - delta: float, sample spacing (1 / sampling rate)
    - sta: unused (can be removed or kept for future use)

    Returns:
    - amplitude: amplitude spectrum from multitaper method (same units as input)
    - freq: frequencies corresponding to the spectrum
    - fft_vals: amplitude spectrum from FFT (same units as input)
    """
    # PSD (amplitude²/Hz)	Amplitude (unit)	amplitude = sqrt(psd * df)

    # Remove mean
    data = data - np.mean(data)

    # Pad data to next power of 2
    N_orig = len(data)
    D = 2 ** math.ceil(math.log2(N_orig))
    data = np.pad(data, (0, D - N_orig), mode='constant')
    N = len(data)

    # Apply Hann taper (window)
    taper = np.hanning(N)
    data_tapered = data * taper

    # Compute multitaper PSD
    freq, psd, _ = tsa.multi_taper_psd(data, 1 / delta, adaptive=True, jackknife=False, low_bias=True)
    df = freq[1] - freq[0]  # Frequency bin width

    # Convert PSD to amplitude spectrum
    amplitude = np.sqrt(psd * df)

    # Compute FFT amplitude spectrum for comparison
    fft_vals = (2.0 / N) * np.abs(np.fft.rfft(data_tapered))
    fft_vals[0] = fft_vals[0] / 2
    if N % 2 == 0:
        fft_vals[-1] = fft_vals[-1] / 2

    return amplitude, freq, fft_vals


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


def envelope(data, sampling_rate):

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


def downsample_trace(trace, factor=10):
    """
    Downsamples an ObsPy Trace by selecting every `factor`-th sample,
    adjusting the metadata to maintain the correct time length.

    Parameters:
        trace (obspy.Trace): Input trace to downsample.
        factor (int): Downsampling factor.

    Returns:
        obspy.Trace: Downsampled trace with corrected metadata.
    """
    original_npts = trace.stats.npts
    downsampled_data = trace.data[::factor]

    # Compute new stats
    new_sampling_rate = trace.stats.sampling_rate / factor
    new_npts = len(downsampled_data)

    # Create new Trace with updated metadata
    new_trace = Trace(data=downsampled_data, header=trace.stats)
    new_trace.stats.sampling_rate = new_sampling_rate
    new_trace.stats.npts = new_npts
    len_time = len(new_trace.times())
    Trace.data = trace.data[0:len_time]

    return new_trace



def __whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
    for j in index:
        den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
        data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den
    return data_f_whiten


# override function with jit decorator if numba is installed
if __use_numba:
    __whiten_aux = jit(nopython=True, parallel=True)(__whiten_aux)


def whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
    return __whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)


def whiten(tr, freq_width=0.05, taper_edge=True):

    """"
    freq_width: Frequency smoothing windows [Hz] / both sides
    taper_edge: taper with cosine window  the low frequencies

    return: whithened trace (Phase is not modified)
    """""

    fs = tr.stats.sampling_rate
    N = tr.count()
    D = 2 ** math.ceil(math.log2(N))
    freq_res = 1 / (D / fs)
    # N_smooth = int(freq_width / (2 * freq_res))
    N_smooth = int(freq_width / (freq_res))

    if N_smooth % 2 == 0:  # To have a central point
        N_smooth = N_smooth + 1
    else:
        pass

    # avarage_window_width = (2 * N_smooth + 1) #Denominador
    avarage_window_width = (N_smooth + 1)  # Denominador
    half_width = int((N_smooth + 1) / 2)  # midpoint
    half_width_pos = half_width - 1

    # Prefilt
    tr.detrend(type='simple')
    tr.taper(max_percentage=0.05)

    # ready to whiten
    data = tr.data
    data_f = np.fft.rfft(data, D)
    freq = np.fft.rfftfreq(D, 1. / fs)
    N_rfft = len(data_f)
    data_f_whiten = data_f.copy()
    index = np.arange(0, N_rfft - half_width, 1)

    data_f_whiten = whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)

    # Taper (optional) and remove mean diffs in edges of the frequency domain

    wf = (np.cos(np.linspace(np.pi / 2, np.pi, half_width)) ** 2)

    if taper_edge:

        diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])) * wf)

    else:

        diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])))

    diff_mean2 = np.abs(
        np.mean(np.abs(data_f[(N_rfft - half_width):])) - np.mean(np.abs(data_f_whiten[(N_rfft - half_width):])))

    if taper_edge:

        data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean)*wf  # First part of spectrum tapered
    else:

        data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean)


    data_f_whiten[(N_rfft - half_width):] = (data_f[(N_rfft - half_width):]) / diff_mean2  # end of spectrum
    data = np.fft.irfft(data_f_whiten)
    data = data[0:N]
    tr.data = data

    return tr


# def whiten_old(tr, freqmin, freqmax):
#     nsamp = tr.stats.sampling_rate
#
#     n = len(tr.data)
#     if n == 1:
#         return tr
#     else:
#         frange = float(freqmax) - float(freqmin)
#         nsmo = int(np.fix(min(0.01, 0.5 * (frange)) * float(n) / nsamp))
#         f = np.arange(n) * nsamp / (n - 1.)
#         JJ = ((f > float(freqmin)) & (f < float(freqmax))).nonzero()[0]
#
#         # signal FFT
#         FFTs = np.fft.fft(tr.data)
#         FFTsW = np.zeros(n) + 1j * np.zeros(n)
#
#         # Apodization to the left with cos^2 (to smooth the discontinuities)
#         smo1 = (np.cos(np.linspace(np.pi / 2, np.pi, nsmo + 1)) ** 2)
#         FFTsW[JJ[0]:JJ[0] + nsmo + 1] = smo1 * np.exp(1j * np.angle(FFTs[JJ[0]:JJ[0] + nsmo + 1]))
#
#         # boxcar
#         FFTsW[JJ[0] + nsmo + 1:JJ[-1] - nsmo] = np.ones(len(JJ) - 2 * (nsmo + 1)) \
#                                                 * np.exp(1j * np.angle(FFTs[JJ[0] + nsmo + 1:JJ[-1] - nsmo]))
#
#         # Apodization to the right with cos^2 (to smooth the discontinuities)
#         smo2 = (np.cos(np.linspace(0, np.pi / 2, nsmo + 1)) ** 2)
#         espo = np.exp(1j * np.angle(FFTs[JJ[-1] - nsmo:JJ[-1] + 1]))
#         FFTsW[JJ[-1] - nsmo:JJ[-1] + 1] = smo2 * espo
#
#         whitedata = 2. * np.fft.ifft(FFTsW).real
#
#         tr.data = np.require(whitedata, dtype="float32")
#
#         return tr


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
# Numba-accelerated 1D bilateral filter
@jit(nopython=True, parallel=True)
def bilateral_filter_1d_numba(signal, window_size, sigma_t, sigma_x):
    """
    Apply a bilateral filter on a 1D signal with Numba acceleration.

    :param signal: Input 1D signal (trace data)
    :param window_size: Size of the window in samples (odd number)
    :param sigma_t: Temporal standard deviation (in samples)
    :param sigma_x: Spatial standard deviation (signal amplitude)

    :return: Filtered signal
    """
    filtered_signal = np.zeros_like(signal)
    half_window = window_size // 2

    # Iterate over the signal and apply the bilateral filter
    for i in range(len(signal)):
        # Define the window boundaries
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, len(signal))

        # Extract the local window of signal values
        local_window = signal[start:end]

        # Compute the temporal weights (Gaussian in time domain)
        time_weights = np.exp(-0.5 * ((np.arange(len(local_window)) - half_window) / sigma_t) ** 2)

        # Compute the spatial weights (Gaussian in amplitude domain)
        amplitude_weights = np.exp(-0.5 * ((local_window - signal[i]) / sigma_x) ** 2)

        # Combine the weights
        weights = time_weights * amplitude_weights
        weights /= weights.sum()  # Normalize weights to sum to 1

        # Apply the filter (weighted average)
        filtered_signal[i] = np.sum(local_window * weights)

    return filtered_signal

@njit
def tv_denoise_numba(signal, weight=0.1, max_iter=100, tol=1e-4, step_size=0.1):
    """
    Total Variation (TV) denoising using gradient descent with Numba.

    Args:
        signal (np.array): Input 1D noisy signal.
        weight (float): Regularization strength (higher = smoother).
        max_iter (int): Maximum number of iterations.
        tol (float): Stopping criterion (based on L2 norm).
        step_size (float): Gradient descent step size.

    Returns:
        np.array: Denoised signal.
    """
    x = signal.copy()
    n = len(signal)

    for _ in range(max_iter):
        x_old = x.copy()

        # Gradient of the fidelity term
        grad_data = x - signal

        # Gradient of total variation term
        grad_tv = np.zeros_like(x)
        for i in range(1, n - 1):
            dx_prev = x[i] - x[i - 1]
            dx_next = x[i + 1] - x[i]
            grad_tv[i] = -np.sign(dx_prev) + np.sign(dx_next)

        # Handle endpoints
        grad_tv[0] = -np.sign(x[1] - x[0])
        grad_tv[-1] = np.sign(x[-1] - x[-2])

        # Gradient update step
        x -= step_size * (grad_data + weight * grad_tv)

        # Convergence check
        diff = np.sum((x - x_old) ** 2)
        if np.sqrt(diff) < tol:
            break

    return x

def smoothing(tr, type='gaussian', k=5, fwhm=0.05):
    # k window size in seconds
    # fwhm in seconds
    # Use a small FWHM, e.g., 0.01 – 0.05 sec --> Remove small noise but preserve sharpevents,
    # yields a narrow Gaussian kernel, which has minimal blur
    # Use a moderate FWHM, e.g., 0.1 – 0.5 sec, Works well for balancing denoising and detail preservation
    # Use a larger FWHM, e.g., 0.5 – 1.5 sec or more, effectively acts like a lowpass filter in time domain
    if type == 'mean':
        # new fast and simple implementation
        k = int(k * tr.stats.sampling_rate)
        kernel = np.ones(k) / k
        tr.data = np.convolve(tr.data, kernel, mode='same')

        # k = int(k * tr.stats.sampling_rate)
        #
        # # initialize filtered signal vector
        # filtsig = np.zeros(n)
        # for i in range(k, n - k - 1):
        #     # each point is the average of k surrounding points
        #     # print(i - k,i + k)
        #     filtsig[i] = np.mean(tr.data[i - k:i + k])
        #
        # tr.data = filtsig

    if type == 'gaussian':
        # new fast and simple implementation

        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_samples = sigma * tr.stats.sampling_rate
        tr.data = gaussian_filter1d(tr.data, sigma=sigma_samples)

        # sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        # padded = np.pad(tr.data, pad_width=int(3 * sigma), mode='reflect')
        # smoothed = gaussian_filter1d(padded, sigma=sigma)
        # tr.data = smoothed[int(3 * sigma):-int(3 * sigma)]

        # ## create Gaussian kernel
        # # full-width half-maximum: the key Gaussian parameter in seconds
        # # normalized time vector in seconds
        # k = int(k * tr.stats.sampling_rate)
        # fwhm = int(fwhm * tr.stats.sampling_rate)
        # gtime = np.arange(-k, k)
        # # create Gaussian window
        # gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)
        # # compute empirical FWHM
        #
        # pstPeakHalf = k + np.argmin((gauswin[k:] - .5) ** 2)
        # prePeakHalf = np.argmin((gauswin - .5) ** 2)
        # # empFWHM = gtime[pstPeakHalf] - gtime[prePeakHalf]
        # # show the Gaussian
        # # plt.plot(gtime/tr.stats.sampling_rate,gauswin)
        # # plt.plot([gtime[prePeakHalf],gtime[pstPeakHalf]],[gauswin[prePeakHalf],gauswin[pstPeakHalf]],'m')
        # # then normalize Gaussian to unit energy
        # gauswin = gauswin / np.sum(gauswin)
        # # implement the filter
        # # initialize filtered signal vector
        # filtsigG = copy.deepcopy(tr.data)
        # # implement the running mean filter
        # for i in range(k + 1, n - k - 1):
        #     # each point is the weighted average of k surrounding points
        #     filtsigG[i] = np.sum(tr.data[i - k:i + k] * gauswin)
        #
        # tr.data = filtsigG

    if type == "adaptive":
        window_length =int(k*tr.stats.sampling_rate)
        if window_length % 2 == 0:
            window_length += 1  # Ensure odd window length
        tr.data = savgol_filter(tr.data, window_length, polyorder=3, mode='nearest')

    # if type == "adaptive":
    #     bilateral_filter
    #     sigma_t = 0.05
    #     sigma_x = 0.1
    #     sr = tr.stats.sampling_rate
    #     window_size = int(k * sr)
    #
    #     # Ensure that window size is odd
    #     if window_size % 2 == 0:
    #         window_size += 1
    #
    #     # Convert sigma_t to samples
    #     sigma_t_samples = sigma_t * sr
    #
    #     # Apply the bilateral filter to the trace data
    #     tr.data = bilateral_filter_1d_numba(tr.data, window_size, sigma_t_samples, sigma_x)

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

def wiener_filter(tr, time_window, noise_power):
    data = tr.data


    if time_window == 0 and noise_power == 0:

       denoise = scipy.signal.wiener(data, mysize=None, noise=None)
       tr.data = denoise

    elif time_window!= 0 and noise_power == 0:

        denoise = scipy.signal.wiener(data, mysize= int(time_window*tr.stats.sampling_rate), noise=None)
        tr.data = denoise

    elif time_window == 0 and noise_power !=0:

         noise = noise_power * np.std(data)
         noise = int(noise)
         denoise = scipy.signal.wiener(data, mysize=None, noise=noise)
         tr.data = denoise

    elif time_window != 0 and noise_power != 0:

         noise = noise_power * np.std(data)
         noise = int(noise)
         denoise = scipy.signal.wiener(data, mysize=int(time_window * tr.stats.sampling_rate), noise=noise)
         tr.data = denoise

    return tr


def __hampel_aux(input_series, window_size, size, n_sigmas):

    k = 1.4826  # scale factor for Gaussian distribution
    #indices = []
    new_series = input_series.copy()
    # possibly use np.nanmedian
    for i in range((window_size), (size - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            #indices.append(i)

    return new_series


if __use_numba:
    __hampel_aux = jit(nopython=True, parallel=True)(__hampel_aux)


def hampel_aux(input_series, window_size, size, n_sigmas):
    return __hampel_aux(input_series, window_size, size, n_sigmas)

def hampel(tr, window_size, n_sigmas=3):

    """
        Median absolute deviation (MAD) outlier in Time Series
        :param ts: a trace obspy object representing the timeseries
        :param window_size: total window size in seconds
        :param n: threshold, default is 3 (Pearson's rule)
        :return: Returns the corrected timeseries
        """

    size = tr.count()
    input_series = tr.data
    window_size = int(window_size*tr.stats.sampling_rate)
    tr.data = hampel_aux(input_series, window_size, size, n_sigmas)

    return tr


def spectral_integration(trace, pad_to_pow2=True, taper_type='cosine', taper_pct=0.05):
    """
    Integrate a seismic signal in the frequency domain with preprocessing.

    Parameters:
        trace : ObsPy Trace object
        lp_freq : Lowpass filter cutoff frequency (Hz) to suppress drift #No implemented
        pad_to_pow2 : Zero-pad to next power of 2 (improves FFT efficiency)
        taper_type : 'cosine', 'hann', etc. taper before FFT
        taper_pct : Taper percentage (e.g., 0.05 = 5%)

    Returns:
        Modified trace (in-place)
    """
    # Copy original data for safety
    tr = trace.copy()

    # Remove mean and apply taper
    tr.detrend(type='demean')
    tr.taper(max_percentage=taper_pct, type=taper_type)

    # Original parameters
    n_orig = tr.stats.npts
    dt = tr.stats.delta

    # Zero-pad to next power of 2 for spectral resolution
    n_pad = 2 ** int(np.ceil(np.log2(n_orig))) if pad_to_pow2 else n_orig
    f = np.fft.rfftfreq(n_pad, d=dt)

    # FFT and scale spectrum
    spectrum = np.fft.rfft(tr.data, n=n_pad)
    scale = np.ones_like(f, dtype=np.complex128)
    scale[1:] = 1 / (1j * 2 * np.pi * f[1:])
    scale[0] = 0.0  # remove DC

    # Integrate in frequency domain
    spectrum_integrated = spectrum * scale
    tr.data = np.fft.irfft(spectrum_integrated, n=n_pad)[:n_orig]

    # # Final lowpass filter to remove accumulated drift (e.g., from low-f noise)
    # tr.filter("lowpass", freq=lp_freq, corners=4, zerophase=True)

    return tr

def spectral_derivative(trace, taper_pct=0.05, taper_type="cosine", pad_to_pow2=True):
    """
    Compute the first derivative of a signal in the frequency domain.

    Parameters:
        trace : ObsPy Trace object
        hp_freq : Optional highpass cutoff frequency (Hz) to remove amplified high-frequency noise
        taper_pct : Taper percentage (e.g., 0.05 = 5%)
        taper_type : Taper window type ('cosine', 'hann', etc.)
        pad_to_pow2 : Whether to zero-pad to next power of 2 before FFT

    Returns:
        Modified trace with differentiated signal
    """
    tr = trace.copy()

    # Demean and taper
    tr.detrend(type="demean")
    tr.taper(max_percentage=taper_pct, type=taper_type)

    # Set up time parameters
    n_orig = tr.stats.npts
    dt = tr.stats.delta
    fs = 1 / dt
    n_pad = 2 ** int(np.ceil(np.log2(n_orig))) if pad_to_pow2 else n_orig

    # Frequency array
    freqs = np.fft.rfftfreq(n_pad, d=dt)

    # FFT of the signal
    spectrum = np.fft.rfft(tr.data, n=n_pad)

    # Differentiate in frequency domain: d/dt <-> i·2πf
    spectrum_diff = (1j * 2 * np.pi * freqs) * spectrum
    tr.data = np.fft.irfft(spectrum_diff, n=n_pad)[:n_orig]

    # Optional post-highpass filter to reduce HF noise if needed
    # if hp_freq is not None:
    #     tr.filter("highpass", freq=hp_freq, corners=4, zerophase=True)

    return tr


# def hampel_old(tr, window_size=5, n=3, imputation=True):
#
#     """
#     Median absolute deviation (MAD) outlier in Time Series
#     :param ts: a pandas Series object representing the timeseries
#     :param window_size: total window size will be computed as 2*window_size + 1
#     :param n: threshold, default is 3 (Pearson's rule)
#     :param imputation: If set to False, then the algorithm will be used for outlier detection.
#         If set to True, then the algorithm will also imput the outliers with the rolling median.
#     :return: Returns the outlier indices if imputation=False and the corrected timeseries if imputation=True
#     """
#
#
#     window_size = int(window_size*tr.stats.sampling_rate)
#
#     # Copy the Series object. This will be the cleaned timeserie
#     ts = pd.Series(tr.data)
#     ts_cleaned = ts.copy()
#
#     # Constant scale factor, which depends on the distribution
#     # In this case, we assume normal distribution
#     k = 1.4826
#
#     rolling_ts = ts_cleaned.rolling(window_size * 2, center=True)
#     rolling_median = rolling_ts.median().fillna(method='bfill').fillna(method='ffill')
#     rolling_sigma = k * (rolling_ts.apply(median_absolute_deviation).fillna(method='bfill').fillna(method='ffill'))
#
#     outlier_indices = list(np.array(np.where(np.abs(ts_cleaned - rolling_median) >= (n * rolling_sigma))).flatten())
#
#     if imputation:
#         ts_cleaned[outlier_indices] = rolling_median[outlier_indices]
#         tr.data = ts_cleaned.array.to_numpy()
#
#     return tr
