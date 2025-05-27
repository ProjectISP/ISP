
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wigner_toolkit.py
@author Luc Kusters
@date 17-03-2022
repository: https://github.com/ljbkusters/python-wigner-distribution
"""

import collections

from numba import njit, prange
from scipy import signal, linalg, ndimage
import numpy as np
from scipy.fftpack import hilbert


@njit(parallel=True)
def compute_wvd_kernel(x):
    N = x.shape[0]
    wvd = np.zeros((N, N), dtype=np.complex128)

    for t in prange(N):
        for tau in range(-min(t, N - t - 1), min(t, N - t - 1) + 1):
            i1 = t + tau
            i2 = t - tau
            if 0 <= i1 < N and 0 <= i2 < N:
                wvd[t, tau] = x[i1] * np.conj(x[i2])
    return wvd

def fast_wigner_distribution(x, sample_frequency=None, t_0=0, t_1=1, flip_frequency_range=True):
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("Input must be 1D.")
    if np.isrealobj(x):
        x = hilbert(x)

    N = x.shape[0]
    kernel = compute_wvd_kernel(x)

    # Perform FFT along tau axis (columns) to get Wigner distribution
    wvd = np.real(np.fft.fft(kernel, axis=1)).T  # transpose for [freq, time]

    if sample_frequency is None:
        sample_frequency = N / (t_1 - t_0)

    max_frequency = sample_frequency / 2

    if flip_frequency_range:
        wvd = np.flipud(wvd)

    return wvd, max_frequency

def wigner_distribution(x, use_analytic=True, sample_frequency=None,
                        t_0=0, t_1=1, flip_frequency_range=True):
    """Discrete Pseudo Wigner Ville Distribution based on [1]

    Args:
        x, array like, signal input array of length N
        use_analytic, bool, whether or not to use analytic associate of input
            data x by default set to True
        sample_frequency, sampling frequency
        t_0, time at which the first sample was recorded
        t_1, time at which the last sample was recorded
        flip_frequency_range, flip the data in about the time axis such that
            the minimum frequency is in the left bottom corner.

    Returns:
        wigner_distribution, N x N matrix
        max_frequency, a positive number, maximum of the frequency range

    References:
        [1] T. Claasen & W. Mecklenbraeuker, The Wigner Distribution -- A Tool
        For Time-Frequency Signal Analysis, Phillips J. Res. 35, 276-300, 1980
    """

    # Ensure the input array is a numpy array
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    # Compute the autocorrelation function matrix
    if x.ndim != 1:
        raise ValueError("Input data should be one dimensional time series.")
    # Use analytic associate if set to True
    if use_analytic:
        if all(np.isreal(x)):
            x = signal.hilbert(x)
        else:
            raise RuntimeError("Keyword 'use_analytic' set to True but signal"
                               " is of complex data type. The analytic signal"
                               " can only be computed if the input signal is"
                               " real valued.")

    # calculate the wigner distribution
    N = x.shape[0]
    bins = np.arange(N)
    indices = linalg.hankel(bins, bins + N - (N % 2))

    padded_x = np.pad(x, (N, N), 'constant')
    wigner_integrand = \
        padded_x[indices+N] * np.conjugate(padded_x[indices[::, ::-1]])

    wigner_distribution = np.real(np.fft.fft(wigner_integrand, axis=1)).T

    # calculate sample frequency
    if sample_frequency is None:
        sample_frequency = N / (t_1 - t_0)

    # calculate frequency range
    if use_analytic:
        max_frequency = sample_frequency/2
    else:
        max_frequency = sample_frequency/4

    # flip the frequency range
    if flip_frequency_range:
        wigner_distribution = wigner_distribution[::-1, ::]

    return wigner_distribution, max_frequency


def interference_reduced_wigner_distribution(
        wigner_distribution, number_smoothing_steps=16,
        t_filt_max_percentage=None, f_filt_max_percentage=None):
    """Method for reducing interference terms based on [1]

    Params:
        wigner_distribution, array like, N x N discrete wigner distribution
        matrix

    Returns:
        interference reduced wigner distribution, N x N ndarray

    Uses a method for interference reduction based on Pikula et al. [1].
    The method works by executing multiple smoothings using a gaussian
    filter, in this implementation using the scipy.ndimage module. The
    optimal smoothing per time-frequency bin is then chosen. Pikula et al.
    [1] goes into more detail on how this optimal smoothing can be chosen.

    The output is then a distribution which contains mainly autoterms with
    strongly suppressed interference terms, better representing the actual
    signal that is present. This, however, destroys many of the
    distributions' mathematical properties, and should only serve as an
    analysis tool for autoterms.

    References:
        [1] Pikula, Stanislav & Beneš, Petr. (2020). A New Method for
        Interference Reduction in the Smoothed Pseudo Wigner-Ville
        Distribution. International Journal on Smart Sensing and
        Intelligent Systems. 7. 1-5. 10.21307/ijssis-2019-101.
    """
    # Ensure the input array is a numpy array
    if not isinstance(wigner_distribution, np.ndarray):
        wigner_distribution = np.asarray(wigner_distribution)
    # Compute the autocorrelation function matrix
    if wigner_distribution.ndim != 2:
        raise ValueError("Input data should be a two dimensional discrete"
                         " wigner distribution.")
    N_f, N_t = wigner_distribution.shape
    if t_filt_max_percentage is None:
        t_filt_max_percentage = 1/(2*number_smoothing_steps)
    if f_filt_max_percentage is None:
        f_filt_max_percentage = 1/(2*number_smoothing_steps)
    print(t_filt_max_percentage, f_filt_max_percentage)

    t_filter_widths = \
        np.linspace(0, N_t * t_filt_max_percentage, number_smoothing_steps)
    f_filter_widths = \
        np.linspace(N_f * f_filt_max_percentage, 0, number_smoothing_steps)

    # filter at various filtration widths
    smoothed_wigner_distributions = \
        np.zeros((number_smoothing_steps, N_f, N_t))
    for i, (f_fw, t_fw) in enumerate(zip(t_filter_widths, f_filter_widths)):
        smoothed_wigner_distributions[i] = \
            ndimage.gaussian_filter(wigner_distribution, sigma=(f_fw, t_fw))

    # differential analysis per time-frequency bin
    first_derivative = np.diff(smoothed_wigner_distributions, axis=0)
    second_derivative = np.diff(first_derivative, axis=0)
    smoothing_index_best_guess = np.argmax(second_derivative, axis=0)

    # choose smoothing per time-frequency bin
    smoothing_steps, f_dim, t_dim = smoothed_wigner_distributions.shape
    interference_reduced_wigner_distribution = \
        smoothed_wigner_distributions[
                smoothing_index_best_guess,
                np.arange(f_dim)[::, np.newaxis],
                np.arange(t_dim)[np.newaxis, ::]]

    return interference_reduced_wigner_distribution


__SampleBase = collections.namedtuple(
            "__SampleBase",
            ("samples", "number_of_samples", "sample_frequency", "t0", "t1")
        )


class TimeSamples(__SampleBase):
    """Generalized time samples data structure

    Contains the samples in the form of a numpy.ndarray, the number of samples,
    the sample frequency, the start time, and the end time.

    Also contains factory methods for defining the time samples based on sample
    rate or number of samples.
    """

    @classmethod
    def from_sample_frequency(cls, sample_frequency, t0=0., t1=1.):
        time_delta = t1 - t0
        number_of_samples = int(float(sample_frequency) / float(time_delta))
        time_samples = np.linspace(t0, t1, number_of_samples)
        return cls(time_samples, number_of_samples, sample_frequency, t0, t1)

    @classmethod
    def from_sample_number(cls, number_of_samples, t0=0.1, t1=1.):
        time_delta = t1 - t0
        sample_frequency = float(float(number_of_samples) / float(time_delta))
        time_samples = np.linspace(t0, t1, number_of_samples)
        return cls(time_samples, number_of_samples, sample_frequency, t0, t1)


# signal sampled at 1024 Hz for a duration of 1 s
DEFAULT_TIME_SAMPLES = TimeSamples.from_sample_frequency(sample_frequency=1024)


def sine_wave(time_samples, frequency) -> np.ndarray:
    """Wrapper for numpy.sin to generate pure sine"""
    omega = 2 * np.pi * frequency
    return np.sin(time_samples * omega)


def chirp(time_samples, start_frequency,
          end_frequency, time_end=None) -> np.ndarray:
    """Wrapper for scipy.signal.chirp to generate linear chirps"""
    if time_end is None:
        time_end = np.max(time_samples.samples)
    return signal.chirp(time_samples.samples, f0=start_frequency,
                              t1=time_end, f1=end_frequency)


def gaussian(x, mean, std, height=1., bias=0.) -> np.ndarray:
    """Gaussian function

    Parameters:
        x, array like, input space
        mean, mean of gaussian
        std, standard deviation of gaussian
        heigt, max height of gaussian (relative to bias), by default 1.
        bias, bias of gaussian, by default 0.
    """
    exponential = np.exp(-0.5 * np.power((x - mean) / std, 2))
    return height * exponential + bias


def gaussian_kernel_sine(time_samples, frequency, envelope_mean, envelope_std)\
        -> np.ndarray:
    """Guassian windowed sine

    Produces a guassian kernel in the time-frequency domain of a
    Wigner distribution.
    """
    sine = sine_wave(time_samples, frequency)
    envelope = gaussian(time_samples, envelope_mean, envelope_std)
    return sine * envelope