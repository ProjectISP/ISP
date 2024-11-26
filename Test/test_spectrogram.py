#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_spectrogram
"""

import math
import nitime.algorithms as tsa
import numpy as np
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt

def find_nearest(array, value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx,val

def MTspectrum(data, win, dt, linf, lsup, step_percentage=0.1):

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
        print(f"{(n + 1) * 100 / lim:.2f}% done")

        #data1 = data[n:nfft + n] - np.mean(data[n:nfft + n])  # Subtract mean
        data1 = data[n:nfft + n]
        data1 = data1 - np.mean(data1)
        freq, spec, _ = tsa.multi_taper_psd(data1, fs, adaptive=False, jackknife=False, low_bias=False)

        #
        S[:,idx] = spec

    value1, freq1 = find_nearest(freq, linf)
    value2, freq2 = find_nearest(freq, lsup)

    spectrum = S[value1:value2,:]

    t = np.linspace(0, len(data)/fs, spectrum.shape[0])
    f = np.linspace(linf, lsup, spectrum.shape[1])
    return spectrum, num_steps, t, f

if __name__ == "__main__":
    tr = read("/Users/admin/Documents/iMacROA/ISP/isp/examples/Earthquake_location_test/ES.EADA..HHZ.D.2015.260")[0]
    t1=UTCDateTime('2015-09-17T15:12:00')
    t2=UTCDateTime('2015/09/17 15:14:00')
    tr.trim(starttime=t1, endtime=t2)
    #tr.plot()
    win=3*tr.stats.sampling_rate
    dt = tr.stats.delta
    linf=1
    lsup=10
    max_log_spectrogram = 0
    min_log_spectrogram = -180
    spectrum, num_steps, t, f = MTspectrum(tr.data, win, dt, linf, lsup, step_percentage=0.1)
    log_spectrogram = 10. * np.log(spectrum / np.max(spectrum))
    t = np.linspace(0, tr.times()[-1], log_spectrogram.shape[1])
    f = np.linspace(linf, lsup, log_spectrogram.shape[0])
    x, y = np.meshgrid(t, f)
    log_spectrogram = np.clip(log_spectrogram, a_min=min_log_spectrogram, a_max=0)

    # plot
    fig, ax = plt.subplots()
    #plt.imshow(log_spectrogram)
    plt.contourf(x, y, log_spectrogram, axes_index=1, clabel="Power [dB]", vmin=min_log_spectrogram, vmax=max_log_spectrogram)
    plt.show()