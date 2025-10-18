#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
whitenning_test

"""

from isp.ant.signal_processing_tools import noise_processing_horizontals
from obspy import read
from isp.seismogramInspector.signal_processing_advanced import spectrumelement
import matplotlib.pyplot as plt


def plot_spectrum(freq, spec, amplitude):
    fig, ax1 = plt.subplots(figsize=(6, 6))

    ax1.loglog(freq, amplitude, linewidth=1.0, alpha=0.75, color='orange', label= " Amplitude")
    ax1.loglog(freq, spec, linewidth=1.0, color='steelblue', label= " Multitaper Amplitude")
    ax1.frequencies = freq
    ax1.spectrum = spec
    # ax1.fill_between(freq, jackknife_errors[0][:], jackknife_errors[1][:], facecolor="0.75",
    #                 alpha=0.5, edgecolor="0.5")
    #ax1.set_ylim(spec.min() / 10.0, spec.max() * 100.0)
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency [Hz]')
    plt.grid(True, which="both", ls="-", color='grey')
    plt.legend()
    plt.show()


tr_N = read('/Volumes/NO NAME/test_EGFs/data_test/8J.UP09..HH1.D.2021.202')[0]
tr_E = read('/Volumes/NO NAME/test_EGFs/data_test/8J.UP09..HH2.D.2021.202')[0]


process_horizontals = noise_processing_horizontals(tr_N, tr_E)
process_horizontals.normalize(norm_win=25, norm_method="running avarage")
process_horizontals.whiten_new_band(freq_width=0.02, fmin=0.01, fmax=2.5)
process_horizontals.tr_N.detrend(type='simple')
process_horizontals.tr_E.detrend(type='simple')
process_horizontals.tr_E.taper(max_percentage=0.05)
process_horizontals.tr_N.taper(max_percentage=0.05)

# process_horizontals.tr_N.filter(type="bandpass", freqmin=0.02,
#                   freqmax=0.1,
#                   zerophase=True, corners=4)
# process_horizontals.tr_E.filter(type="bandpass", freqmin=0.02,
#                   freqmax=0.1,
#                   zerophase=True, corners=4)
data = process_horizontals.tr_N.data
delta = 1 / 5
[spec, freq, amplitude] = spectrumelement(data, delta, id)

plot_spectrum(freq, spec, amplitude)