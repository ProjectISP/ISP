#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 02:01:51 2019

@author: robertocabieces
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:44:06 2019

@author: robertocabieces
"""

##########CWT with convolution of bamk of atoms######


from obspy.signal.filter import lowpass
import numpy as np

def cwt_fast(data,ba,nConv,frex,half_wave, fs):
    npts=len(data)
    ##FFT data
    cf = []
    tf = []
    dataX = np.fft.fft(data,nConv)
##loop over frequencies
    for fi in range(len(frex)):
            cmwX=ba[fi,:]
            #cwt = np.fft.irfft(np.multiply(cmwX,dataX))
            cwt = np.fft.ifft(np.multiply(cmwX,dataX))
            #end =len(cwt)
            cwt = cwt[int(half_wave+1):npts+int(half_wave+1)]
            d = np.diff(np.log10(np.abs(cwt)**2))
            cf.append(d)
            tf.append(cwt)

    tf=np.asarray(tf)
    sc=np.mean(tf, axis=0)
    cf = np.mean(cf, axis = 0)
    cf = cf - np.mean(cf)
    n = len(cf)
    window = np.hamming(n)
    cf = window * cf
    cf = lowpass(cf, 0.2, fs, corners=4, zerophase=True)
    cf = window * cf
    return cf,sc,tf



class CwtFast:
    def __init__(self, data, ba, nConv, frex, half_wave, fs):

        self.data = data
        self.ba = ba
        self.nConv = nConv
        self.frex = frex
        self.half_wave = half_wave
        self.fs = fs

    # def cwt_fast(self):
    #     npts = len(self.data)
    #     ##FFT data
    #     cf = []
    #     tf = []
    #     dataX = np.fft.fft(self.data, self.nConv)
    #     ##loop over frequencies
    #     for fi in range(len(self.frex)):
    #         cmwX = self.ba[fi, :]
    #         # cwt = np.fft.irfft(np.multiply(cmwX,dataX))
    #         cwt = np.fft.ifft(np.multiply(cmwX, dataX))
    #         # end =len(cwt)
    #         cwt = cwt[int(self.half_wave + 1):npts + int(self.half_wave + 1)]
    #
    #         tf.append(cwt)
    #
    #     tf = np.asarray(tf)
    #
    #     return tf
    # def charachteristic_function_kurt(self, tr, window_size_seconds=5):
    #
    #
    #
    #     pow_scalogram = np.abs(self._tf)**2
    #     kurtosis_values, time_vector = self.conventional_kurtosis(pow_scalogram, window_size_seconds=window_size_seconds,
    #                                                               sampling_rate=self._sample_rate)
    #
    #     time_vector_resample = np.linspace(time_vector[0], time_vector[-1], int(time_vector[-1]*self._sample_rate))
    #
    #     kurtosis_values_resample = np.interp(time_vector_resample, time_vector, kurtosis_values)
    #
    #     # Create Trace object with the synthetic data
    #     tr_kurt = Trace(data=kurtosis_values_resample)
    #
    #     # Set trace metadata
    #     tr_kurt.stats.station = self.trace.stats.station  # Station name
    #     tr_kurt.stats.network = self.trace.stats.network # Network code
    #     tr_kurt.stats.channel = self.trace.stats.channel  # Channel code
    #     tr_kurt.stats.starttime = self.trace.stats.starttime + time_vector_resample[0] # Set to current time as start time
    #     tr_kurt.stats.sampling_rate = self.trace.stats.sampling_rate
    #
    #     tr_kurt.detrend(type="simple")
    #     tr_kurt.taper(type="blackman", max_percentage=0.05)
    #
    #     tr_kurt.filter(type='lowpass', freq=0.15, zerophase=True, corners=4)
    #
    #     tr_kurt.detrend(type="simple")
    #     tr_kurt.taper(type="blackman", max_percentage=0.05)
    #
    #     return tr_kurt
    #
    # def conventional_kurtosis(self, data, window_size_seconds, sampling_rate):
    #
    #     n = data.shape[1]
    #     window_size_samples = int(window_size_seconds * sampling_rate)
    #     slide = int(sampling_rate/2)
    #
    #     # Call the Numba-accelerated helper function
    #     kurtosis_values = self._conventional_kurtosis_helper(data, window_size_samples, slide, n)
    #
    #     # Create time vector
    #     time_vector = np.linspace(0, int((n - window_size_samples) / sampling_rate), len(kurtosis_values)) + int(window_size_seconds)
    #     #time_vector = time_vector[0:-1]
    #     # kurtosis_values = np.abs(np.diff(kurtosis_values))
    #     return kurtosis_values, time_vector
    #
    # def _conventional_kurtosis_helper(self, data, window_size_samples, slide, n):
    #     kurtosis_values = []
    #
    #     # Loop through the data with the sliding window
    #     for i in range(0, n - window_size_samples + 1, slide):
    #         window_data = data[:, i:i + window_size_samples]  # Extract data in window
    #
    #         # Compute mean for the window
    #         mean = np.mean(window_data)
    #
    #         # Compute variance (second central moment)
    #         variance = np.mean((window_data - mean) ** 2)
    #
    #         # Compute fourth central moment
    #         fourth_moment = np.mean((window_data - mean) ** 4)
    #
    #         # Compute kurtosis (excess kurtosis)
    #         if variance > 1e-10:  # Ensure variance is not effectively zero
    #             kurtosis = (fourth_moment / (variance ** 2)) - 3  # Subtract 3 for excess kurtosis
    #
    #             # Check for extremely large kurtosis and cap it
    #             if not np.isfinite(kurtosis):  # Handles inf and NaN cases
    #                 kurtosis = 0.0
    #             elif abs(kurtosis) > 1e6:  # Prevent unreasonably large values
    #                 kurtosis = np.sign(kurtosis) * 1e6
    #         else:
    #             kurtosis = 0.0  # Set kurtosis to 0 if variance is effectively zero
    #
    #         # Append result to list
    #         kurtosis_values.append(kurtosis)
    #
    #     return np.array(kurtosis_values)
   
