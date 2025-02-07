# -*- coding: utf-8 -*-
"""
This file is part of Rfun, a toolbox for the analysis of teleseismic receiver
functions.

Copyright (C) 2020-2021 Andrés Olivar-Castaño

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
For questions, bug reports, or to suggest new features, please contact me at
olivar.ac@gmail.com.
"""

import os
import obspy
import obspy.signal.filter
import obspy.signal.rotate
import numpy as np
import math
import scipy.interpolate as scint
import dill as pickle
import cartopy.crs as ccrs
import shapefile as shp
import copy
import h5py
from pathlib import Path

from isp.receiverfunctions.definitions import ROOT_DIR, CONFIG_PATH

def read_preferences(file='rfun.conf', return_defaults=False):
    
    default_settings = {'ccp':{'appearance':{'include_stations':True,
                                             'plotting_method':'colored grid',
                                             'colormap':'viridis',
                                             'station_marker':'^',
                                             'station_marker_color':'#00FF00'},
                               'shapefiles':{'include':True,
                                             'path':None},
                               'computation':{'earth_model':'iasp91',
                                              'stacking_method':'weighted mean'}},
                        'rfs':{'appearance':{'line_color':'#000000',
                                             'line_width':0.5,
                                             'positive_fill_color':'#0000FF',
                                             'negative_fill_color':'#FF0000'},
                               'general_settings':{'normalize':True,
                                                   'w0':0.0,
                                                   'time_shift':5.0},
                               'computation_method':{'method':'Time-domain',
                                                     'method_settings':{'max_iters': 400,
                                                                        'min_deltaE':0.001}},
                               'stacking':{'ref_slowness':6.30}},
                        'hk':{'appearance':{'plotting_method':'colored grid',
                                            'colormap':'viridis',
                                            'line_color':'#FFFFFF',
                                            'ser_color':'#00FF00'},
                              'computation':{'semblance_weighting':True,
                                             'H_points':200,
                                             'k_points':200,
                                             'avg_vp':6.30},
                              'theoretical_atimes':{'ref_slowness':6.30,
                                                    'avg_vp':6.30}},
                        'map':{'appearance':{'include_stations':True,
                                             'plotting_method':'colored grid',
                                             'colormap':'viridis',
                                             'station_marker':'^',
                                             'station_marker_color':'#00FF00'},
                               'shapefiles':{'include':True,
                                             'path':None}}}
    
    if return_defaults:
        return default_settings
    
    try:
        settings = pickle.load(open(CONFIG_PATH, 'rb'))
    except FileNotFoundError:
        settings = default_settings
        pickle.dump(settings, open(CONFIG_PATH, "wb"))
    
    return settings

def read_hk_results_file(path):
    with open(path) as f:
        lines = f.readlines()

    result_dict = {}
    
    for line in lines[1:]:
        stnm = line.split(',')[0]
        lon = float(line.split(',')[1])
        lat = float(line.split(',')[2])
        H =  float(line.split(',')[4])
        min_H =  float(line.split(',')[5])
        max_H =  float(line.split(',')[6])
        k =  float(line.split(',')[7])
        min_k =  float(line.split(',')[8])
        max_k =  float(line.split(',')[9].strip('\n'))
    
        
        result_dict[stnm] = {"H":H, "k":k,
                             "loc":[lon, lat]}
    
    return result_dict

def read_shapefiles(path):
    shp_files = []

    for top_dir, sub_dir, files in os.walk(path):
        for file in files:
            full_path = os.path.join(top_dir, file)
            if full_path[-4:] == '.shp':
                shp_files.append(full_path)

    sfs = [shp.Reader(f) for f in shp_files]
    
    return sfs

def read_hdf5(path, mode="r"):
    file = h5py.File(path, mode)
    return file

def resize_colobar(event, figure, map_ax, cbar_ax):
    figure.canvas.draw()

    posn = map_ax.get_position()
    cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0,
                          0.04, posn.height])

def waterlevel_deconvolution(dcmp, scmp, delta, a, c, tshift, w0=0,
                             normalize=True):
    w0 = w0*2*np.pi
    max_len = np.maximum(len(dcmp), len(scmp))
    next_pow2 = 2**math.ceil(math.log(max_len, 2))

    zeroes = next_pow2 - len(dcmp)
    
    pdcmp = np.pad(dcmp, (0, zeroes), mode='constant')
    pscmp = np.pad(scmp, (0, zeroes), mode='constant')
    
    dfft = np.fft.rfft(pdcmp)
    sfft = np.fft.rfft(pscmp)
    freq = np.fft.rfftfreq(len(pdcmp), d=delta)
    
    # Langston, 1979
    num = dfft * np.conj(sfft)
    deno = np.maximum(sfft*np.conj(sfft), c*np.max(sfft*np.conj(sfft)))
    rf = np.fft.irfft((num/deno)* np.exp(-0.5*(freq-w0)**2/(a**2))/delta * np.exp(-1j * tshift * 2 * np.pi * freq))

    if normalize:
        rf = rf / np.max(np.abs(rf))
        
    return rf[:len(dcmp)]

###############################################################################
# https://github.com/trichter/rf/blob/master/rf/deconvolve.py #################
###############################################################################
from scipy.fftpack import fft, ifft, next_fast_len

def _gauss_filter(dt, nft, f0, waterlevel=None):
    """
    Gaussian filter with width f0
    :param dt: sample spacing in seconds
    :param nft: length of filter in points
    :param f0: Standard deviation of the Gaussian Low-pass filter,
        corresponds to cut-off frequency in Hz for a response value of
        exp(0.5)=0.607.
    :param waterlevel: waterlevel for eliminating very low values
        (default: no waterlevel)
    :return: array with Gaussian filter frequency response
    """
    f = np.fft.fftfreq(nft, dt)
    gauss_arg = -0.5 * (f/f0) ** 2
    if waterlevel is not None:
        gauss_arg = np.maximum(gauss_arg, waterlevel)
    return np.exp(gauss_arg)


def _phase_shift_filter(nft, dt, tshift):
    """
    Construct filter to shift an array to account for time before onset
    :param nft: number of points for fft
    :param dt: sample spacing in seconds
    :param tshift: time to shift by in seconds
    :return: shifted array
    """
    freq = np.fft.fftfreq(nft, d=dt)
    return np.exp(-2j * np.pi * freq * tshift)


def _apply_filter(x, filt):
    """
    Apply a filter defined in frequency domain to a data array
    :param x: array of data to filter
    :param filter: filter to apply in frequency domain,
        e.g. from _gauss_filter()
    :return: real part of filtered array
    """
    nfft = len(filt)
    xf = fft(x, n=nfft)
    xnew = ifft(xf*filt, n=nfft)
    return xnew.real

def deconv_iterative(rsp, src, dt, tshift=10, gauss=0.5, itmax=400,
                     minderr=0.001, mute_shift=False, normalize=0):
    """
    Iterative deconvolution.
    Deconvolve src from arrays in rsp.
    Iteratively construct a spike train based on least-squares minimzation of the
    difference between one component of an observed seismogram and a predicted
    signal generated by convolving the spike train with an orthogonal component
    of the seismogram.
    Reference: Ligorria, J. P., & Ammon, C. J. (1999). Iterative Deconvolution
    and Receiver-Function Estimation. Bulletin of the Seismological Society of
    America, 89, 5.
    :param rsp: either a list of arrays containing the response functions
        or a single array
    :param src: array with source function
    :param sampling_rate: sampling rate of the data
    :param tshift: delay time 0s will be at time tshift afterwards
    :param gauss: Gauss parameter (standard deviation) of the
        Gaussian Low-pass filter,
        corresponds to cut-off frequency in Hz for a response value of
        exp(-0.5)=0.607.
    :param itmax: limit on number of iterations/spikes to add
    :param minderr: stop iteration when the change in error from adding another
        spike drops below this threshold
    :param mute_shift: Mutes all samples at beginning of trace
        (lenght given by time shift).
        For `len(src)==len(rsp)` this mutes all samples before the onset.
    :param normalize: normalize all results so that the maximum of the trace
        with the supplied index is 1. Set normalize to None for no normalization.
    :return: (list of) array(s) with deconvolution(s)
    """
    sampling_rate = 1/dt
    RF_out = []
    it_out = []  # number of iterations each component uses
    rms_out = []

    r0 = rsp
    nfft = next_fast_len(2 * len(r0))
    rms = np.zeros(itmax)    # to store rms
    p0 = np.zeros(nfft)      # and rf for this component iteration

    gaussF = _gauss_filter(dt, nfft, gauss)  # construct and apply gaussian filter
    r_flt = _apply_filter(r0, gaussF)
    s_spec_flt = fft(src, nfft) * gaussF  # spectrum of source
    powerS = np.sum(ifft(s_spec_flt).real ** 2)  # power in the source for scaling
    powerR = np.sum(r_flt**2)  # power in the response for scaling
    rem_flt = copy.deepcopy(r_flt)  # thing to subtract from as spikes are added to p0

    it = 0
    sumsq_i = 1
    d_error = 100*powerR + minderr
    mute_min = len(r0)
    mute_max = nfft if mute_shift else nfft - int(tshift*sampling_rate)
    while np.abs(d_error) > minderr and it < itmax:  # loop iterations, add spikes
        rs = ifft(fft(rem_flt) * np.conj(s_spec_flt)).real  # correlate (what's left of) the num & demon, scale
        rs = rs / powerS / dt  # scale the correlation
        rs[mute_min:mute_max] = 0
        i1 = np.argmax(np.abs(rs))  # index for getting spike amplitude
        # note that ^abs there means negative spikes are allowed
        p0[i1] = p0[i1] + rs[i1]  # add the amplitude of the spike to our spike-train RF
        p_flt = ifft(fft(p0) * s_spec_flt).real * dt  # convolve with source

        rem_flt = r_flt - p_flt  # subtract spike estimate from source to see what's left to model
        sumsq = np.sum(rem_flt**2) / powerR
        rms[it] = sumsq   # save rms
        d_error = 100 * (sumsq_i - sumsq)  # check change in error as a result of this iteration

        sumsq_i = sumsq     # update rms
        it = it + 1         # and add one to the iteration count
    # once we get out of the loop:
    shift_filt = _phase_shift_filter(nfft, dt, tshift)
    p_flt = _apply_filter(p0, gaussF * shift_filt)
    
    RF_out = p_flt[:len(r0)]
    it_out = it
    rms_out = rms


    if normalize:
        norm = 1 / np.max(np.abs(RF_out))
        RF_out *= norm

    return RF_out, it_out, rms_out

###############################################################################
# https://github.com/trichter/rf/blob/master/rf/deconvolve.py #################
###############################################################################

def apply_hann_taper(arr):
    """
    CHATGPT bullshit
    
    Applies a 5% Hann taper to a numpy array.
    
    Parameters:
    arr (numpy.ndarray): The input array to which the taper is applied.
    
    Returns:
    numpy.ndarray: The tapered array.
    """
    n = len(arr)
    taper_length = int(0.05 * n)
    
    if taper_length == 0:
        raise ValueError("Array too short for a 5% taper.")
    
    # Create the Hann window
    hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(taper_length) / (2 * taper_length - 1)))
    
    # Create the full taper
    taper = np.ones(n)
    taper[:taper_length] = hann_window
    taper[-taper_length:] = hann_window[::-1]
    
    # Apply the taper to the array
    tapered_arr = arr * taper
    
    return tapered_arr

def compute_rfs(stnm, hdf5_file, method, method_settings, w0=0,
                time_shift=5,
                normalize=True,
                pbar=None,
                rotation="LQT",
                use_estimated_source_time_function=False,
                component="R",
                event_ids="all"):
    
    EARTH_RADIUS = 6378.137
    rfs = []
    
    if pbar is not None:
        total_count = len(list(hdf5_file[stnm].keys()))
        pbar.setRange(0, total_count)
    
    done = 0
    if event_ids == "all":
        event_ids = hdf5_file[stnm].keys()
    
    for event_id in event_ids:
        arr = hdf5_file[stnm][event_id][:]
        # Select and rotate
        z = arr[np.where(hdf5_file[stnm][event_id].attrs["data_structure"] == "Z")[0][0]]
        e = arr[np.where(hdf5_file[stnm][event_id].attrs["data_structure"] == "E")[0][0]]
        n = arr[np.where(hdf5_file[stnm][event_id].attrs["data_structure"] == "N")[0][0]]
        times = arr[np.where(hdf5_file[stnm][event_id].attrs["data_structure"] == "time")[0][0]]
        
        baz = hdf5_file[stnm][event_id].attrs["baz"]
        snr = hdf5_file[stnm][event_id].attrs["SNR"]
        inc = hdf5_file[stnm][event_id].attrs["incident_angle"]
        
        if rotation == "LQT":
            l, q, t = obspy.signal.rotate.rotate_zne_lqt(z, n, e, baz, inc)
            R_or_Q = q
            Z_or_L = l
        elif rotation == "ZRT":
            r, t = obspy.signal.rotate.rotate_ne_rt(n, e, baz)
            R_or_Q = r
            Z_or_L = z
        
        delta = np.diff(times)[0]
        phase = hdf5_file[stnm][event_id].attrs["phase"]
        
        if phase == "p":
            if component == "R" or component == "Q":
                dcmp = R_or_Q
                scmp = Z_or_L
            elif component == "Z" or component == "L":
                dcmp = Z_or_L
                scmp = Z_or_L
            elif component == "T":
                dcmp = t
                scmp = Z_or_L
        elif phase[0] == "s":
            dcmp = R_or_Q[::-1]
            scmp = Z_or_L[::-1]
        
        dcmp = apply_hann_taper(dcmp)
        scmp = apply_hann_taper(scmp)
        
        eq_magnitude = hdf5_file[stnm][event_id].attrs["mag"]
        ray_param = hdf5_file[stnm][event_id].attrs["ray_param"]
        distance = hdf5_file[stnm][event_id].attrs["dist_degrees"]
        data_structure = hdf5_file[stnm][event_id].attrs["data_structure"]
        
       
        # Pad data and perform deconvolution
        a = method_settings['gaussian_filter_width']
        if method == "Waterlevel":
            c = method_settings['waterlevel_parameter']
            rf = waterlevel_deconvolution(dcmp, scmp, delta, a, c, time_shift, w0=w0,
                                          normalize=normalize)
        if method == "Time-domain":
            max_iters = method_settings['max_iters']
            min_deltaE = method_settings['min_deltaE']
            rf, it, rms = deconv_iterative(dcmp, scmp, delta, tshift=time_shift, gauss=a,
                                           itmax=max_iters, minderr=min_deltaE, normalize=normalize)
        
        t = np.arange(-time_shift, -time_shift+delta*len(dcmp), delta)
        
        # It can happen that t and rf differ in length by 1 sample. This if block prevents
        # that from happening.
        if len(rf) > len(t):
            rf = rf[:len(t)]
        elif len(t) > len(rf):
            t = t[:len(rf)]
 
        if rotation == "LQT":
            rf = -rf            
        
        rfs.append([rf, t, baz, distance, ray_param, 1, event_id, eq_magnitude, snr]) # int 1 = accept this rf; 0 = discard (for use in the gui)
        done += 1
        
        if pbar is not None:
            pbar.setValue(done)
            
    
    return rfs

def bin_rfs(rfs, sort_param=2, min_=0, max_=360, bin_size=10, overlap=5):
    
    if bin_size == 0:

        bin_plot_ycoords = []
        rf_bin_indexes = []
        for j, rf in enumerate(rfs):
            rf_bin_indexes.append(j)
            bin_plot_ycoords.append(rf[sort_param])

    else:
    
        bins = []
        
        i = 0
        while min_ + (bin_size * (i + 1) - overlap * i) <= max_:
            llim = min_ + (bin_size * i - overlap * i)
            rlim = min_ + (bin_size * (i + 1) - overlap * i)
            
            bins.append((llim, rlim))
            
            i += 1
        
        if bins[-1][1] < max_:
            bins.append((bins[-1][1] - overlap, max_))
        
        bin_plot_ycoords = [(x[0] + x[1])/2 for x in bins]
        rf_bin_indexes = []
    
        for rf in rfs:
            for i, bin_ in enumerate(bins):
                if rf[sort_param] >= bin_[0] and rf[sort_param] <= bin_[1]:
                    rf_bin_indexes.append(i)
                    break

    return rf_bin_indexes, bin_plot_ycoords

def moveout_correction(rfs, phase='Ps', p_ref=6.4):
    
    # Moveout correction uses a single crustal layer of vp = 6.3 and vp/vs = 1.73
    avg_vp = 6.3
    avg_vs = avg_vp/1.73
    depths = np.arange(0, 1000, 0.01)
    dz = np.diff(depths)
    p_ref = p_ref/111.2
    
    # Reference delay for the Ps arrival
    tref = np.concatenate((np.zeros(1), np.cumsum((np.sqrt(1/avg_vs**2 - p_ref**2) - np.sqrt(1/avg_vp**2 - p_ref**2)) * dz)))
    
    for i, rf in enumerate(rfs):
        p = rf[4]/111.2 # Ray parameter in s/km
        tps = np.concatenate((np.zeros(1), np.cumsum((np.sqrt(1/avg_vs**2 - p**2) - np.sqrt(1/avg_vp**2 - p**2)) * dz))) # Ps delay time after P arrival
        # Create a function that y(x) that relates the reference delay times
        # with the theoretical ones for the given ray parameter and interpolate it
        # at the original times for the data samples
        corrected_time = scint.interp1d(tps, tref, bounds_error=False, fill_value=0)(rf[1])
        corrected_time[corrected_time == 0] = rf[1][rf[1] <= 0] # We do not correct below t = 0
        
        # Finally interpolate the receiver function at regularly spaced time invervals
        # so all the rfs are the same and it's possible to stack them and whatnot
        t = np.linspace(np.min(rf[1]), np.max(rf[1]), len(rf[1]))
        corrected_rf = scint.interp1d(corrected_time, rf[0], bounds_error=False, fill_value=0)(t)
        
        rfs[i][0] = corrected_rf
        rfs[i][1] = t
    
    return rfs

def compute_stack(rfs, bin_size=0, overlap=0, moveout_phase="Ps",
                  avg_vp=6.3, vpvs=1.73, ref_slowness=6.4, stack_by="Back az.",
                  normalize=True, min_dist=30, max_dist=90):
    
    # Moveout correction is performed using a single-layer model with
    # P velocity avg_vp. This could be changed to a global model, i.e.
    # iasp91 or something
    if moveout_phase == 'Ps':
        moveout_corrected_rfs = moveout_correction(copy.deepcopy(rfs))
    else:
        moveout_corrected_rfs = copy.deepcopy(rfs)
    
    moveout_corrected_rfs_copy2 = copy.deepcopy(moveout_corrected_rfs) # Something was wrong with the full linear stack
    # probably I am not handling some operation appropriately, for now this will have to do
    
    if stack_by == "Back az.":
        if bin_size > 0:
            bins = np.arange(0, 360+bin_size, bin_size)
        
            stacks = []
            y_coords = []
            for i in range(len(bins)-1):
                min_az = bins[i] - overlap
                max_az = bins[i+1] + overlap
    
                if min_az < 0:
                    min_az += 360
                if max_az > 360:
                    max_az -= 360
                
                stack = np.zeros(len(rfs[0][0]))
                stacked = 0
                for rf in moveout_corrected_rfs:
                    if rf[5] == 0:
                        continue
    
                    baz = rf[2]
                    if min_az < max_az:
                        inside = (min_az <= baz <= max_az)
                    else:
                        inside = (baz >= min_az or baz <= max_az)
                    if inside:
                        stack += rf[0]
                        stacked += 1
                
                if stacked == 0:
                    continue
                else:
                    stack /= stacked
                    stack /= np.max(np.abs(stack))
                    stacks.append(stack)
                    
                    y = bins[i]+bin_size/2
                    if y > 360:
                        y -= 360
                    y_coords.append(y)
        elif bin_size == 0:
            stacks = [rf[0]/np.max(np.abs(rf[0])) for rf in moveout_corrected_rfs]
            y_coords = [rf[2] for rf in moveout_corrected_rfs]
    elif stack_by == "Distance":
        if bin_size > 0:
            bins = np.arange(min_dist, max_dist+bin_size, bin_size)
            
            stacks = []
            y_coords = []
            for i in range(len(bins)-1):
                min_dist = bins[i] - overlap
                max_dist = bins[i+1] + overlap

                stack = np.zeros(len(rfs[0][0]))
                stacked = 0
                for rf in moveout_corrected_rfs:
                    if rf[5] == 0:
                        continue

                    dist = rf[3]     
                    inside = (min_dist <= dist <= max_dist)
                    if inside:
                        stack += rf[0]
                        stacked += 1

                if stacked == 0:
                    continue
                else:
                    stack /= stacked
                    stack /= np.max(np.abs(stack))
                    stacks.append(stack)
                    
                    y = bins[i]+bin_size/2
                    if y > 360:
                        y -= 360
                    y_coords.append(y)
        elif bin_size == 0:
            stacks = [rf[0]/np.max(np.abs(rf[0])) for rf in moveout_corrected_rfs]
            y_coords = [rf[3] for rf in moveout_corrected_rfs]
        
    
    full_linear_stack = np.zeros(len(rfs[0][0]))
    for rf in moveout_corrected_rfs_copy2:
        if rf[5] != 0:
            full_linear_stack += rf[0]
    full_linear_stack /= len([x for x in moveout_corrected_rfs_copy2 if x[5] == 1])

    return full_linear_stack, stacks, y_coords

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def compute_hk_stack(rfs, avg_vp=6.3, H_range=(25, 55), H_values=100, k_range=(1.60, 1.80),
                     k_values=100, w1=0.50, w2=0.25, w3=0.25, semblance_weighting=True):

    H_arr = np.linspace(min(H_range), max(H_range), H_values)
    k_arr = np.linspace(min(k_range), max(k_range), k_values)
    
    # Semblance function (Eaton et al. 2006)
    S1_num = np.zeros((len(k_arr), len(H_arr)))
    S1_deno = np.zeros((len(k_arr), len(H_arr)))
    
    S2_num = np.zeros((len(k_arr), len(H_arr)))
    S2_deno = np.zeros((len(k_arr), len(H_arr)))
    
    S3_num = np.zeros((len(k_arr), len(H_arr)))
    S3_deno = np.zeros((len(k_arr), len(H_arr)))
    
    matrix1 = np.zeros((len(k_arr), len(H_arr)))
    matrix2 = np.zeros((len(k_arr), len(H_arr)))
    matrix3 = np.zeros((len(k_arr), len(H_arr)))
    
    vs = np.array([avg_vp/k for k in k_arr])
    
    events = 0
    for rf_arr in rfs:
        
        if rf_arr[5]:
            events += 1
            rf = scint.interp1d(rf_arr[1], rf_arr[0], bounds_error=False, fill_value=np.nan)
            p = rf_arr[4]/6378.137
            tps = np.einsum('i,j->ji', H_arr, (np.sqrt(1/(vs**2) - p**2) - np.sqrt(1/(avg_vp**2) - p**2)))
            tppps = np.einsum('i,j->ji', H_arr, (np.sqrt(1/(vs**2) - p**2) + np.sqrt(1/(avg_vp**2) - p**2)))
            tpsps = np.einsum('i,j->ji', H_arr, 2 * (np.sqrt(1/(vs**2) - p**2)))
            
            matrix1 += w1 * rf(tps)
            matrix2 += w2 * rf(tppps)
            matrix3 += -w3 * rf(tpsps)
            
            S1_num += rf(tps)
            S1_deno += rf(tps)**2
        
            S2_num += rf(tppps)
            S2_deno += rf(tppps)**2
        
            S3_num += rf(tpsps)
            S3_deno += rf(tpsps)**2
    
    S1 = S1_num**2 / S1_deno
    S2 = S2_num**2 / S2_deno
    S3 = S3_num**2 / S3_deno
    
    if semblance_weighting:
        matrix = np.nansum(np.dstack((S1 * matrix1, S2 * matrix2, S3 * matrix3)),2)
    else:
        matrix = np.nansum(np.dstack((matrix1, matrix2, matrix3)), 2)
    
    # matrix[matrix < 0] = 0
    maxy = np.where(matrix == np.max(matrix))[0][0]
    maxx = np.where(matrix == np.max(matrix))[1][0]
    H = H_arr[maxx]
    k = k_arr[maxy]

    return H_arr, k_arr, matrix, H, k, events

def compute_theoretical_arrival_times(H, k, ref_slowness=6.4, avg_vp=6.4):
    vs = avg_vp / k
    p = ref_slowness/111.2
    
    tps = H * (np.sqrt(1/(vs**2) - p**2) - np.sqrt(1/(avg_vp**2) - p**2))
    tppps = H * (np.sqrt(1/(vs**2) - p**2) + np.sqrt(1/(avg_vp**2) - p**2))
    tpsps = H* (2 * (np.sqrt(1/(vs**2) - p**2)))
    
    return tps, tppps, tpsps

def save_rfs_hdf5(stnm, stla, stlo, stel, method, method_settings, rfs, outfile):

    # 1. check if file exists, otherwise create it
    if Path(outfile).is_file():
        try:
            hdf5 = h5py.File(outfile, "r+")
        except OSError:
            return 1
    else:
        hdf5 = h5py.File(outfile, "w")
        
    # 2. if group already exists, delete it
    if stnm in list(hdf5.keys()):
        del hdf5[stnm]
    
    # 3. Create group and set attributes
    g = hdf5.create_group(stnm)
    g.attrs["stla"] = stla
    g.attrs["stlo"] = stlo
    g.attrs["stel"] = stel
    g.attrs["method"] = method
    for key in method_settings.keys():
        g.attrs[key] = method_settings[key]
    
    # 4. Add receiver functions as datasets
    for rf in rfs:
        rf_data, t, baz, distance, ray_param, use, event_id, eq_magnitude = rf
        d = g.create_dataset(str(event_id), data=np.array([rf_data, t]))
        d.attrs["data_structure"] = ["receiver_function", "time"]
        d.attrs["back_azimuth"] = baz
        d.attrs["distance_degrees"] = distance
        d.attrs["ray_param"] = ray_param
        d.attrs["accepted"] = use
        d.attrs["magnitude"] = eq_magnitude
    
    hdf5.close()
    return 0          


def map_rfs(rfs_dir="rf"):
    rfs_map = {}
    for top_dir, sub_dir, files in os.walk(rfs_dir):
        for file in files:
            path = os.path.join(top_dir, file)
            if file.endswith(".pickle"):
                rfs = pickle.load(open(path, "rb"))
                try:
                    stnm = rfs['station']
                except KeyError:
                    continue
                rfs_map[stnm] = path

    return rfs_map

def ccp_stack(rfs_hdf5, min_x, max_x, min_y, max_y, dx, dy, dz, max_depth,
              model='iasp91', stacking_method='mean', pbar=None):

    elevs = []
    for stnm in rfs_hdf5.keys():
        elevs.append(rfs_hdf5[stnm].attrs["stel"]/1000) # We assume its in m
    
    y = np.arange(min_y, max_y, dy)
    x = np.arange(min_x, max_x, dx)
    z = np.arange(-np.max(elevs), max_depth, dz)
    
    x_mask = x[:,np.newaxis]
    y_mask = y[np.newaxis,:]

    stack = np.zeros((len(z), len(x), len(y)))
    weights = np.zeros((len(z), len(x), len(y)))    
    
    # Read earth model:
    path_model=os.path.join(os.path.dirname(os.path.abspath(__file__)), "earth_models")
    with open(path_model+"/{}.csv".format(model), 'r') as f:
        model_lines = f.readlines()
    
    radius_arr = []
    vp_arr = []
    vs_arr = []
    
    for line in model_lines:
        depth, radius, vp, vs = line.split(',')
        radius_arr.append(float(radius))
        vp_arr.append(float(vp))
        vs_arr.append(float(vs))
    
    ivp = scint.interp1d(radius_arr, vp_arr, bounds_error=False, fill_value=(vp_arr[0], vp_arr[-1]))
    ivs = scint.interp1d(radius_arr, vs_arr, bounds_error=False, fill_value=(vs_arr[0], vs_arr[-1]))
    
    r_earth = float(model_lines[0].split(',')[1])
    
    total_count = 0
    for stnm in rfs_hdf5.keys():
        for id_ in rfs_hdf5[stnm].keys():
            total_count += 1
            
    pbar.setRange(0, total_count)
    
    print(rfs_hdf5[stnm].attrs["stel"])
    
    done = 0
    for stnm in rfs_hdf5.keys():
        stla = rfs_hdf5[stnm].attrs["stla"]
        stlo = rfs_hdf5[stnm].attrs["stlo"]
        stel = rfs_hdf5[stnm].attrs["stel"]/1000 # We assume it is in m
        
        for id_ in rfs_hdf5[stnm].keys():
            d = rfs_hdf5[stnm][id_]
            if not d.attrs["accepted"]:
                continue
            
            rf = d[0]
            time = d[1]
            p = d.attrs["ray_param"]
            baz = d.attrs["back_azimuth"]
            tps = 0
            r0 = radius_arr[0]+np.max(elevs)
            r_sta = radius_arr[0]+stel
            p_h = p/112
            
            intp_rf = scint.interp1d(time, rf)
            lat = math.radians(stla)
            lon = math.radians(stlo)
            H = 0

            
            T0 = 2 # hard-coded, change
        
            for k in range(len(z)):
                r = r0 - dz*k
                if r > r_sta:
                    continue
                H += dz
                vs = ivs(r)
                vp = ivp(r)
                dt = (np.sqrt(vs**-2 - p_h**2) - np.sqrt(vp**-2 - p_h**2)) * dz
                tps += dt
                amp = intp_rf(tps)
    
                ddist =  (p_h / np.sqrt(vs**-2 - p_h**2) * dz)
                ddist = math.radians(ddist/(2.0 * r * math.pi / 360.0))
                nlat = np.arcsin(np.sin(lat) * np.cos(ddist) + np.cos(lat) * np.sin(ddist) * np.cos(baz))
                nlon = lon + np.arctan2(np.sin(baz) * np.sin(ddist) * np.cos(lat), np.cos(ddist) - np.sin(lat) * np.sin(nlat))
                
                fresnell_radius = np.sqrt(H * (vs*T0))/(2.0 * r * math.pi / 360.0) # From the Ylmaz book
                lat = nlat
                lon = nlon
                deglat = math.degrees(nlat)
                deglon = math.degrees(nlon)

                if deglon < min_x or deglon > max_x or deglat < min_y or deglat > max_y:
                    break

                dist_x = (x_mask-deglon)**2
                dist_y = (y_mask-deglat)**2
                dist_matrix = np.sqrt(dist_x + dist_y)
                mask = np.less(dist_matrix, fresnell_radius)
                
                if stacking_method == "weighted mean":
                    weight = np.exp(-dist_x/(2*(fresnell_radius/2)**2) - dist_y/(2*(fresnell_radius/2)**2))
                    masked_weight = weight[mask]
                    stack[k][mask] += amp*masked_weight
                    weights[k][mask] += masked_weight
                elif stacking_method == "mean":
                    stack[k][mask] += amp
                    weights[k][mask] += 1    
        
            done += 1
            pbar.setValue(done)

    stack_average = np.divide(stack, weights, where=(weights != 0))
    return stack_average, x, y, z

def compute_intermediate_points(start, end, npts):
    A_lats = np.radians(start[0])
    A_lons = np.radians(start[1])
    B_lats = np.radians(end[0])
    B_lons = np.radians(end[1])

    fs = np.linspace(0, 1, npts)
    
    dfi = B_lats - A_lats
    dlambd = B_lons - A_lons
    
    a = np.sin(dfi/2)**2 + np.cos(A_lats) * np.cos(B_lats) * np.sin(dlambd/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    newlats = [start[0]]
    newlons = [start[1]]

    for f in fs:
        d = np.sin((1-f)*c) / np.sin(c)
        e = np.sin(f*c) / np.sin(c)
        
        x = d * np.cos(A_lats) * np.cos(A_lons) + e * np.cos(B_lats) * np.cos(B_lons)
        y = d * np.cos(A_lats) * np.sin(A_lons) + e * np.cos(B_lats) * np.sin(B_lons)
        z = d * np.sin(A_lats) + e * np.sin(B_lats)
        
        phi = np.arctan2(z, np.sqrt(x**2 + y**2))
        lambd = np.arctan2(y, x)
    
        lat, lon = np.degrees(phi), np.degrees(lambd)
        
        newlats.append(lat)
        newlons.append(lon)
        
    newlats.append(end[0])
    newlons.append(end[1])
    
    # approximate distance array (in km)
    total_dist = c*6371.0
    dist_arr = np.arange(0, total_dist, total_dist/npts)
    
    return newlats, newlons, dist_arr
        

def point_inside_polygon(x, y, poly, include_edges=True):
    '''
    Test if point (x,y) is inside polygon poly.

    poly is N-vertices polygon defined as 
    [(x1,y1),...,(xN,yN)] or [(x1,y1),...,(xN,yN),(x1,y1)]
    (function works fine in both cases)

    Geometrical idea: point is inside polygon if horisontal beam
    to the right from point crosses polygon even number of times. 
    Works fine for non-convex polygons.
    '''
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if p1y == p2y:
            if y == p1y:
                if min([p1x, p2x]) <= x <= max([p1x, p2x]):
                    # point is on horisontal edge
                    inside = include_edges
                    break
                elif x < min([p1x, p2x]):  # point is to the left from current edge
                    inside = not inside
        else:  # p1y!= p2y
            if min([p1y, p2y]) <= y <= max([p1y, p2y]):
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                if x == xinters:  # point is right on the edge
                    inside = include_edges
                    break

                if x < xinters:  # point is to the left from current edge
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside

def iscontiguous(coords, region):
    moves = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]
    for p in region:
        move = (p[0] - coords[0], p[1] - coords[1])
        if move in moves:
            return True

    return False

def determine_error_region(matrix, H_arr, k_arr, N_stacked):
    error_area = None
    dict_ = {"H_arr":H_arr,"k_arr":k_arr, "matrix":matrix}

    maxy = np.where(matrix == np.max(matrix))[0][0]
    maxx = np.where(matrix == np.max(matrix))[1][0]
    error_contour_level = np.sqrt(np.std(matrix)**2/N_stacked)
    error_region = np.max(matrix) - error_contour_level
    error_matrix = np.zeros(matrix.shape)
    error_matrix[np.where(matrix > error_region)[0], np.where(matrix > error_region)[1]] = 1
    
    a = np.diff(error_matrix)
    a[np.where(a == -1)[0], np.where(a == -1)[1]] = 1
    
    regions = {}    
    
    label = 1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            coords = (i,j)
            value = a[i,j]
            
            if value == 1 and not regions:
                regions.setdefault(label, [])
                regions[label].append(coords)
            elif value == 1:
                region_exists = False
                for region in regions.keys():
                    if iscontiguous(coords, regions[region]):
                        regions[region].append(coords)
                        region_exists = True
                        break
                
                if not region_exists:
                    label += 1
                    regions.setdefault(label, [])
                    regions[label].append(coords)
    
    for r in regions.keys():
        region = regions[r]
        if point_inside_polygon(maxy, maxx, region):
            error_area = region
            break

    if error_area != None:
        error_k_values = k_arr[[x[0] for x in error_area]]
        error_H_values = H_arr[[x[1] for x in error_area]]
        k_95 = (np.min(error_k_values), np.max(error_k_values))
        H_95 = (np.min(error_H_values), np.max(error_H_values))
    else:
        k_95 = None
        H_95 = None
    
    return a, error_area, k_95, H_95, error_contour_level

def interpolate_ccp_stack(x, y, stack):
    
    interps = []
    for i in range(stack.shape[0]):
        interps.append(scint.interp2d(y, x, stack[i,:,:], bounds_error=False,
                                      fill_value=np.NaN))
    
    return interps

def compute_radius(ortho, lat, lon, radius_degrees):
    # Used for computing distance circles in earthquake map
    phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
    _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
    return abs(y1)

    
        
