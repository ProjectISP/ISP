# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:04:08 2020

@author: olivar

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
import numpy as np
import math
import scipy.interpolate as scint
import dill as pickle
import cartopy.crs as ccrs
import copy

def map_earthquakes(eq_dir="earthquakes"):
    earthquake_map = {}
    for top_dir, sub_dir, files in os.walk(eq_dir):
        for file in files:
            path = os.path.join(top_dir, file)
            station = path.split(os.path.sep)[-2]
            if file.endswith(".mseed"):
                event_id = int(path.split(os.path.sep)[-1].split('_')[0].strip("EQ"))
                earthquake_map.setdefault(station, {})
                earthquake_map[station][event_id] = path
    
    return earthquake_map

def waterlevel_deconvolution(dcmp, scmp, delta, a, c, tshift, w0=0):
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
    rf = np.fft.irfft((num/deno)* np.exp(-0.5*(2*np.pi*freq-w0)**2/(a**2)) * np.exp(-1j * tshift * 2 * np.pi * freq))

    rf = rf / np.max(np.abs(rf))
        
    return rf[:len(dcmp)]

def compute_rfs(stnm, data_map, arrivals, srfs={}, dcmpn="Q", scmpn="L",
                filter_=True, corner_freqs=(0.2, 20), a=2.5, c=0.01):
    
    EARTH_RADIUS = 6378.137
    rfs = []

    for event_id in data_map[stnm].keys():
        # Read data
        st = obspy.read(data_map[stnm][event_id], format="MSEED")
        st.detrend(type='demean')
        st.detrend(type='linear')
        st.taper(0.05)
        
        delta = st[0].stats.delta
        
        # Check for errors in mseed file
        if len(st) < 3:
            continue
        
        # Check that there actually is data in the mseed file (since merge(fill_value=0) is used
        # when cutting events, it could be all zeros.)
        not_enough_data = False
        for tr in st:
            data_len = len(tr.data)
            eighty_perc = int(round(data_len*0.8))
            data_zeros = (tr.data == 0)
            if np.sum(data_zeros) >= eighty_perc:
                not_enough_data = True
                break
        
        if not_enough_data:
            continue
        
        if filter_:
            dcmp = st.select(component=dcmpn).filter('bandpass', freqmin=min(corner_freqs),
                                                     freqmax=max(corner_freqs), zerophase=True)
            dcmp = dcmp[0].data
        else:
            dcmp = st.select(component=dcmpn)[0].data
        
        if srfs != {}:
            scmp = srfs[event_id]
        else:
            if filter_:
                scmp = st.select(component=scmpn).filter('bandpass', freqmin=min(corner_freqs),
                                                         freqmax=max(corner_freqs), zerophase=True)
                scmp = scmp[0].data
            else:
                scmp = st.select(component=scmpn)[0].data

        # Get time shift for P onset from the event metadata
        otime = arrivals['events'][event_id]['event_info']['origin_time']
        atime = otime + arrivals['events'][event_id]['arrivals'][stnm]['arrival_time']
        stime = -round(atime - st[0].stats.starttime)
        etime = round(st[0].stats.endtime - atime)
        t = np.linspace(stime, etime, len(dcmp))
        
        # Rf metadata
        try:
            eq_magnitude = arrivals['events'][event_id]['event_info']['magnitude']
        except KeyError:
            eq_magnitude = 0
        eq_file = data_map[stnm][event_id]
        baz = arrivals['events'][event_id]['arrivals'][stnm]['back_azimuth']
        ray_param = arrivals['events'][event_id]['arrivals'][stnm]['ray_parameter']/EARTH_RADIUS
        distance = arrivals['events'][event_id]['arrivals'][stnm]['distance']
        
        rf = waterlevel_deconvolution(dcmp, -scmp, st[0].stats.delta, a, c, 5, corner_freqs[0])

        t = np.arange(-5, -5+delta*len(dcmp), delta)
        
        # It can happen that t and rf differ in length by 1 sample due to rounding
        # errors. This if block prevents that from happening.
        if len(rf) > len(t):
            rf = rf[:len(t)]
        elif len(t) > len(rf):
            t = t[:len(rf)]
        
        rfs.append([rf, t, baz, distance, ray_param, 1, event_id, eq_magnitude,
                    eq_file]) # int 1 = accept this rf; 0 = discard (for use in the gui)
    
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

    tref = np.concatenate((np.zeros(1), np.cumsum((np.sqrt(1/avg_vs**2 - p_ref**2) - np.sqrt(1/avg_vp**2 - p_ref**2)) * dz)))
    
    for i, rf in enumerate(rfs):
        p = rf[4]
        tps = np.concatenate((np.zeros(1), np.cumsum((np.sqrt(1/avg_vs**2 - p**2) - np.sqrt(1/avg_vp**2 - p**2)) * dz)))
        corrected_time = scint.interp1d(tps, tref, bounds_error=False, fill_value=0)(rf[1])
        corrected_time[corrected_time == 0] = rf[1][rf[1] <= 0]
        t = np.linspace(np.min(rf[1]), np.max(rf[1]), len(rf[1]))
        corrected_rf = scint.interp1d(corrected_time, rf[0], bounds_error=False, fill_value=0)(t)
        
        rfs[i][0] = corrected_rf
        rfs[i][1] = t
    
    return rfs

def compute_stack(rfs, bin_size=0, overlap=0, moveout_phase="Ps",
                  avg_vp=6.3, vpvs=1.73, ref_slowness=6.4, stack_by="Back az.",
                  normalize=True, min_dist=30, max_dist=90):
    
    # Moveout correction is performed using a single-layer model with
    # P velocity avg_vp
    if stack_by == "Back az.":
        sort_index = 2
        min_ = 0
        max_ = 360
    elif stack_by == "Distance":
        sort_index = 3
        min_ = min_dist
        max_ = max_dist
    
    rf_bin_indexes, bin_plot_ycoords = bin_rfs(rfs, sort_param=sort_index, min_=min_,
                                               max_=max_, bin_size=bin_size,
                                               overlap=overlap)
    
    bin_stacks = np.zeros((len(rfs[0][0]), len(bin_plot_ycoords)))
    stack = np.zeros(len(rfs[0][0]))
    
    moveout_corrected_rfs = moveout_correction(copy.deepcopy(rfs))
    
    for i, rf in enumerate(moveout_corrected_rfs):
        try:
            if rf[5]:
                bin_index = rf_bin_indexes[i]
                bin_stacks[:,bin_index] += rf[0]
                stack += rf[0]
        except ValueError:
            continue

    if normalize:
        bin_stacks /= np.abs(bin_stacks.max(axis=0))
        
        bin_stacks *= 5
        """if bin_size >= 5:
            bin_stacks *= bin_size
        else:
            bin_stacks *= 5"""
        
        stack /= np.abs(np.max(stack))

    return stack, bin_stacks, bin_plot_ycoords, min(bin_plot_ycoords) - bin_size, max(bin_plot_ycoords) + bin_size

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def compute_hk_stack(rfs, avg_vp=6.3, H_range=(25, 55), H_values=100, k_range=(1.60, 1.80),
                     k_values=100, w1=0.50, w2=0.25, w3=0.25):

    H_arr = np.linspace(min(H_range), max(H_range), H_values)
    k_arr = np.linspace(min(k_range), max(k_range), k_values)
    
    # semblance function (Eaton et al. 2006)
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
            rf = scint.interp1d(rf_arr[1], rf_arr[0])
            p = rf_arr[4]
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
    
    matrix = S1 * matrix1 + S2 * matrix2 + S3 * matrix3
    
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

def save_rfs(stnm, a, c, rfs, outdir="rf/"):

        rfs = [rf for rf in rfs if rf[5]]    

        rfs_dict = {"station": stnm,
                    "deconvolution_parameters": {"a":a, "c":c},
                    "receiver_functions": rfs}

        pickle.dump(rfs_dict, open(os.path.join(outdir, "{}.pickle".format(stnm)), "wb"))

def map_rfs(rfs_dir="rf"):
    rfs_map = {}
    for top_dir, sub_dir, files in os.walk(rfs_dir):
        for file in files:
            path = os.path.join(top_dir, file)
            if file.endswith(".pickle"):
                rfs = pickle.load(open(path, "rb"))
                stnm = rfs['station']
                rfs_map[stnm] = path

    return rfs_map

def ccp_stack(rfs_map, evdata, min_x, max_x, min_y, max_y, dx, dy, dz, max_depth,
              model='iasp91'):

    y = np.arange(min_y, max_y, dy)
    x = np.arange(min_x, max_x, dx)
    z = np.arange(0, max_depth, dz)
    
    stack = [[[[] for i in range(len(z))] for i in range(len(y))] for i in range(len(x))]

    stack = np.zeros((len(x), len(y), len(z)))
    hits = np.zeros((len(x), len(y), len(z)))    
    
    
    path_model=os.path.join(os.path.dirname(os.path.abspath(__file__)), "earth_models")
    with open(path_model+"/{}.csv".format(model), 'r') as f:
        model_lines = f.readlines()
    
    depth_arr = []
    vp_arr = []
    vs_arr = []
    
    for line in model_lines:
        depth, radius, vp, vs = line.split(',')
        depth_arr.append(float(depth))
        vp_arr.append(float(vp))
        vs_arr.append(float(vs))
    
    ivp = scint.interp1d(depth_arr, vp_arr)
    ivs = scint.interp1d(depth_arr, vs_arr)
    
    r_earth = float(model_lines[0].split(',')[1])
    
    for stnm in rfs_map.keys():
        rfs = pickle.load(open(rfs_map[stnm], "rb"))
        stla = evdata["stations"][stnm]["lat"]
        stlo = evdata["stations"][stnm]["lon"]
        stel = evdata["stations"][stnm]["elev"]
        
        for rf_arr in rfs['receiver_functions']:
            eq_id = rf_arr[6]
            p = rf_arr[4]
            p_sph = p*r_earth
            
            p_h = p/111.2
            
            t = rf_arr[1]
            rf = rf_arr[0]/np.max(np.abs(rf_arr[0]))
            intp_rf = scint.interp1d(t, rf)
            baz = evdata["events"][eq_id]['arrivals'][stnm]["back_azimuth"]

            r_earth = 6371
            H = 0
            T0 = 1
        
            lat = math.radians(stla)
            lon = math.radians(stlo)
            baz = math.radians(baz)

            dist = 0
            tps = 0
            r0 = r_earth+stel/1000

            for k in range(int(round(max_depth/dz))):
                H = k*dz
                r = r0-dz

                vs = ivs(H)
                vp = ivp(H)

                dist =  p_h * (r - r0) / np.sqrt(vs**-2 - p_h**2)
                rdist = math.radians(dist)

                dt = (np.sqrt(vs**-2 - p**2) - np.sqrt(vp**-2 - p**2)) * dz
                tps += dt
                
                fzone_x = np.sqrt(H * (vs*T0))/111.2
                fzone_y = np.sqrt(H * (vs*T0))/111.2
    
                nlat = np.arcsin(np.sin(lat) * np.cos(rdist) + np.cos(lat) * np.sin(rdist) * np.cos(baz))
                nlon = lon + np.arctan2(np.sin(baz) * np.sin(rdist) * np.cos(lat), np.cos(rdist) - np.sin(lat) * np.sin(nlat))
                
                lat = nlat
                lon = nlon
                amp = intp_rf(tps)

                deglat = math.degrees(nlat)
                deglon = math.degrees(nlon)
                
                x_index = int((deglon-x[0])/dx)
                y_index = int((deglat-y[0])/dy)
                
                fzone_x_size = int(fzone_x/dx)
                fzone_y_size = int(fzone_y/dy)
                
                x_cells = np.arange(x_index - fzone_x_size, x_index + fzone_x_size, 1)
                y_cells = np.arange(y_index - fzone_y_size, y_index + fzone_y_size, 1)
                
                x_mask = (x_cells >= 0) & (x_cells < len(x))
                y_mask = (y_cells >= 0) & (y_cells < len(y))
                x_cells = x_cells[x_mask]
                y_cells = y_cells[y_mask]
                
                for i in x_cells:
                    for j in y_cells:
                        stack[i,j,k] += amp
                        hits[i,j,k] += 1
                        
    stack_average = np.divide(stack, hits, where=(stack != 0))

    return stack_average, np.arange(0, max_depth, dz)

def compute_intermediate_points(start, end, npts):
    A_lats = np.radians(start[1])
    A_lons = np.radians(start[0])
    B_lats = np.radians(end[1])
    B_lons = np.radians(end[0])

    fs = np.linspace(0, 1, npts)
    
    dfi = B_lats - A_lats
    dlambd = B_lons - A_lons
    
    a = np.sin(dfi/2)**2 + np.cos(A_lats) * np.cos(B_lats) * np.sin(dlambd/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    newlats = [start[1]]
    newlons = [start[0]]

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
        
    newlats.append(end[1])
    newlons.append(end[0])
    
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
    
    Source: 
    https://stackoverflow.com/questions/39660851/deciding-if-a-point-is-inside-a-polygon
    (last checked 2021-04-26)
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
    for i in range(stack.shape[-1]):
        interps.append(scint.interp2d(y, x, stack[:,:,i], bounds_error=False,
                                      fill_value=np.NaN))
    
    return interps

def compute_radius(ortho, lat, lon, radius_degrees):
    # Used for computing distance circles in earthquake map
    phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
    _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
    return abs(y1)
    
    
    
    
        