# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:04:08 2020

@author: olivar
"""

import os
import obspy
import numpy as np
import math
import scipy.interpolate as scint
import pickle


def map_earthquakes(eq_dir="earthquakes"):
    earthquake_map = {}
    for top_dir, sub_dir, files in os.walk(eq_dir):
        for file in files:
            path = os.path.join(top_dir, file)
            station = path.split(os.path.sep)[-2]
            event_id = int(path.split(os.path.sep)[-1].split('_')[0].strip("EQ"))
            if file.endswith(".mseed"):
                earthquake_map.setdefault(station, {})
                earthquake_map[station][event_id] = path
    
    return earthquake_map

def compute_source_functions(data_map, scmpn="L", filter_=True, corner_freqs=(0.2, 20),
                             normalize=False):
    srfs = {}
    for stnm in data_map.keys():
        for event_id in data_map[stnm]:
            st = obspy.read(data_map[stnm][event_id], format="MSEED")
            st = st.select(component=scmpn)
            
            if filter_:
                st.filter('bandpass', freqmin=min(corner_freqs),
                          freqmax=max(corner_freqs))
            
            srfs.setdefault(event_id, [])
            
            try:
                if normalize:
                    srfs[event_id].append(st[0].data/np.max(np.abs(st[0].data)))
                else:
                    srfs[event_id].append(st[0].data)
            except IndexError:
                print("Warning: No data found for station {}, channel {},".format(stnm, scmpn) +
                      "event id: {}".format(event_id))
    
    for event_id in srfs.keys():
        srfs[event_id] = np.sum(np.array(srfs[event_id]), axis=0)/len(srfs[event_id])
    
    return srfs
            

def compute_rfs(stnm, data_map, arrivals, srfs={}, dcmpn="Q", scmpn="L",
                filter_=True, corner_freqs=(0.2, 20), a=2.5, c=0.01):
    
    EARTH_RADIUS = 6378.137
    rfs = []

    for event_id in data_map[stnm].keys():
        # Read data
        st = obspy.read(data_map[stnm][event_id], format="MSEED")
        
        if filter_:
            dcmp = st.select(component=dcmpn).filter('bandpass', freqmin=min(corner_freqs),
                                                     freqmax=max(corner_freqs))
            dcmp = - dcmp[0].data
        else:
            dcmp = st.select(component=dcmpn)[0].data
        
        if srfs != {}:
            scmp = srfs[event_id]
        else:
            if filter_:
                scmp = st.select(component=scmpn).filter('bandpass', freqmin=min(corner_freqs),
                                                         freqmax=max(corner_freqs))
                scmp = scmp[0].data
            else:
                scmp = st.select(component=scmpn)[0].data

        # Get time shift for P onset from the event metadata
        otime = arrivals['events'][event_id]['event_info']['origin_time']
        atime = otime + arrivals['events'][event_id]['stations'][stnm]['arrival_time']
        stime = -round(atime - st[0].stats.starttime)
        etime = round(st[0].stats.endtime - atime)
        t = np.linspace(stime, etime, len(dcmp))
        
        # Rf metadata
        baz = arrivals['events'][event_id]['stations'][stnm]['azimuth'] # Esto está mal porque en el cut_earthquakes lo has puesto al reves (hay q. seguir la convencion de iris)
        ray_param = arrivals['events'][event_id]['stations'][stnm]['ray_parameter']/EARTH_RADIUS
        distance = arrivals['events'][event_id]['stations'][stnm]['distance']
        
        # Pad data and perform deconvolution
        max_len = np.maximum(len(dcmp), len(scmp))
        next_pow2 = 2**math.ceil(math.log(max_len, 2))        
        
        dcmp_pad = np.zeros(next_pow2 - len(dcmp))
        scmp_pad = np.zeros(next_pow2 - len(scmp))
        
        pdcmp = np.insert(dcmp, len(dcmp), dcmp_pad)
        pscmp = np.insert(scmp, len(scmp), scmp_pad)
        
        dfft = np.fft.fft(pdcmp)
        sfft = np.fft.fft(pscmp)
        freq = np.fft.fftfreq(len(pdcmp), d=st[0].stats.delta)
        
        num = dfft * np.conj(sfft)
        deno = np.maximum(sfft * np.conj(sfft), c*np.max(sfft * np.conj(sfft)))
        gfilt = np.exp(np.maximum(-(2*np.pi*freq**2)/(a**2), -700) - 1j * -stime * 2*np.pi*freq)
        rf = (1 + c) * np.fft.ifft(num/deno * gfilt)[:len(dcmp)].real
        
        rfs.append([rf, t, baz, distance, ray_param, 1, event_id]) # int 1 = accept this rf; 0 = discard (for use in the gui)
    
    return rfs

def bin_rfs(rfs, sort_param=2, min_=0, max_=360, bin_size=10, overlap=5):
    
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

def compute_stack(rfs, bin_size=0, overlap=0, moveout_correction="Ps",
                  avg_vp=6.4, vpvs=1.73, ref_slowness=6.4, stack_by="Back az.",
                  normalize=True, min_dist=30, max_dist=90):
    
    # Moveout correction is performed using a single-layer model with
    # P velocity avg_vp. This could be changed to a global model, i.e.
    # iasp91 or something
    
    # Compute reference time
    avg_vs = avg_vp/vpvs
    p_ref = ref_slowness/111.2
    depths = np.arange(0, 200, 0.1)
    dz = np.diff(depths)
    t_ref = np.cumsum((np.sqrt(1/avg_vs**2 - p_ref**2) - np.sqrt(1/avg_vp**2 - p_ref**2)) * dz)
    t_ref = np.hstack((-t_ref[:][::-1], t_ref))

    # Determine sort index
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
    
    for i, rf in enumerate(rfs):
        if rf[5]:
            bin_index = rf_bin_indexes[i]
            if moveout_correction == "Ps":
                p = rf[4]
                t_p = np.cumsum((np.sqrt(1/avg_vs**2 - p**2) - np.sqrt(1/avg_vp**2 - p**2)) * dz)
                t_p = np.hstack((-t_p[:][::-1], t_p))
                t_corr = np.interp(rf[1], t_p, t_ref, left=0, right=None)
                corr_rf = np.interp(rf[1], t_corr, rf[0], left=None, right=0)
                
                bin_stacks[:,bin_index] += corr_rf
                stack += corr_rf
            else:
                bin_stacks[:,bin_index] += rf[0]
                stack += rf[0]

    if normalize:
        bin_stacks /= np.abs(bin_stacks.max(axis=0))
        bin_stacks *= bin_size
        
        stack /= np.abs(np.max(stack))

    return stack, bin_stacks, bin_plot_ycoords, min(bin_plot_ycoords) - bin_size, max(bin_plot_ycoords) + bin_size

def compute_hk_stack(rfs, avg_vp=6.3, H_range=(25, 55), H_values=100, k_range=(1.60, 1.80),
                     k_values=100, w1=0.34, w2=0.33, w3=0.33):

    H_arr = np.linspace(min(H_range), max(H_range), H_values)
    k_arr = np.linspace(min(k_range), max(k_range), k_values)
    
    # semblance function (Eaton et al. 2006)
    S1_num = np.zeros((len(k_arr), len(H_arr)))
    S1_deno = np.zeros((len(k_arr), len(H_arr)))
    
    S2_num = np.zeros((len(k_arr), len(H_arr)))
    S2_deno = np.zeros((len(k_arr), len(H_arr)))
    
    S3_num = np.zeros((len(k_arr), len(H_arr)))
    S3_deno = np.zeros((len(k_arr), len(H_arr)))
    
    matrix = np.zeros((len(k_arr), len(H_arr)))
    
    vs = np.array([avg_vp/k for k in k_arr])
    
    for rf_arr in rfs:
        rf = scint.interp1d(rf_arr[1], rf_arr[0])
        p = rf_arr[4]
        tps = np.einsum('i,j->ji', H_arr, (np.sqrt(1/(vs**2) - p**2) - np.sqrt(1/(avg_vp**2) - p**2)))
        tppps = np.einsum('i,j->ji', H_arr, (np.sqrt(1/(vs**2) - p**2) + np.sqrt(1/(avg_vp**2) - p**2)))
        tpsps = np.einsum('i,j->ji', H_arr, 2 * (np.sqrt(1/(vs**2) - p**2)))
        rf_sum = w1 * rf(tps) + w2 * rf(tppps) - w3 * rf(tpsps)
        
        S1_num += rf(tps)
        S1_deno += rf(tps)**2
    
        S2_num += rf(tppps)
        S2_deno += rf(tppps)**2
    
        S3_num += rf(tpsps)
        S3_deno += rf(tpsps)**2
        
        matrix += rf_sum
    
    S1 = S1_num**2 / S1_deno
    S2 = S2_num**2 / S2_deno
    S3 = S3_num**2 / S3_deno
    
    S = S1 + S2 + S3
    matrix = S * matrix
    
    maxy = np.where(matrix == np.max(matrix))[0][0]
    maxx = np.where(matrix == np.max(matrix))[1][0]
    H = H_arr[maxx]
    k = k_arr[maxy]

    return H_arr, k_arr, matrix

def save_rfs(stnm, a, c, rfs, outdir="rf/"):

        rfs_dict = {"station": stnm,
                    "deconvolution_parameters": {"a":a, "c":c},
                    "receiver_functions": rfs}
        
        pickle.dump(rfs_dict, open(outdir+"{}.pickle".format(stnm), "wb"))

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

def ccp_stack(rfs_map, evdata, min_x, max_x, min_y, max_y, dx=0.01, dy=0.01,
              model='iasp91', dz=0.1, max_depth=70):

    y = np.arange(min_y, max_y, dy)
    x = np.arange(min_x, max_x, dx)
    z = np.arange(0, max_depth, dz)

    stack = np.zeros((len(x), len(y), len(z)))
    counts = np.ones((len(x), len(y), len(z)))
    
    # Read earth model:
    with open("earth_models/{}.csv".format(model), 'r') as f:
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
    
    # Read receiver functions and perform stacking
    for stnm in rfs_map.keys():
        rfs = pickle.load(open(rfs_map[stnm], "rb"))
        stla = evdata["stations"][stnm]["lat"]
        stlo = evdata["stations"][stnm]["lon"]
        stel = evdata["stations"][stnm]["elev"]
        
        for rf_arr in rfs['receiver_functions']:
            eq_id = rf_arr[6]
            p = rf_arr[4]
            p_sph = p*r_earth
            t = rf_arr[1]
            rf = rf_arr[0]
            intp_rf = scint.interp1d(t, rf)
            baz = evdata["events"][eq_id]['stations'][stnm]["azimuth"] # esto esta "MAL" PQ NO SIGUE EL CONVENIO DE IRIS, DEBE USARSE EL BACK AZIMUTH MEDIDO DESDE EL EVENTO, QUE ES EL AZIMUTH MEDIDO DESDE LA ESTACION

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
        
                ddist = (r0-r) / np.sqrt(r**2/(p_sph**2*vs**2)-1) / r
                theta = 360*(ddist/(2*np.pi*r))
                ddist = 2*np.pi*r_earth*(theta/360)
                #dist += ddist
                #theta = 360*(ddist/(2*np.pi*r))
                #dist += 2*np.pi*r_earth*(theta/360)

                dt = (np.sqrt(vs**-2 - p**2) - np.sqrt(vp**-2 - p**2)) * dz
                tps += dt
                fzone = np.sqrt(H * (vs*T0))/111.2
    
                nlat = np.arcsin(np.sin(lat) * np.cos(ddist) + np.cos(lat) * np.sin(ddist) * np.cos(baz))
                nlon = lon + np.arctan2(np.sin(baz) * np.sin(ddist) * np.cos(lat), np.cos(ddist) - np.sin(lat) * np.sin(nlat))
                
                lat = nlat
                lon = nlon
                amp = intp_rf(tps)
                
                # stack
                # Determine cells using fresnell zone
                deglat = math.degrees(nlat)
                deglon = math.degrees(nlon)
                
                smthx = np.floor(fzone/dx)
                smthy = np.floor(fzone/dy)         

                ix = np.floor((deglon-x[0])/dx)
                iy = np.floor((deglat-y[0])/dy)
                for ix2 in range(int(ix-smthx), int(ix+smthx)):
                  if ix2 > 0 and ix2 < len(x):
                    for iy2 in range(int(iy-smthy), int(iy+smthy)):
                      if iy2 > 0 and iy2 < len(y):
                        stack[ix2,iy2,k] += amp
                        counts[ix2,iy2,k] += 1

                
                r0 = r        

    return stack

def compute_intermediate_points(start, end, npts):
    A_lats = start[1]
    A_lons = start[0]
    B_lats = end[1]
    B_lons = end[0]

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
    
    return newlats, newlons
        

def point_inside_polygon(x,y,poly):

    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def interpolate_ccp_stack(x, y, stack):
    
    interps = []
    for i in range(stack.shape[-1]):
        interps.append(scint.interp2d(x, y, stack[:,:,i], bounds_error=False,
                                      fill_value=np.NaN))
    
    return interps
        
def ccp_cross_section(x, y, z, stack, coords):
    pass
        
        
if __name__ == "__main__":
    eqs = map_earthquakes()
    rfs = compute_rfs("SC01", eqs, pickle.load(open("event_metadata", 'rb')))
    save_rfs("SC01", 2.5, 0.01, rfs)
    rfs_map = map_rfs()
    stack = ccp_stack(rfs_map, pickle.load(open("event_metadata", 'rb')),
                      -4.5, -4.0, 43.17, 43.57)
    
    
    
    
        