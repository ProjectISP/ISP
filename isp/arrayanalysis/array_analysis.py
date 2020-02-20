#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Feb  6 12:03:39 2018
    @author: robertocabieces
"""

import numpy as np
from obspy.core import read
from obspy.core import UTCDateTime
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
import pandas as pd
import os
from obspy.signal.array_analysis import array_transff_freqslowness
import nitime.algorithms as alg
import matplotlib.pyplot as plt
from obspy.signal.array_analysis import get_geometry
import math
from scipy import fftpack
from nitime import utils
from datetime import date

class array:

    def __init__(self):
        """
                Manage nll files for run nll program.

                Important: The  obs_file_path is provide by the class :class:`PickerManager`.

                :param obs_file_path: The file path of pick observations.
        """
        # self.__dataless_dir = dataless_path
        # self.__obs_file_path = obs_file_path
        # self.__create_dirs()


    def arf(self,path,fmin,flim,slim):

        data=np.loadtxt(path,skiprows=1,usecols = (1,2,3))
        n=len(data)
        coords=np.zeros([n,3])
        for i in range(n):
            coords[i]=data[i]
        sstep = slim / 50
        fstep= flim / 50
        fmax=flim
        transff=array_transff_freqslowness(coords, slim, sstep, fmin, fmax, fstep, coordsys='lonlat')

        return transff,coords

    def FK(self, path, path_coords, stime, etime, fmin, fmax, slim, sres, win_len, win_frac):

        diff=etime-stime
        path = path+"/*.*"
        path_coords = os.path.join(path_coords,"coords.txt")

        st = read(path)
        maxstart = np.max([tr.stats.starttime for tr in st])
        minend = np.min([tr.stats.endtime for tr in st])
        #st.trim(maxstart, minend)
        #st.trim(stime, etime)
        print(st)
        df = pd.read_csv(path_coords, sep='\t')
        n = df.Name.count()

        for i in range(n):
            st[i].stats.coordinates = AttribDict(
                {'latitude': df.loc[i].Lat, 'elevation': 0.0, 'longitude': df.loc[i].Lon})
        # coord =get_geometry(st, coordsys='lonlat', return_center=True, verbose=True)
        #tr = st[0]
        #delta = tr.stats.delta
        fs = st[0].stats.sampling_rate
        #stime = UTCDateTime(stime)
        #etime = stime + DT
        DT = etime-stime
        print("Computing FK")
        kwargs = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-1 * slim, slm_x=slim, sll_y=-1 * slim, slm_y=slim, sl_s=sres,
            # sliding open_main_window properties
            win_len=win_len, win_frac=win_frac,
            # frequency properties
            frqlow=fmin, frqhigh=fmax, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
            stime=stime, etime=etime)

        #nsamp = int(win_len * fs)
        #nstep = int(nsamp * win_frac)
        print(stime)
        print(etime)
        out = array_processing(st, **kwargs)
        print("Finished")
        #xlocator = mdates.AutoDateLocator()
        T = out[:, 0]
        relpower = out[:, 1]
        abspower = out[:, 2]
        AZ = out[:, 3]
        AZ[AZ < 0.0] += 360
        Slowness = out[:, 4]
        #time = np.linspace(0, DT, num=len(T))
        return relpower, abspower, AZ, Slowness, T

    def FKCoherence(self, path, path_coords, start, DT, linf, lsup, slim, win_len, sinc, method):
        #print(start)
        print(DT)
        #print(win_len)
        def find_nearest(array, value):

            idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
            return idx, val

        path = path + "/" + "*.*"
        path_coords = path_coords + "/" + "coords.txt"
        sides = 'onesided'
        pi = math.pi
        rad = 180 / pi

        smax = slim
        smin = -1 * smax

        Sx = np.arange(smin, smax, sinc)[np.newaxis]
        Sy = np.arange(smin, smax, sinc)[np.newaxis]

        nx = ny = len(Sx[0])
        Sy = np.fliplr(Sy)


        #####Convert start from Greogorian to actual date###############

        # Time = d.timetuple()
        Time = DT

        Time = Time - int(Time)
        d = date.fromordinal(int(DT))
        date1 = d.isoformat()

        H = (Time * 24)
        H1 = int(H)  # Horas
        minutes = (H - int(H)) * 60
        minutes1 = int(minutes)
        seconds = (minutes - int(minutes)) * 60

        H1 = str(H1).zfill(2)
        minutes1 = str(minutes1).zfill(2)
        seconds = "%.2f" % seconds
        seconds = str(seconds).zfill(2)

        ##
        # time.struct_time(tm_year=2002, tm_mon=3, tm_mday=11, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=0, tm_yday=70, tm_isdst=-1)
        ##Build the initial date

        DATE = date1 + "T" + str(H1) + minutes1 + seconds
        print(DATE)
        t1 = UTCDateTime(DATE)


        #st = read(path, starttime=start+DT, endtime=start+DT+win_len)
        st = read(path, starttime=t1, endtime=t1 + win_len)
        print(st)
        st.sort()
        df = pd.read_csv(path_coords, sep='\t')
        n = df.Name.count()

        for i in range(n):
            st[i].stats.coordinates = AttribDict(
                {'latitude': df.loc[i].Lat, 'elevation': 0.0, 'longitude': df.loc[i].Lon})

        coord = get_geometry(st, coordsys='lonlat', return_center=True)
        # =============================
        tr = st[0]
        win = len(tr.data)
        if (win % 2) == 0:
            nfft = win / 2 + 1
        else:
            nfft = (win + 1) / 2

        nr = st.count()  # number of stations
        delta = st[0].stats.delta
        fs = 1 / delta
        fn = fs / 2
        freq = np.arange(0, fn, fn / nfft)

        value1, freq1 = find_nearest(freq, linf)
        value2, freq2 = find_nearest(freq, lsup)
        df = value2 - value1
        m = np.zeros((win, nr))

        WW=np.hamming(int(win))
        WW=np.transpose(WW)
        for i in range(nr):
            tr = st[i]
            if method == "FK":
                m[:,i]=(tr.data-np.mean(tr.data))*WW
            else:
                m[:, i] = (tr.data - np.mean(tr.data))
        pdata = np.transpose(m)
        #####Coherence######
        NW = 2  # the time-bandwidth product##Buena seleccion de 2-3
        K = 2 * NW - 1
        tapers, eigs = alg.dpss_windows(win, NW, K)
        tdata = tapers[None, :, :] * pdata[:, None, :]  # filas estaciones,
        # columnas por tapers, profundiadad data
        tspectra = fftpack.fft(tdata)

        w = np.empty((nr, int(K), int(nfft)))
        for i in range(nr):
            w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides=sides)

        nseq = nr
        L = int(nfft)
        csd_mat = np.zeros((nseq, nseq, L), 'D')
        psd_mat = np.zeros((2, nseq, nseq, L), 'd')
        coh_mat = np.zeros((nseq, nseq, L), 'd')
        coh_var = np.zeros_like(coh_mat)
        Cx = np.ones((nr, nr, df), dtype=np.complex128)

        if method == "MTP.COHERENCE":
            for i in range(nr):
                for j in range(nr):
                    sxy = alg.mtm_cross_spectrum(tspectra[i], (tspectra[j]), (w[i], w[j]), sides='onesided')
                    sxx = alg.mtm_cross_spectrum(tspectra[i], tspectra[i], w[i], sides='onesided')
                    syy = alg.mtm_cross_spectrum(tspectra[j], tspectra[j], w[j], sides='onesided')
                    s = sxy / np.sqrt((sxx * syy))
                    cxcohe = s[value1:value2]
                    Cx[i, j, :] = cxcohe

        ####Calculates Conventional FK-power  ##without normalization
        if method == "FK":
            for i in range(nr):
                for j in range(nr):
                    A = np.fft.rfft(m[:, i])
                    B = np.fft.rfft(m[:, j])
                    out = A * np.conjugate(B)

                    cxcohe = out[value1:value2]
                    Cx[i, j, :] = cxcohe

        r = np.zeros((nr, 2))
        A = np.zeros((nr, 1), dtype=np.complex128)
        S = np.zeros((1, 2))
        K = np.zeros((1, 5))
        Pow = np.zeros((len(Sx[0]), len(Sy[0])))
        for n in range(nr):
            r[n, :] = coord[n][0:2]

        freq = freq[value1:value2]

        for i in range(ny):
            for j in range(nx):
                S[0, 0] = Sx[0][j]
                S[0, 1] = Sy[0][i]
                k = (S * r)
                K = np.sum(k, axis=1)
                n = 0
                for f in freq:
                    A = np.exp(-1j * 2 * pi * f) ** K
                    B = np.conjugate(np.transpose(A))
                    D = np.matmul(B, Cx[:, :, n]) / nr
                    P = np.matmul(D, A) / nr
                    Pow[i, j] += np.abs(P)
                    n = n + 1

        Pow = Pow / len(freq)
        Pow = np.fliplr(Pow)
        # Plotting part

        # Pow = np.flipud(Pow)
        #
        # fig, ax = plt.subplots()
        x = y = np.linspace(smin, smax, nx)
        X, Y = np.meshgrid(x, y)
        # plt.contourf(X, Y, Pow, 50, cmap=plt.cm.jet)
        # cs = ax.contourf(X, Y, Pow, 50, cmap=plt.cm.jet)
        # plt.ylabel('Sy [s/km]')
        # plt.xlabel('Sx [s/km]')
        # cbar = fig.colorbar(cs)
        # plt.show()

        return X , Y, Pow





