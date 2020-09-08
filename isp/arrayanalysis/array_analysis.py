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
from obspy.signal.array_analysis import array_transff_freqslowness
import nitime.algorithms as alg
from obspy.signal.array_analysis import get_geometry
import math
from scipy import fftpack
from nitime import utils
from datetime import date
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len





class array:

    def __init__(self):
        """
                ARF, (FK and Multitaper) Coherence program and Vespagram.



                :param No params required to initialize the class
        """





    def arf(self, coords, fmin, flim, slim, sgrid):


        sstep = sgrid
        fstep= flim / 40
        fmax=flim
        transff = array_transff_freqslowness(coords, slim, sstep, fmin, fmax, fstep, coordsys='lonlat')

        return transff

    def azimuth2mathangle(self, azimuth):
        if azimuth <= 90:
            mathangle = 90 - azimuth
        elif 90 < azimuth <= 180:
            mathangle = 270 + (180 - azimuth)
        elif 180 < azimuth <= 270:
            mathangle = 180 + (270 - azimuth)
        else:
            mathangle = 90 + (360 - azimuth)
        return mathangle

    def gregorian2date(self, DT):
        #####Convert start from Greogorian to actual date###############
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
        DATE = date1 + "T" + str(H1) + minutes1 + seconds
        t1 = UTCDateTime(DATE)
        return t1

    def FK(self, st, inv, stime, etime, fmin, fmax, slim, sres, win_len, win_frac):

        n = len(st)
        for i in range(n):
            coords = inv.get_coordinates(st[i].id)
            st[i].stats.coordinates = AttribDict(
                {'latitude': coords['latitude'], 'elevation': coords['elevation'], 'longitude': coords['longitude']})

        kwargs = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-1 * slim, slm_x=slim, sll_y=-1 * slim, slm_y=slim, sl_s=sres,
            # sliding open_main_window properties
            win_len=win_len, win_frac=win_frac,
            # frequency properties
            frqlow=fmin, frqhigh=fmax, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
            stime=stime+0.1, etime=etime-0.1)

        try:
            out = array_processing(st, **kwargs)


            T = out[:, 0]
            relpower = out[:, 1]
            abspower = out[:, 2]
            AZ = out[:, 3]
            AZ[AZ < 0.0] += 360
            Slowness = out[:, 4]

        except:
            print("Check Parameters and Starttime/Endtime")

            relpower = []
            abspower = []
            AZ = []
            Slowness = []
            T = []

        return relpower, abspower, AZ, Slowness, T

    def FKCoherence(self, st, inv,  DT, linf, lsup, slim, win_len, sinc, method):

        def find_nearest(array, value):

            idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
            return idx, val

        sides = 'onesided'
        pi = math.pi

        smax = slim
        smin = -1 * smax
        Sx = np.arange(smin, smax, sinc)[np.newaxis]
        Sy = np.arange(smin, smax, sinc)[np.newaxis]
        nx = ny = len(Sx[0])
        Sy = np.fliplr(Sy)

        #####Convert start from Greogorian to actual date###############
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
        DATE = date1 + "T" + str(H1) + minutes1 + seconds
        t1 = UTCDateTime(DATE)
        ########End conversion###############################

        st.trim(starttime=t1, endtime=t1 + win_len)
        st.sort()
        n = len(st)
        for i in range(n):
            coords = inv.get_coordinates(st[i].id)
            st[i].stats.coordinates = AttribDict(
                {'latitude': coords['latitude'], 'elevation': coords['elevation'], 'longitude': coords['longitude']})

        coord = get_geometry(st, coordsys='lonlat', return_center=True)

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
        tdata = tapers[None, :, :] * pdata[:, None, :]
        tspectra = fftpack.fft(tdata)

        w = np.empty((nr, int(K), int(nfft)))
        for i in range(nr):
            w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides=sides)

        nseq = nr
        L = int(nfft)
        #csd_mat = np.zeros((nseq, nseq, L), 'D')
        #psd_mat = np.zeros((2, nseq, nseq, L), 'd')
        coh_mat = np.zeros((nseq, nseq, L), 'd')
        #coh_var = np.zeros_like(coh_mat)
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
                    #Power
                    #out = A * np.conjugate(B)

                    #Relative Power
                    den=np.absolute(A)*np.absolute(np.conjugate(B))
                    out = (A * np.conjugate(B))/den

                    cxcohe = out[value1:value2]
                    Cx[i, j, :] = cxcohe

        r = np.zeros((nr, 2))
        S = np.zeros((1, 2))
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
        x = y = np.linspace(smin, smax, nx)

        nn=len(x)
        maximum_power=np.where(Pow == np.amax(Pow))
        Sxpow = (maximum_power[1]-nn/2)*sinc
        Sypow = (maximum_power[0]-nn/2)*sinc

        return Pow, Sxpow, Sypow, coord



    def stack_stream(self, st2, sx, sy, coord):
        sx = -1*sx
        sy = -1*sy
        s = np.array([sx, sy, 0])
        #st = read(path+ "/"+"*.*")
        x = []
        y = []
        r = []
        for i in range(len(coord) - 1):
            r.append(coord[i])
        for j in range(len(st2)):
            TAU = np.dot(r[j], s)
            TAU= TAU[0]
            st2[j].stats.starttime = st2[j].stats.starttime + TAU


        maxstart = np.max([tr.stats.starttime for tr in st2])
        minend = np.min([tr.stats.endtime for tr in st2])
        st2.trim(maxstart, minend)
        fs=st2[0].stats.sampling_rate
        time = np.linspace(0, fs , num=len(st2[0].data))
        mat = np.zeros([len(st2), len(st2[0].data)])
        N = len(st2)
        for i in range(N - 1):
            mat[i, :] = st2[i].data

        stats = {'network': st2[0].stats.station, 'station': 'STACK', 'location': '',
                      'channel': st2[0].stats.channel, 'npts': st2[0].stats.npts,
                      'sampling_rate':st2[0].stats.sampling_rate, 'mseed': {'dataquality': 'D'},
                 'starttime': st2[0].stats.starttime}

        return mat, time, stats

    def stack(self, data, stack_type = 'linear', order = 2):

        """
        Stack data by first axis.

        :type stack_type: str or tuple
        :param stack_type: Type of stack, one of the following:
            ``'linear'``: average stack (default),
            ``('pw', order)``: phase weighted stack of given order
            (see [Schimmel1997]_, order 0 corresponds to linear stack),
            ``('root', order)``: root stack of given order
            (order 1 corresponds to linear stack).
        """
        if stack_type == 'linear':
            stack = np.mean(data, axis=0)
        elif stack_type == 'PWS':
            npts = np.shape(data)[1]
            nfft = next_fast_len(npts)
            anal_sig = hilbert(data, N=nfft)[:, :npts]
            phase_stack = np.abs(np.mean(anal_sig, axis=0)) ** order
            stack = np.mean(data, axis=0) * phase_stack
        elif stack_type == 'root':
            r = np.mean(np.sign(data) * np.abs(data) ** (1 / order), axis=0)
            stack = np.sign(r) * np.abs(r) ** order
        else:
            raise ValueError('stack type is not valid.')

        return stack


class vespagram_util:
    
    def __init__(self, st, linf, lsup, win_len, slow, baz, inv, t1, t2, res, selection = "Slowness", method = "FK"):

        """
        Vespagram, computed a fixed slowness or backazimuth

        :param params required to initialize the class:
        linf min frequency (Hz)
        lsup max frequency (Hz)
        win_len win size in seconds to be analyzed in every step
        slow = slowness of analysis
        baz backazimuth of analysis
        """

        self.st = st
        self.t1 = t1
        self.t2 = t2
        self.inv = inv
        self.method = method
        self.linf = linf
        self.lsup = lsup
        self.win_len = win_len
        self.slow = slow
        self.baz = baz
        self.res = res
        self.selection = selection


    def azimuth2mathangle(self, azimuth):
        if azimuth <= 90:
            mathangle = 90 - azimuth
        elif 90 < azimuth <= 180:
            mathangle = 270 + (180 - azimuth)
        elif 180 < azimuth <= 270:
            mathangle = 180 + (270 - azimuth)
        else:
            mathangle = 90 + (360 - azimuth)
        return mathangle

    def vespa_deg(self):
        self.st.trim(starttime= self.t1, endtime=self.t2)

        return self.__vespa()


    def __vespa(self):

        win_len = int(self.win_len*self.st[0].stats.sampling_rate)
        npts = len(self.st[0].data)
        delta = self.st[0].stats.delta

        lim = npts - win_len


        x = ((self.res)/100)*win_len
        steps = np.arange(0, lim, npts/x)

        vespa_spec = np.zeros([360, len(steps)])
        j= 0
        for n in np.nditer(steps):

            st = self.st.copy()
            start_time = self.t1 + n*delta
            end_time = self.t1+self.win_len + n*delta
            st.trim(starttime=start_time, endtime=end_time)
            if self.selection == "Slowness":
                slow_spectrum = self.__vespa_slow(st)
            else:
                slow_spectrum = self.__vespa_az(st)

            vespa_spec[:, j] = slow_spectrum
            j = j + 1

        log_vespa_spectrogram = 10. * np.log(vespa_spec / np.max(vespa_spec))
        t = np.linspace(0, self.t2-self.t1, log_vespa_spectrogram.shape[1])

        if self.selection == "Slowness":
            x, y = np.meshgrid(t, np.linspace(0, 360, log_vespa_spectrogram.shape[0]))
        else:
            x, y = np.meshgrid(t, np.linspace(-1*self.slow, self.slow, log_vespa_spectrogram.shape[0]))

        return x, y, log_vespa_spectrogram



    def __vespa_slow(self, st):

        def find_nearest(array, value):

            idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
            return idx, val

        def azimuth2mathangle(azimuth):
            if azimuth <= 90:
                mathangle = 90 - azimuth
            elif 90 < azimuth <= 180:
                mathangle = 270 + (180 - azimuth)
            elif 180 < azimuth <= 270:
                mathangle = 180 + (270 - azimuth)
            else:
                mathangle = 90 + (360 - azimuth)
            return mathangle

        sides = 'onesided'
        pi = math.pi
        st.sort()
        n = len(st)
        for i in range(n):
            coords = self.inv.get_coordinates(st[i].id)
            st[i].stats.coordinates = AttribDict(
                {'latitude': coords['latitude'], 'elevation': coords['elevation'], 'longitude': coords['longitude']})

        coord = get_geometry(st, coordsys='lonlat', return_center=True)

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

        value1, freq1 = find_nearest(freq, self.linf)
        value2, freq2 = find_nearest(freq, self.lsup)
        df = value2 - value1
        m = np.zeros((win, nr))

        WW=np.hamming(int(win))
        WW=np.transpose(WW)
        for i in range(nr):
            tr = st[i]
            if self.method == "FK":
                m[:,i]=(tr.data-np.mean(tr.data))*WW
            else:
                m[:, i] = (tr.data - np.mean(tr.data))
        pdata = np.transpose(m)

        #####Coherence######
        NW = 2  # the time-bandwidth product##Buena seleccion de 2-3
        K = 2 * NW - 1
        tapers, eigs = alg.dpss_windows(win, NW, K)
        tdata = tapers[None, :, :] * pdata[:, None, :]
        tspectra = fftpack.fft(tdata)

        w = np.empty((nr, int(K), int(nfft)))
        for i in range(nr):
            w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides=sides)


        Cx = np.ones((nr, nr, df), dtype=np.complex128)

        if self.method == "MTP.COHERENCE":
            for i in range(nr):
                for j in range(nr):
                    sxy = alg.mtm_cross_spectrum(tspectra[i], (tspectra[j]), (w[i], w[j]), sides='onesided')
                    sxx = alg.mtm_cross_spectrum(tspectra[i], tspectra[i], w[i], sides='onesided')
                    syy = alg.mtm_cross_spectrum(tspectra[j], tspectra[j], w[j], sides='onesided')
                    s = sxy / np.sqrt((sxx * syy))
                    cxcohe = s[value1:value2]
                    Cx[i, j, :] = cxcohe

        ####Calculates Conventional FK-power  ##without normalization
        if self.method == "FK":
            for i in range(nr):
                for j in range(nr):
                    A = np.fft.rfft(m[:, i])
                    B = np.fft.rfft(m[:, j])
                    #Power
                    #out = A * np.conjugate(B)

                    #Relative Power
                    den=np.absolute(A)*np.absolute(np.conjugate(B))
                    out = (A * np.conjugate(B))/den

                    cxcohe = out[value1:value2]
                    Cx[i, j, :] = cxcohe

        r = np.zeros((nr, 2))
        S = np.zeros((1, 2))
        Pow = np.zeros(360)
        for n in range(nr):
            r[n, :] = coord[n][0:2]

        freq = freq[value1:value2]

        rad = np.pi/180
        for j in range(360):
            j = azimuth2mathangle(j)
            S[0, 0] = self.slow*np.cos(rad * (j))
            S[0, 1] = self.slow*np.sin(rad * (j))
            k = (S * r)
            K = np.sum(k, axis=1)
            n = 0
            for f in freq:
                A = np.exp(-1j * 2 * pi * f) ** K
                B = np.conjugate(np.transpose(A))
                D = np.matmul(B, Cx[:, :, n]) / nr
                P = np.matmul(D, A) / nr
                Pow[j] += np.abs(P)
                n = n + 1


        Pow = Pow / len(freq)

        return Pow

    def __vespa_az(self, st):

        def find_nearest(array, value):

            idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
            return idx, val

        def azimuth2mathangle(azimuth):
            if azimuth <= 90:
                mathangle = 90 - azimuth
            elif 90 < azimuth <= 180:
                mathangle = 270 + (180 - azimuth)
            elif 180 < azimuth <= 270:
                mathangle = 180 + (270 - azimuth)
            else:
                mathangle = 90 + (360 - azimuth)
            return mathangle

        sides = 'onesided'
        pi = math.pi
        st.sort()
        n = len(st)
        for i in range(n):
            coords = self.inv.get_coordinates(st[i].id)
            st[i].stats.coordinates = AttribDict(
                {'latitude': coords['latitude'], 'elevation': coords['elevation'], 'longitude': coords['longitude']})

        coord = get_geometry(st, coordsys='lonlat', return_center=True)

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

        value1, freq1 = find_nearest(freq, self.linf)
        value2, freq2 = find_nearest(freq, self.lsup)
        df = value2 - value1
        m = np.zeros((win, nr))

        WW=np.hamming(int(win))
        WW=np.transpose(WW)
        for i in range(nr):
            tr = st[i]
            if self.method == "FK":
                m[:,i]=(tr.data-np.mean(tr.data))*WW
            else:
                m[:, i] = (tr.data - np.mean(tr.data))
        pdata = np.transpose(m)

        #####Coherence######
        NW = 2  # the time-bandwidth product##Buena seleccion de 2-3
        K = 2 * NW - 1
        tapers, eigs = alg.dpss_windows(win, NW, K)
        tdata = tapers[None, :, :] * pdata[:, None, :]
        tspectra = fftpack.fft(tdata)

        w = np.empty((nr, int(K), int(nfft)))
        for i in range(nr):
            w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides=sides)

        Cx = np.ones((nr, nr, df), dtype=np.complex128)

        if self.method == "MTP.COHERENCE":
            for i in range(nr):
                for j in range(nr):
                    sxy = alg.mtm_cross_spectrum(tspectra[i], (tspectra[j]), (w[i], w[j]), sides='onesided')
                    sxx = alg.mtm_cross_spectrum(tspectra[i], tspectra[i], w[i], sides='onesided')
                    syy = alg.mtm_cross_spectrum(tspectra[j], tspectra[j], w[j], sides='onesided')
                    s = sxy / np.sqrt((sxx * syy))
                    cxcohe = s[value1:value2]
                    Cx[i, j, :] = cxcohe

        ####Calculates Conventional FK-power  ##without normalization
        if self.method == "FK":
            for i in range(nr):
                for j in range(nr):
                    A = np.fft.rfft(m[:, i])
                    B = np.fft.rfft(m[:, j])
                    #Power
                    #out = A * np.conjugate(B)

                    #Relative Power
                    den=np.absolute(A)*np.absolute(np.conjugate(B))
                    out = (A * np.conjugate(B))/den

                    cxcohe = out[value1:value2]
                    Cx[i, j, :] = cxcohe

        r = np.zeros((nr, 2))
        S = np.zeros((1, 2))
        Pow = np.zeros(360)
        for n in range(nr):
            r[n, :] = coord[n][0:2]

        freq = freq[value1:value2]

        rad = np.pi/180
        #slow_range = np.arange(-1*self.slow, self.slow, self.slow/360)
        slow_range = np.linspace(-1*self.slow, self.slow, 360)

        for j in range(360):

            S[0, 0] = slow_range[j]*np.cos(rad * self.baz)
            S[0, 1] = slow_range[j]*np.sin(rad * self.baz)
            k = (S * r)
            K = np.sum(k, axis=1)
            n = 0
            for f in freq:
                A = np.exp(-1j * 2 * pi * f) ** K
                B = np.conjugate(np.transpose(A))
                D = np.matmul(B, Cx[:, :, n]) / nr
                P = np.matmul(D, A) / nr
                Pow[j] += np.abs(P)
                n = n + 1


        Pow = Pow / len(freq)

        return Pow
