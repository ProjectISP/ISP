
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:07:15 2018

@author: robertocabieces
"""

import matplotlib.pyplot as plt
import numpy as np
from obspy import Trace, read

from isp.Exceptions import InvalidFile
from isp.Structures.structures import TracerStats
from isp.Utils import ObspyUtil, MseedUtil
from isp.seismogramInspector.signal_processing_advanced import MTspectrum
from mtspec import wigner_ville_spectrum, mtspec
import math

class MTspectrogram:

    def __init__(self, file_path, win, tbp, ntapers, f_min, f_max):
        self.f_min = f_min
        self.f_max = f_max
        self.win = win
        self.tbp = tbp
        self.ntapers = ntapers

        self.file_path = file_path
        self.__stats = TracerStats()
        if not MseedUtil.is_valid_mseed(file_path):
            raise InvalidFile

    @property
    def stats(self):
        return self.__stats

    # def __compute_spectrogram(self, trace_data):
    #     npts = len(trace_data)
    #     t = np.linspace(0, (self.stats.Delta * npts), npts - self.win)
    #     mt_spectrum = MTspectrum(trace_data, self.win, self.stats.Delta, self.tbp, self.ntapers, self.f_min, self.f_max)
    #     log_spectrogram = 10. * np.log(mt_spectrum / np.max(mt_spectrum))
    #     x, y = np.meshgrid(t, np.linspace(self.f_min, self.f_max, log_spectrogram.shape[0]))
    #     return x, y, log_spectrogram

    # def compute_spectrogram(self, start_time=None, end_time=None, trace_filter=None):
    #     tr: Trace = ObspyUtil.get_tracer_from_file(self.file_path)
    #     self.__stats = TracerStats.from_dict(tr.stats)
    #     tr.trim(starttime=start_time, endtime=end_time)
    #     tr.detrend(type="demean")
    #     ObspyUtil.filter_trace(tr, trace_filter, self.f_min, self.f_max)
    #     x, y, log_spectrogram = self.__compute_spectrogram(tr.data)
    #     return x, y, log_spectrogram

    def __compute_spectrogram(self, tr):
         npts = len(tr)
         t = np.linspace(0, (tr.stats.delta * npts), npts - self.win)
         mt_spectrum = MTspectrum(tr.data, self.win, tr.stats.delta, self.tbp, self.ntapers, self.f_min, self.f_max)
         log_spectrogram = 10. * np.log(mt_spectrum / np.max(mt_spectrum))
         x, y = np.meshgrid(t, np.linspace(self.f_min, self.f_max, log_spectrogram.shape[0]))
         return x, y, log_spectrogram



    def compute_spectrogram(self, tr, start_time=None, end_time=None):

         tr.trim(starttime=start_time, endtime=end_time)

         x, y, log_spectrogram = self.__compute_spectrogram(tr)
         return x, y, log_spectrogram

    def plot_spectrogram(self, tr, fig=None, show=False):
        x, y, log_spectrogram = self.compute_spectrogram(tr)

        if not fig:
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)

        cs = plt.contourf(x, y, log_spectrogram, 100, cmap=plt.jet())
        plt.title("Multi Taper Spectrogram" + self.stats.Station)
        plt.xlabel("Time after %s [s]" % self.stats.StartTime)
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(cs)
        if show:
            plt.plot()
        else:
            return fig

    def plot_spectrograms(self, mseed_files, show=True):
        nfilas = len(mseed_files)
        k = 1
        fig = plt.figure()
        for file in mseed_files:
            tr = ObspyUtil.get_tracer_from_file(file)
            fig.add_subplot(nfilas, 1, k)
            self.plot_spectrogram(tr, fig)
            k += 1
        if show:
            plt.plot()
        else:
            return fig


class  WignerVille:

    def __init__(self, file_path, win, tbp, ntapers, f_min, f_max):

        self.f_min = f_min
        self.f_max = f_max
        self.win = win
        self.tbp = tbp
        self.ntapers = ntapers

        self.file_path = file_path
        self.__stats = TracerStats()
        if not MseedUtil.is_valid_mseed(file_path):
            raise InvalidFile

    def find_nearest(self, array, value):
        idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
        return idx, val


    def mt_wigner_wille_spectrum(self, data1, win, dt, tbp, ntapers, linf, lsup):


        #data2 = np.zeros(2 ** math.ceil(math.log2(win)))
        data1 = data1 - np.mean(data1)
        #data2[0:win] = data1
        spec, freq = mtspec(data1, dt, 3.5)
        wv = wigner_ville_spectrum(data1, dt, tbp, ntapers, smoothing_filter='gauss')

        value1, freq1 = self.find_nearest(freq, linf)
        value2, freq2 = self.find_nearest(freq, lsup)

        wv = wv[value1:value2]

        return wv


    def __compute_spectrogram(self, tr):
        npts = len(tr)
        t = np.linspace(0, (tr.stats.delta * npts), npts)
        mt_wigner_spectrum = self.mt_wigner_wille_spectrum(tr.data, self.win, tr.stats.delta, self.tbp, self.ntapers, self.f_min, self.f_max)
        mt_wigner_spectrum = np.flipud(mt_wigner_spectrum)
        mt_wigner_spectrum =np.sqrt(abs(mt_wigner_spectrum))
        #log_spectrogram = 10. * np.log(mt_wigner_spectrum / np.max(mt_wigner_spectrum))
        mt_wigner_spectrum = mt_wigner_spectrum / np.max(mt_wigner_spectrum)
        x, y = np.meshgrid(t, np.linspace(self.f_min, self.f_max, mt_wigner_spectrum.shape[0]))
        return x, y, mt_wigner_spectrum


    def compute_wigner_spectrogram(self, tr, start_time=None, end_time=None):
        tr.trim(starttime=start_time, endtime=end_time)
        x, y, log_spectrogram = self.__compute_spectrogram(tr)
        return x, y, log_spectrogram






# def MTspectrogram(ficheros_procesar_path,win,tbp,ntapers,f_min,f_max):
#     from isp.Structures.structures import TracerStats
#
#     obsfiles = MseedUtil.get_mseed_files(ficheros_procesar_path)
#     nfilas=len(obsfiles)
#     k=1
#     fig = plt.figure()
#     for f in obsfiles:
#         if f != ".DS_Store":
#             st1=read(f)
#             trace = st1[0]
#             stats = TracerStats.from_dict(trace.stats)
#             # Obs: after doing trace.detrend it add processing key to stats
#             trace.detrend(type="demean")
#             t = np.linspace(0, (stats.Delta * stats.Npts), stats.Npts-win)
#             mtspectrogram=MTspectrum(trace.data, win, stats.Delta, tbp, ntapers, f_min, f_max)
#             M=np.max(mtspectrogram)
#             mtspectrogram2=10*np.log(mtspectrogram/M)
#             x, y = np.meshgrid(t,np.linspace(f_min, f_max, mtspectrogram2.shape[0]))
#
#             # plt.ion()
#             # subplot = plt.subplot(nfilas, 1, k)
#             fig.add_subplot(nfilas, 1, k)
#             cs = plt.contourf(x, y, mtspectrogram2, 100, cmap=plt.jet())
#             plt.title("Multi Taper Spectrogram" + stats.Station)
#             plt.xlabel("Time after %s [s]" % stats.StartTime)
#             plt.ylabel("Frequency [Hz]")
#             plt.colorbar(cs)
#             k += 1
#
#     return fig

    # st=read(ficheros_procesar_path+"/"+"*.*")
    # #st.plot()
    # mp = MatplotlibFrame(st)
    # mp.show()
