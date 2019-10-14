
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:07:15 2018

@author: robertocabieces
"""

import matplotlib.pyplot as plt
import numpy as np
from obspy import read

from isp.Structures.structures import TracerStats
from isp.Utils import MseedUtil, ObspyUtil
from isp.seismogramInspector.Auxiliary2 import MTspectrum


class MTspectrogram:

    def __init__(self, file_path: str, win, tbp, ntapers, fmin, fsup):
        self.file_path = file_path
        self.fmin = fmin
        self.fsup = fsup
        self.win = win
        self.tbp = tbp
        self.ntapers = ntapers

        self.mseed_files = MseedUtil.get_mseed_files(file_path)

    def compute_spectrogram(self, trace_data, stats: TracerStats):
        t = np.linspace(0, (stats.Delta * stats.Npts), stats.Npts - self.win)
        mt_spectrum = MTspectrum(trace_data, self.win, stats.Delta, self.tbp, self.ntapers, self.fmin, self.fsup)
        log_spectrogram = 10. * np.log(mt_spectrum / np.max(mt_spectrum))
        x, y = np.meshgrid(t, np.linspace(self.fmin, self.fsup, log_spectrogram.shape[0]))
        return x, y, log_spectrogram

    def plot_spectrogram(self, show=True):
        nfilas = len(self.mseed_files)
        k = 1
        fig = plt.figure()
        for file in self.mseed_files:
            tr = ObspyUtil.get_tracer_from_file(file)
            stats = TracerStats.from_dict(tr.stats)
            # Obs: after doing tr.detrend it add processing key to stats
            tr.detrend(type="demean")
            x, y, log_spectrogram = self.compute_spectrogram(tr.data, stats)

            fig.add_subplot(nfilas, 1, k)
            cs = plt.contourf(x, y, log_spectrogram, 100, cmap=plt.jet())
            plt.title("Multi Taper Spectrogram" + stats.Station)
            plt.xlabel("Time after %s [s]" % stats.StartTime)
            plt.ylabel("Frequency [Hz]")
            plt.colorbar(cs)
            k += 1
        if show:
            plt.plot()
        else:
            return fig



# def MTspectrogram(ficheros_procesar_path,win,tbp,ntapers,fmin,fsup):
#     from isp.Structures.structures import TracerStats
#
#     obsfiles = MseedUtil.get_mseed_files(ficheros_procesar_path)
#     nfilas=len(obsfiles)
#     k=1
#     fig = plt.figure()
#     for f in obsfiles:
#         if f != ".DS_Store":
#             st1=read(f)
#             tr = st1[0]
#             stats = TracerStats.from_dict(tr.stats)
#             # Obs: after doing tr.detrend it add processing key to stats
#             tr.detrend(type="demean")
#             t = np.linspace(0, (stats.Delta * stats.Npts), stats.Npts-win)
#             mtspectrogram=MTspectrum(tr.data, win, stats.Delta, tbp, ntapers, fmin, fsup)
#             M=np.max(mtspectrogram)
#             mtspectrogram2=10*np.log(mtspectrogram/M)
#             x, y = np.meshgrid(t,np.linspace(fmin, fsup, mtspectrogram2.shape[0]))
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
