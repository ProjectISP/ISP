
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:07:15 2018

@author: robertocabieces
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from scipy import ndimage
from scipy.signal import hilbert
from isp.Exceptions import InvalidFile
from isp.Structures.structures import TracerStats
from isp.Utils import ObspyUtil, MseedUtil
from isp.seismogramInspector.signal_processing_advanced import MTspectrum
from mtspec import wigner_ville_spectrum, mtspec, mt_coherence

class MTspectrogram:

    def __init__(self, file_path, win, tbp, ntapers, f_min, f_max):
        self.f_min = f_min
        self.f_max = f_max
        self.win = win
        self.tbp = tbp
        self.ntapers = ntapers

        self.file_path = file_path
        self.__stats = TracerStats()
        try:
            read(file_path)
        except:
        #if not MseedUtil.is_valid_mseed(file_path):
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

    def __compute_spectrogram(self, tr, res):
        npts = len(tr)

        mt_spectrum = MTspectrum(tr.data, self.win, tr.stats.delta, self.tbp, self.ntapers, self.f_min, self.f_max)
        log_spectrogram = 10. * np.log(mt_spectrum / np.max(mt_spectrum))

        if res > 1:
             log_spectrogram = ndimage.zoom(log_spectrogram, (1.0, 1 / log_spectrogram))
             t = np.linspace(0, res * tr.stats.delta * log_spectrogram.shape[1], log_spectrogram.shape[1])
             f = np.linspace(self.f_min, self.f_max, log_spectrogram.shape[0])
             x, y = np.meshgrid(t, f)
        else:
             t = np.linspace(0, (tr.stats.delta * npts), npts - self.win)
             f = np.linspace(self.f_min, self.f_max, log_spectrogram.shape[0])
             x, y = np.meshgrid(t, f)

        return x, y, log_spectrogram



    def compute_spectrogram(self, tr, res = 1, start_time=None, end_time=None):

         tr.trim(starttime=start_time, endtime=end_time)

         x, y, log_spectrogram = self.__compute_spectrogram(tr, res)
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


    def __compute_spectrogram(self, tr, res):
        npts = len(tr)
        mt_wigner_spectrum = self.mt_wigner_wille_spectrum(tr.data, self.win, tr.stats.delta, self.tbp, self.ntapers, self.f_min, self.f_max)
        mt_wigner_spectrum = np.flipud(mt_wigner_spectrum)
        mt_wigner_spectrum =np.sqrt(abs(mt_wigner_spectrum))
        mt_wigner_spectrum = mt_wigner_spectrum / np.max(mt_wigner_spectrum)
        if res > 1:
             log_spectrogram = ndimage.zoom(mt_wigner_spectrum, (1.0, 1 / mt_wigner_spectrum))
             t = np.linspace(0, res * tr.stats.delta * log_spectrogram.shape[1], log_spectrogram.shape[1])
             f = np.linspace(self.f_min, self.f_max, log_spectrogram.shape[0])
             x, y = np.meshgrid(t, f)
        else:
             t = np.linspace(0, (tr.stats.delta * npts), npts)
             f = np.linspace(self.f_min, self.f_max, mt_wigner_spectrum.shape[0])
             x, y = np.meshgrid(t, f)
        return x, y, mt_wigner_spectrum


    def compute_wigner_spectrogram(self, tr, start_time=None, end_time=None):
        tr.trim(starttime=start_time, endtime=end_time)
        x, y, log_spectrogram = self.__compute_spectrogram(tr)
        return x, y, log_spectrogram

class cross_spectrogram:

    def __init__(self, tr1, tr2, win, tbp, ntapers):
        self.tr1 = tr1
        self.tr2 = tr2
        self.win = win
        self.tbp = tbp
        self.ntapers = ntapers

    def resampling_trace(self):
        fs1 = self.tr1.stats.sampling_rate
        fs2 = self.tr2.stats.sampling_rate
        self.fs = max(fs1, fs2)
        if fs1 < self.fs:
            self.tr1.resample(self.fs)
        elif fs2 < self.fs:
            self.tr2.resample(self.fs)

    def __compute_coherence_cross(self):
        """
        :param tr1:
        :param tr2:
        :param sampling_rate:
        :param tbp:
        :param kspec:
        :param nf:
        :param p:
        :return: Dictionary

        Parameters:
        freq – the frequency bins
        cohe – coherence of the two series (0 - 1)
        phase – the phase at each frequency
        speci – spectrum of first series
        specj – spectrum of second series
        conf – p confidence value for each freq.
        cohe_ci – 95% bounds on coherence (not larger than 1)
        phase_ci – 95% bounds on phase estimates
        """
        self.resampling_trace()
        win = int(self.win*self.fs)
        if (win % 2) == 0:
            nfft = win / 2 + 1
        else:
            nfft = (win + 1) / 2

        data_tr1 = self.tr1.data
        data_tr2 = self.tr2.data
        npts = len(data_tr1)
        data_tr2=data_tr2[0:npts]
        lim = len(data_tr1) - win
        S = np.zeros([int(nfft), int(lim)])
        t = np.linspace(0, (npts/self.fs), npts - win)

        for n in range(lim):
            data1 = data_tr1[n:win + n]
            data1 = data1 - np.mean(data1)
            data1[0:win] = data1
            data2 = data_tr2[n:win + n]
            data2 = data2 - np.mean(data2)
            data2[0:win] = data2
            coherence=mt_coherence(1/self.fs, data1, data2, int(self.tbp), int(self.ntapers), int(nfft), p =.9,
                                       freq=True, cohe=True, iadapt=1 )
            spec= coherence['cohe']

            S[:, n] = spec

        freq = coherence['freq']
        #print("Dimensions", S.shape[1],len(t),n,lim)
        return S, freq, t

    def compute_coherence_crosspectrogram(self):

         [coherence_cross, freq, t] = self.__compute_coherence_cross()

         return coherence_cross, freq, t



class hilbert_gauss:

      def __init__(self,tr, f1, f2, df):

          self.tr = tr
          self.f1 = f1
          self.f2 = f2
          self.df = df


      def compute_filter_bank(self):

          f = np.arange(self.f1, self.f2, self.df)
          N = self.tr.stats.npts
          self.envelope = np.zeros([len(f)-1,N])
          self.phase = np.zeros([len(f)-1,N])
          self.inst_freq = np.zeros([len(f)-1,N-1])

          for k in range(len(f)-1):

              tr1 = self.tr.copy()
              tr1.filter(type= "bandpass", freqmin=f[k], freqmax=f[k+1], corners=4, zerophase=True)
              data_envelope, phase, inst_freq = self.__envelope(tr1.data)
              self.envelope[k,:] = data_envelope
              self.phase[k,:] = phase
              self.inst_freq[k,:] = np.abs(inst_freq)
          inst_freq_hz = self.inst_freq*self.tr.stats.sampling_rate/2*np.pi
          return self.envelope, self.phase, self.inst_freq, inst_freq_hz, f

      def envelope_db(self):

          envelope = 10 *np.log(self.envelope/np.max(self.envelope))

          return envelope



      def __envelope(self, data):
          N = len(data)
          D = 2 ** math.ceil(math.log2(N))
          z = np.zeros(D - N)
          data = np.concatenate((data, z), axis=0)
          ###Necesary padding with zeros
          analytic = hilbert(data)
          data_envelope = np.abs(analytic[0:N])
          phase = np.angle(analytic[0:N])
          x = np.where(phase < 0)
          phase[x] = 2 * np.pi + phase[x]
          inst_freq = np.diff(phase)
          return data_envelope, phase, inst_freq


      def phase(self):

          phase = np.angle(self.map)

          x = np.where(phase < 0)
          phase[x] = 2 * np.pi + phase[x]
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
