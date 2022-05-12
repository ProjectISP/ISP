import math
import numpy as np
from mtspec import mtspec
from isp.seismogramInspector.signal_processing_advanced import spectrumelement


class PlotToolsManager:

    def __init__(self, id):
        """
        Manage Plot signal analysis in Earthquake Frame.

        :param obs_file_path: The file path of pick observations.
        """
        self.__id = id


    def plot_spectrum(self, freq, spec, jackknife_errors):
        import matplotlib.pyplot as plt
        from isp.Gui.Frames import MatplotlibFrame
        fig, ax1 = plt.subplots(figsize=(6, 6))
        self.mpf = MatplotlibFrame(fig, window_title="Amplitude spectrum")
        ax1.loglog(freq, spec, linewidth=1.0, color='steelblue', label=self.__id)
        ax1.frequencies = freq
        ax1.spectrum = spec
        ax1.fill_between(freq, jackknife_errors[:, 0], jackknife_errors[:, 1], facecolor="0.75",
                         alpha=0.5, edgecolor="0.5")
        ax1.set_ylim(spec.min() / 10.0, spec.max() * 100.0)
        plt.ylabel('Amplitude')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, which="both", ls="-", color='grey')
        plt.legend()
        self.mpf.show()

    def plot_spectrum_all(self, all_items):
        import matplotlib.pyplot as plt
        from isp.Gui.Frames import MatplotlibFrame
        fig, ax1 = plt.subplots(figsize=(6, 6))
        self.mpf = MatplotlibFrame(fig)

        for key, seismogram in all_items:
            data = seismogram[2]
            delta = 1 / seismogram[0][6]
            sta = seismogram[0][1]
            [spec, freq, jackknife_errors] = spectrumelement(data, delta, sta)
            info = "{}.{}.{}".format(seismogram[0][0], seismogram[0][1], seismogram[0][3])
            ax1.loglog(freq, spec, linewidth=1.0, alpha = 0.5, label=info)
            ax1.frequencies = freq
            ax1.spectrum = spec
            ax1.set_ylim(spec.min() / 10.0, spec.max() * 100.0)
            # ax1.set_xlim(freq[0], 1/(2*delta))
            plt.ylabel('Amplitude')
            plt.xlabel('Frequency [Hz]')
            plt.grid(True, which="both", ls="-", color='grey')
            plt.legend()
        self.mpf.show()





    def find_nearest(self,array, value):
        idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
        return idx, val

    # def __compute_spectrogram(self, tr):
    #      npts = len(tr)
    #      t = np.linspace(0, (tr.stats.delta * npts), npts - self.win)
    #      mt_spectrum = self.MTspectrum(tr.data, self.win, tr.stats.delta, self.tbp, self.ntapers, self.f_min, self.f_max)
    #      log_spectrogram = 10. * np.log(mt_spectrum / np.max(mt_spectrum))
    #      x, y = np.meshgrid(t, np.linspace(self.f_min, self.f_max, log_spectrogram.shape[0]))
    #      return x, y, log_spectrogram

    def MTspectrum_plot(self, data, win, dt, tbp, ntapers, linf, lsup):

        if (win % 2) == 0:
            nfft = win / 2 + 1
        else:
            nfft = (win + 1) / 2

        lim = len(data) - win
        S = np.zeros([int(nfft), int(lim)])
        data2 = np.zeros(2 ** math.ceil(math.log2(win)))

        for n in range(lim):
            data1 = data[n:win + n]
            data1 = data1 - np.mean(data1)
            data2[0:win] = data1
            spec, freq = mtspec(data2, delta=dt, time_bandwidth=tbp, number_of_tapers=ntapers)
            spec = spec[0:int(nfft)]
            S[:, n] = spec

        value1, freq1 = self.find_nearest(freq, linf)
        value2, freq2 = self.find_nearest(freq, lsup)
        S = S[value1:value2]

        return S

    def compute_spectrogram_plot(self, data, win, delta, tbp, ntapers, f_min, f_max, t):

      npts = len(data)
      x = np.linspace(0, (delta * npts), npts - win)
      t = t[0:len(x)]
      mt_spectrum = self.MTspectrum_plot(data, win, delta, tbp, ntapers, f_min, f_max)
      log_spectrogram = 10. * np.log(mt_spectrum / np.max(mt_spectrum))
      x, y = np.meshgrid(t, np.linspace(f_min, f_max, log_spectrogram.shape[0]))
      return x, y, log_spectrogram
