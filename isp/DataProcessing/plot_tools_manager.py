import math
import os
import pickle

import numpy as np
from mtspec import mtspec

from isp import CLOCK_PATH
from isp.seismogramInspector.signal_processing_advanced import spectrumelement
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from isp.ant.signal_processing_tools import noise_processing

class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


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
        #ax1.fill_between(freq, jackknife_errors[0][:], jackknife_errors[1][:], facecolor="0.75",
        #                 alpha=0.5, edgecolor="0.5")
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
        self.mpf = MatplotlibFrame(fig, window_title="Amplitude spectrum comparison")

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

    def plot_spectrum_stream(self, stream):
        import matplotlib.pyplot as plt
        from isp.Gui.Frames import MatplotlibFrame
        fig, ax1 = plt.subplots(figsize=(6, 6))
        self.mpf = MatplotlibFrame(fig, window_title="Amplitude spectrum comparison")

        for tr in stream:
            data = tr.data
            delta = tr.stats.delta
            sta = tr.stats.station
            [spec, freq, jackknife_errors] = spectrumelement(data, delta, sta)
            info = "{}.{}.{}".format(tr.stats.network, tr.stats.station, tr.stats.channel)
            ax1.loglog(freq, spec, linewidth=1.0, alpha=0.5, label=info)
            ax1.frequencies = freq
            ax1.spectrum = spec
            ax1.set_ylim(spec.min() / 10.0, spec.max() * 100.0)
            plt.ylabel('Amplitude')
            plt.xlabel('Frequency [Hz]')
            plt.grid(True, which="both", ls="-", color='grey')
            plt.legend()
        self.mpf.show()

    def multiple_shifts_array(self, stack, shifts):
        import matplotlib.pyplot as plt
        from isp.Gui.Frames import MatplotlibFrame

        fig, ax = plt.subplots(2, figsize=(10, 5), sharex= True, layout='constrained')
        self.mpf = MatplotlibFrame(fig, window_title="Stack and Shifted Waveforms")

        for tr in shifts[0]:
            info = "{}.{}.{}".format(tr.stats.network, tr.stats.station, tr.stats.channel)
            ax[1].plot(tr.times(), tr.data, linewidth=0.75, alpha=0.5, label=info)

        ax[0].plot(stack.times(), stack.data, linewidth=1.0, color='black', label="Stack")
        ax[1].set_xlim(np.min(stack.times()), np.max(stack.times()))
        ax[1].set_xlabel("Time")
        ax[0].set_ylabel("Amplitude")
        ax[1].set_ylabel("Amplitude")
        legend1 = ax[0].legend(loc='upper right')
        legend2 = ax[1].legend(loc='upper right')
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

    def plot_fit(self, x, y, type, deg, clocks_station_name, ref, dates, crosscorrelate, skew):

        import matplotlib.pyplot as plt
        from isp.Gui.Frames import MatplotlibFrame
        print(clocks_station_name)
        sta1 = clocks_station_name.split("_")[0]
        sta2 = clocks_station_name.split("_")[1]
        fig, ax1 = plt.subplots(figsize=(6, 6))
        plt.ylabel('Skew [s]')
        plt.xlabel('Jul day')

        # Correction by the reference point
        y = y-y[0]

        self.mpf = MatplotlibFrame(fig, window_title="Fit Plot")
        if type == "Logarithmic":
            x = np.log(x)
            pts = ax1.scatter(x, y, c=crosscorrelate, marker='o', edgecolors='k', s=18, vmin = 0.0, vmax = 1.0)
        else:
            pts = ax1.scatter(x, y, c=crosscorrelate, marker='o', edgecolors='k', s=18, vmin = 0.0, vmax = 1.0)
        fig.colorbar(pts, ax=ax1, orientation='horizontal', fraction=0.05,
                                               extend='both', pad=0.15, label='Normalized Cross Correlation')

        try:
            skew1 = sta1 +" Skew " + str(skew[0])
        except:
            skew1 = "No"

        try:
            skew2 = sta2 + " Skew " + str(skew[1])
        except:
            skew2 = "No"

        ax1.text(0.95, 0.08, skew1, verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes,
                color='black', fontsize=12)

        ax1.text(0.95, 0.01, skew2, verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes,
                color='black', fontsize=12)

        x_old = x
        selector = SelectFromCollection(ax1, pts, )

        def accept(event):
            if event.key == "enter":
                print("Selected points:")
                #print(selector.xys[selector.ind])
                x = selector.xys[selector.ind][:,0]
                y = selector.xys[selector.ind][:,1]
                m, n, R2, p, y_model, model, c, t_critical, resid, chi2_red, std_err,ci, pi, x, y = \
                    noise_processing.statisics_fit(x, y, type, deg)
                if type == "Logarithmic":
                   x = np.log(x)
                ax1.scatter(x, y, color="blue", linewidth=1)
                ax1.plot(x, y_model, color="red", linewidth=1, label=f'Line of Best Fit, R² = {R2:.2f}')
                idx = np.abs(np.array(x) - ref).argmin()
                ax1.scatter(ref, y[idx], c="red", marker='o', edgecolors='k', s=18)
                selector.disconnect()
                ax1.set_title("")
                fig.canvas.draw()

                cc = []
                for value in x:
                    index = np.where(x_old == value)
                    cc.append(crosscorrelate[int(index[0])])
                cc = np.array(cc)
                path = os.path.join(CLOCK_PATH, clocks_station_name)
                p = np.flip(p)
                polynom = {clocks_station_name: p.tolist(), 'Dates': dates, 'Dates_selected': x, 'Drift': y, 'Ref': ref,
                           'R2': R2, 'resid': resid, 'chi2_red': chi2_red, 'std_err': std_err, 'cross_correlation': cc,
                           'skew': skew, 'model': model, 'y_model':y_model}
                print(polynom)
                file_to_store = open(path, "wb")
                pickle.dump(polynom, file_to_store)
                #polynom = {clocks_station_name: p.tolist()}
                #df = pd.DataFrame(polynom)
                #df.to_csv(path)

        fig.canvas.mpl_connect("key_press_event", accept)
        ax1.set_title("Press enter to accept selected points.")

        self.mpf.show()





