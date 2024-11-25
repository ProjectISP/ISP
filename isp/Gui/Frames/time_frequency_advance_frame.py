from obspy import Stream
from scipy import ndimage
from isp.Gui import pw
from isp.Gui.Frames import MatplotlibCanvas
from isp.Gui.Frames.uis_frames import UiTimeFrequencyWidget
#from isp.seismogramInspector.MTspectrogram import cross_spectrogram # TODO CHANGE TO NITIME
from isp.seismogramInspector.signal_processing_advanced import spectrumelement, cohe, correlate_template, \
    correlate_maxlag, get_lags
import numpy as np
import math
from isp.seismogramInspector.ba_fast import ccwt_ba_fast
from isp.seismogramInspector.CWT_fast import cwt_fast


class TimeFrequencyAdvance(pw.QFrame, UiTimeFrequencyWidget):
    def __init__(self, tr1, tr2):
        super(TimeFrequencyAdvance, self).__init__()
        self.setupUi(self)
        self.spectrum_Widget_Canvas = MatplotlibCanvas(self.spectrumWidget, nrows=2, ncols=2,
                                                      sharex=False, constrained_layout=True)

        self.spectrum_Widget_Canvas2 = MatplotlibCanvas(self.spectrumWidget2, nrows=1, ncols=1,
                                                      sharex=False, constrained_layout=True)

        self.coherence_Widget_Canvas = MatplotlibCanvas(self.coherenceWidget, nrows=2, ncols=1, sharex=False,
                                                       constrained_layout=True)
        self.cross_correlation_Widget_Canvas = MatplotlibCanvas(self.cross_correlationWidget, nrows = 3, ncols = 1)

        self.cross_spectrumWidget_Widget_Canvas = MatplotlibCanvas(self.cross_spectrumWidget,nrows = 2, ncols = 1,
                                sharex=True, constrained_layout=True)


        self.plot_spectrumBtn.clicked.connect(self.plot_spectrum)
        self.coherenceBtn.clicked.connect(self.coherence)
        self.cross_correlationsBtn.clicked.connect(self.plot_correlation)
        #self.Cross_spectrumBtn.clicked.connect(self.plot_cross_spectrogram)
        self.Cross_scalogramBtn.clicked.connect(self.plot_cross_scalogram)
        self.tr1 = tr1
        self.tr2 = tr2

        self.horizontalSlider.valueChanged.connect(self.valueChanged)

        # Resolution
        # Based on 100 Hz,
        self.very_low_res = 100 * 3600 * 0.5
        self.low_res = 100 * 3600 * 1
        self.high_res = 100 * 3600 * 3
        self.very_high_res = 100 * 3600 * 6

    def valueChanged(self):
        self.resSB.setValue(self.horizontalSlider.value())

    def plot_spectrum(self):
        self.specialWidget.setCurrentIndex(0)
        if len(self.tr1) >0:
            [spec1, freq1, _] = spectrumelement(self.tr1.data, self.tr1.stats.delta, self.tr1.id)
            self.spectrum_Widget_Canvas.plot(freq1, spec1, 0)
            ax1 = self.spectrum_Widget_Canvas.get_axe(0)
            ax1.cla()
            ax1.loglog(freq1, spec1, '0.1', linewidth=0.5, color='steelblue', label=self.tr1.id)
            #ax1.fill_between(freq1, jackknife_errors[:, 0], jackknife_errors[:, 1], facecolor="0.75",
            #                 alpha=0.5, edgecolor="0.5")
            ax1.set_ylim(spec1.min() / 10.0, spec1.max() * 100.0)
            ax1.set_xlim(freq1[1], freq1[len(freq1)-1])
            ax1.set_ylabel('Amplitude')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.grid(True, which="both", ls="-", color='grey')
            ax1.legend()

            #plot the phase
            N = len(self.tr1.data)
            D = 2 ** math.ceil(math.log2(N))
            z = np.zeros(D - N)
            data = np.concatenate((self.tr1.data, z), axis=0)
            spectrum = np.fft.rfft(data, D)
            phase = np.angle(spectrum)
            ax3 = self.spectrum_Widget_Canvas.get_axe(2)
            ax3.semilogx(freq1, phase*180/math.pi, color = 'orangered', linewidth=0.5)
            ax3.set_xlim(freq1[1], freq1[len(freq1) - 1])
            ax3.grid(True, which="both", ls="-", color='grey')
            ax3.set_ylabel('Phase')
            ax3.set_xlabel('Frequency [Hz]')

        else:
            pass

        if len(self.tr2) > 0:
            [spec2, freq2, _] = spectrumelement(self.tr2.data, self.tr2.stats.delta, self.tr2.id)
            self.spectrum_Widget_Canvas.plot(freq2, spec2, 1)
            ax2 = self.spectrum_Widget_Canvas.get_axe(1)
            ax2.cla()
            ax2.loglog(freq2, spec2, '0.1', linewidth=0.5, color='steelblue', label=self.tr2.id)
            #ax2.fill_between(freq2, jackknife_errors[:, 0], jackknife_errors[:, 1], facecolor="0.75",
            #                 alpha=0.5, edgecolor="0.5")
            ax2.set_ylim(spec2.min() / 10.0, spec2.max() * 100.0)
            ax2.set_xlim(freq2[1], freq2[len(freq2) - 1])
            ax2.set_ylabel('Amplitude')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.grid(True, which="both", ls="-", color='grey')
            ax2.legend()

            # plot the phase
            N = len(self.tr2.data)
            D = 2 ** math.ceil(math.log2(N))
            z = np.zeros(D - N)
            data = np.concatenate((self.tr2.data, z), axis=0)
            spectrum = np.fft.rfft(data, D)
            phase = np.angle(spectrum)
            ax4 = self.spectrum_Widget_Canvas.get_axe(3)
            ax4.semilogx(freq2, phase * 180 / math.pi, color = 'orangered', linewidth=0.5)
            ax4.set_xlim(freq2[1], freq2[len(freq2) - 1])
            ax4.grid(True, which="both", ls="-", color='grey')
            ax4.set_ylabel('Phase')
            ax4.set_xlabel('Frequency [Hz]')
        else:
            pass

        # Power

        try:
            self.spectrum_Widget_Canvas2.plot(freq1, spec1, 0)
            ax5 = self.spectrum_Widget_Canvas2.get_axe(0)
            ax5.cla()
            spec1=spec1**2
            spec2=spec2**2
            ax5.loglog(freq1, spec1, '0.1', linewidth=0.5, color='steelblue', label=self.tr1.id)
            ax5.loglog(freq2, spec2, '0.1', linewidth=0.5, color='orangered', label=self.tr2.id)
            ax5.set_ylabel('Amplitude')
            ax5.set_xlabel('Frequency [Hz]')
            ax5.grid(True, which="both", ls="-", color='grey')
            ax5.legend()

        except:
            raise("Some data is missing")




    def coherence(self):
        self.specialWidget.setCurrentIndex(1)
        if len(self.tr1) > 0 and len(self.tr2) > 0:
            sampling_rates = []
            fs1=self.tr1.stats.sampling_rate
            fs2=self.tr2.stats.sampling_rate
            sampling_rates.append(fs1)
            sampling_rates.append(fs2)
            max_sampling_rates = np.max(sampling_rates)
            overlap = self.cohe_overlapSB.value()
            time_window = self.time_window_coheSB.value()
            nfft = time_window*max_sampling_rates
            if fs1 != fs2:
                self.tr1.resample(max_sampling_rates)
                self.tr1.resample(max_sampling_rates)

            # Plot Amplitude
            [A,f, phase] = cohe(self.tr1.data, self.tr2.data, max_sampling_rates, nfft, overlap)

            self.coherence_Widget_Canvas.plot(f, A, 0)

            ax1 = self.coherence_Widget_Canvas.get_axe(0)
            ax1.cla()
            ax1.loglog(f, A, '0.1', linewidth=0.5, color='steelblue', label="Amplitude Coherence "+
                                                                            self.tr1.id+" "+ self.tr2.id)

            ax1.set_ylim(A.min() / 10.0, A.max() * 100.0)
            ax1.set_xlim(f[1], f[len(f) - 1])
            ax1.set_ylabel('Magnitude Coherence')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.grid(True, which="both", ls="-", color='grey')
            ax1.legend()

            # Plot Phase
            ax2 = self.coherence_Widget_Canvas.get_axe(1)
            ax2.cla()
            ax2.semilogx(f, phase * 180 / math.pi, color='orangered', linewidth=0.5, label="Phase Coherence "+
                                                                            self.tr1.id+" "+ self.tr2.id)
            ax2.set_xlim(f[1], f[len(f) - 1])
            ax2.set_ylabel('Phase')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.grid(True, which="both", ls="-", color='grey')
            ax2.legend()

    def plot_correlation(self):
        self.specialWidget.setCurrentIndex(2)
        if len(self.tr1) > 0 and len(self.tr2) > 0:
            sampling_rates = []
            fs1=self.tr1.stats.sampling_rate
            fs2=self.tr2.stats.sampling_rate
            sampling_rates.append(fs1)
            sampling_rates.append(fs2)
            max_sampling_rates = np.max(sampling_rates)

            if fs1 != fs2:
                self.tr1.resample(max_sampling_rates)
                self.tr2.resample(max_sampling_rates)

            cc1 = correlate_maxlag(self.tr1.data, self.tr1.data, maxlag = max([len(self.tr1.data),len(self.tr2.data)]))
            cc2 = correlate_maxlag(self.tr2.data, self.tr2.data, maxlag = max([len(self.tr1.data),len(self.tr2.data)]))
            cc3 = correlate_maxlag(self.tr1.data, self.tr2.data, maxlag = max([len(self.tr1.data),len(self.tr2.data)]))
            N1 = len(cc1)
            N2 = len(cc2)
            N3 = len(cc3)
            self.cross_correlation_Widget_Canvas.plot(get_lags(cc1)/max_sampling_rates, cc1, 0,
                                                      clear_plot=True, linewidth=0.5,color='black')
            self.cross_correlation_Widget_Canvas.plot(get_lags(cc2)/max_sampling_rates, cc2, 1,
                                                      clear_plot=True, linewidth=0.5, color = 'black')
            self.cross_correlation_Widget_Canvas.plot(get_lags(cc3)/max_sampling_rates, cc3, 2,
                                                      clear_plot=True, linewidth=0.5, color = 'red')

    # def plot_cross_spectrogram(self):
    #     if len(self.tr1) > 0 and len(self.tr2) > 0:
    #         csp = cross_spectrogram(self.tr1, self.tr2, win=self.time_windowSB.value(),
    #                                 tbp=self.time_bandwidthSB.value(), ntapers = self.num_tapersSB.value())
    #
    #         [coherence_cross, freq, t] = csp.compute_coherence_crosspectrogram()
    #         f_min = 0
    #         f_max = 0.5*(1/max(freq))
    #         f_max=50
    #         x, y = np.meshgrid(t, np.linspace(f_min, f_max, coherence_cross.shape[0]))
    #
    #         self.cross_spectrumWidget_Widget_Canvas.plot_contour(x, y, coherence_cross, axes_index=0,
    #                  clabel="Coherence", cmap=plt.get_cmap("jet"), vmin=0,vmax=1)
    #         self.cross_spectrumWidget_Widget_Canvas.set_xlabel(0, "Time (s)")
    #         self.cross_spectrumWidget_Widget_Canvas.set_ylabel(0, "Frequency (Hz)")

    def __estimate_res(self, npts):

        res_user = self.resSB.value()
        if npts <= self.very_low_res:
            self.res_factor = 1
            if res_user > self.res_factor:
                self.res_factor = res_user
        elif self.very_low_res < npts <= self.low_res:
            self.res_factor = 20
            if res_user > self.res_factor:
                self.res_factor = res_user
        elif self.high_res < npts <= self.very_high_res:
            self.res_factor = 50
            if res_user > self.res_factor:
                self.res_factor = res_user
        elif npts > self.very_high_res:
            self.res_factor = 100
            if res_user > self.res_factor:
                self.res_factor = res_user
        print("resolution Factor", self.res_factor)

    def plot_cross_scalogram(self):
        self.specialWidget.setCurrentIndex(3)
        base_line = self.base_lineDB.value()
        colour = self.colourCB.currentText()

        if len(self.tr1) > 0 and len(self.tr2) > 0:
            #
            fs1 = self.tr1.stats.sampling_rate
            fs2 = self.tr2.stats.sampling_rate
            fs = max(fs1, fs2)
            if fs1 < fs:
                self.tr1.resample(fs)
            elif fs2 < fs:
                self.tr2.resample(fs)
            all_traces = [self.tr1, self.tr2]
            st = Stream(traces=all_traces)
            maxstart = np.max([tr.stats.starttime for tr in st])
            minend = np.min([tr.stats.endtime for tr in st])
            st.trim(maxstart, minend)
            tr1 = st[0]
            tr2 = st[1]
            tr1.detrend(type='demean')
            tr1.taper(max_percentage=0.05)
            tr2.detrend(type='demean')
            tr2.taper(max_percentage=0.05)

            npts =len(tr1.data)
            self.__estimate_res( npts)
            t = np.linspace(0,  npts/fs, npts)
            f_max = fs/2
            nf = 40
            wmin=wmax=self.num_cyclesSB.value()
            f_min = self.freq_min_waveletSB.value()
            tt = int(fs / f_min)
            [ba, nConv, frex, half_wave] = ccwt_ba_fast(npts, fs, f_min, f_max, wmin, wmax, tt, nf)

            cf, sc, scalogram1 = cwt_fast(tr1.data, ba, nConv, frex, half_wave, fs)

            cf, sc, scalogram2 = cwt_fast(tr2.data, ba, nConv, frex, half_wave, fs)

            crossCFS = scalogram1 * np.conj(scalogram2)

            cross_scalogram = np.abs(crossCFS)**2
            cross_scalogram = 10*np.log(cross_scalogram/np.max(cross_scalogram))
            cross_scalogram = np.clip(cross_scalogram, a_min=base_line, a_max=0)

            if self.res_factor > 1:

                cross_scalogram = ndimage.zoom(cross_scalogram, (1.0, 1/self.res_factor))
                #tt = t[::factor]
                t = np.linspace(0, self.res_factor * (1/fs) * cross_scalogram.shape[1], cross_scalogram.shape[1])
                freq = np.logspace(np.log10(f_min), np.log10(f_max), cross_scalogram.shape[0])
                x, y = np.meshgrid(t, freq)
            else:
                freq = np.logspace(np.log10(f_min), np.log10(f_max), cross_scalogram.shape[0])
                x, y = np.meshgrid(t, freq)


            c_f = wmin / 2 * math.pi
            f = np.linspace((f_min), (f_max), cross_scalogram.shape[0])
            pred = (math.sqrt(2) * c_f / f) - (math.sqrt(2) * c_f / f_max)

            pred_comp = t[len(t) - 1] - pred


            #Plot Seismograms

            self.cross_spectrumWidget_Widget_Canvas.plot(tr1.times(), self.tr1.data, 0, color="black",
                    linewidth=0.5,alpha = 0.75)

            self.cross_spectrumWidget_Widget_Canvas.plot(tr2.times(), tr2.data, 0, clear_plot=False, is_twinx=True,
                                                         color="orangered", linewidth=0.5, alpha=0.75)
            info = tr1.id + " " + tr2.id
            self.cross_spectrumWidget_Widget_Canvas.set_plot_label(0, info)
            self.cross_spectrumWidget_Widget_Canvas.set_ylabel(0, "Amplitude")
            #info = "{}.{}.{}".format(tr1.stats.network, tr1.stats.station, tr1.stats.channel)
            #self.cross_spectrumWidget_Widget_Canvas.set_plot_label(0, info)
            #info = "{}.{}.{}".format(tr2.stats.network, tr2.stats.station, tr1.stats.channel)
            #self.cross_spectrumWidget_Widget_Canvas.set_plot_label(0, info)


            # Plot Cross Scalogram

            if self.res_factor <= 1:
                self.cross_spectrumWidget_Widget_Canvas.plot_contour(x, y, cross_scalogram, axes_index=1, clear_plot=True,
                                                                     clabel="Cross Power [dB]", cmap=self.colourCB.currentText())
            else:
                self.cross_spectrumWidget_Widget_Canvas.pcolormesh(x, y, cross_scalogram, axes_index=1, clear_plot=True,
                                                                     clabel="Cross Power [dB]", cmap=self.colourCB.currentText())
            # elif self.typeCB.currentText() == 'imshow':
            #     self.cross_spectrumWidget_Widget_Canvas.image(x, y, cross_scalogram, axes_index=1, clear_plot=True,
            #                                                          clabel="Cross Power [dB]", cmap=self.colourCB.currentText())

            # Plot Cone
            ax_cone = self.cross_spectrumWidget_Widget_Canvas.get_axe(1)
            ax_cone.fill_between(pred, f, 0, color="black", edgecolor="red", alpha=0.3)
            ax_cone.fill_between(pred_comp, f, 0, color="black", edgecolor="red", alpha=0.3)
            self.cross_spectrumWidget_Widget_Canvas.set_xlabel(1, "Time (s)")
            self.cross_spectrumWidget_Widget_Canvas.set_ylabel(1, "Frequency (Hz)")