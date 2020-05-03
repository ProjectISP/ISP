from isp.Gui import pw
from isp.Gui.Frames import MatplotlibCanvas
from isp.Gui.Frames.uis_frames import UiTimeFrequencyWidget
from isp.seismogramInspector.signal_processing_advanced import spectrumelement, cohe, correlate_template, \
    correlate_maxlag, get_lags
import numpy as np
import math


class TimeFrequencyAdvance(pw.QFrame, UiTimeFrequencyWidget):
    def __init__(self, tr1, tr2):
        super(TimeFrequencyAdvance, self).__init__()
        self.setupUi(self)
        self.spectrum_Widget_Canvas =MatplotlibCanvas(self.spectrumWidget, nrows=2, ncols=2,
                                                      sharex=False, constrained_layout=True)
        self.coherence_Widget_Canvas = MatplotlibCanvas(self.coherenceWidget, nrows=2, ncols=1, sharex=False,
                                                       constrained_layout=True)
        self.cross_correlation_Widget_Canvas = MatplotlibCanvas(self.cross_correlationWidget, nrows = 3, ncols = 1)
        self.plot_spectrumBtn.clicked.connect(self.plot_spectrum)
        self.coherenceBtn.clicked.connect(self.coherence)
        self.cross_correlationsBtn.clicked.connect(self.plot_correlation)
        self.tr1 = tr1
        self.tr2 = tr2

    def plot_spectrum(self):
        if len(self.tr1) >0:
            [spec, freq, jackknife_errors] = spectrumelement(self.tr1.data, self.tr1.stats.delta, self.tr1.id)
            self.spectrum_Widget_Canvas.plot(freq, spec, 0)
            ax1 = self.spectrum_Widget_Canvas.get_axe(0)
            ax1.cla()
            ax1.loglog(freq, spec, '0.1', linewidth=0.5, color='steelblue', label=self.tr1.id)
            ax1.fill_between(freq, jackknife_errors[:, 0], jackknife_errors[:, 1], facecolor="0.75",
                             alpha=0.5, edgecolor="0.5")
            ax1.set_ylim(spec.min() / 10.0, spec.max() * 100.0)
            ax1.set_xlim(freq[1], freq[len(freq)-1])
            ax1.set_ylabel('Amplitude')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.grid(True, which="both", ls="-", color='grey')
            ax1.legend()

            #plot the phase
            spectrum = np.fft.rfft(self.tr1.data)
            phase = np.angle(spectrum)
            ax3 = self.spectrum_Widget_Canvas.get_axe(2)
            ax3.semilogx(freq, phase*180/math.pi, color = 'orangered', linewidth=0.5)
            ax3.set_xlim(freq[1], freq[len(freq) - 1])
            ax3.set_ylabel('Phase')
            ax3.set_xlabel('Frequency [Hz]')

        else:
            pass

        if len(self.tr2) > 0:
            [spec, freq, jackknife_errors] = spectrumelement(self.tr2.data, self.tr2.stats.delta, self.tr2.id)
            self.spectrum_Widget_Canvas.plot(freq, spec, 1)
            ax2 = self.spectrum_Widget_Canvas.get_axe(1)
            ax2.cla()
            ax2.loglog(freq, spec, '0.1', linewidth=0.5, color='steelblue', label=self.tr2.id)
            ax2.fill_between(freq, jackknife_errors[:, 0], jackknife_errors[:, 1], facecolor="0.75",
                             alpha=0.5, edgecolor="0.5")
            ax2.set_ylim(spec.min() / 10.0, spec.max() * 100.0)
            ax2.set_xlim(freq[1], freq[len(freq) - 1])
            ax2.set_ylabel('Amplitude')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.grid(True, which="both", ls="-", color='grey')
            ax2.legend()

            # plot the phase
            spectrum = np.fft.rfft(self.tr2.data)
            phase = np.angle(spectrum)
            ax4 = self.spectrum_Widget_Canvas.get_axe(3)
            ax4.semilogx(freq, phase * 180 / math.pi, color = 'orangered', linewidth=0.5)
            ax4.set_xlim(freq[1], freq[len(freq) - 1])
            ax4.set_ylabel('Phase')
            ax4.set_xlabel('Frequency [Hz]')
        else:
            pass


    def coherence(self):
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

        if len(self.tr1) > 0 and len(self.tr2) > 0:
            sampling_rates = []
            fs1=self.tr1.stats.sampling_rate
            fs2=self.tr2.stats.sampling_rate
            sampling_rates.append(fs1)
            sampling_rates.append(fs2)
            max_sampling_rates = np.max(sampling_rates)

            if fs1 != fs2:
                self.tr1.resample(max_sampling_rates)
                self.tr1.resample(max_sampling_rates)

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


