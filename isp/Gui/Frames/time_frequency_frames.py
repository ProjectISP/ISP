#import time
import math
from matplotlib.colors import Normalize
from isp.DataProcessing import SeismogramDataAdvanced, ConvolveWaveletScipy
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import InvalidFile
from isp.Gui import pw
from isp.Gui.Frames import BaseFrame, UiTimeFrequencyFrame, FilesView, \
    MatplotlibCanvas, MessageDialog
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.time_frequency_advance_frame import TimeFrequencyAdvance
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, add_save_load, convert_qdatetime_utcdatetime
from isp.Utils import MseedUtil, ObspyUtil
from isp.seismogramInspector.MTspectrogram import MTspectrogram, WignerVille
import matplotlib.pyplot as plt
import numpy as np
#from isp.seismogramInspector.ba_fast import ccwt_ba_fast
#from isp.seismogramInspector.CWT_fast import cwt_fast


@add_save_load()
class TimeFrequencyFrame(BaseFrame, UiTimeFrequencyFrame):

    def __init__(self):

        super(TimeFrequencyFrame, self).__init__()
        self.setupUi(self)
        self.__stations_dir = None
        self.__metadata_manager = None
        self.inventory = {}
        self._stations_info = {}
        self.tr1 = []
        self.tr2 = []
        self.canvas_plot1 = MatplotlibCanvas(self.widget_plot_up, nrows=2)
        self.canvas_plot1.set_xlabel(1, "Time (s)")
        self.canvas_plot1.set_ylabel(0, "Amplitude ")
        self.canvas_plot1.set_ylabel(1, "Frequency (Hz)")
        self.canvas_plot2 = MatplotlibCanvas(self.widget_plot_down, nrows=2)
        self.canvas_plot2.set_xlabel(1, "Time (s)")
        self.canvas_plot2.set_ylabel(0, "Amplitude ")
        self.canvas_plot2.set_ylabel(1, "Frequency (Hz)")
        # Binding
        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        # Add file selector to the widget
        self.file_selector = FilesView(self.root_path_bind.value, parent=self.fileSelectorWidget,
                                       on_change_file_callback=lambda file_path: self.onChange_file(file_path))
        # Binds
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.datalessBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        # Action Buttons
        self.actionSettings.triggered.connect(lambda: self.open_parameters_settings())
        self.actionOpen_Spectral_Analysis.triggered.connect(self.time_frequency_advance)
        self.plotBtn.clicked.connect(self.plot_seismogram)
        # Parameters settings
        self.parameters = ParametersSettings()
        # Time Frequency Advance
        #self.time_frequency_advance = TimeFrequencyAdvance()

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def message_dataless_not_found(self):
        if len(self.dataless_not_found) > 1:
            md = MessageDialog(self)
            md.set_info_message("Metadata not found.")
        else:
            for file in self.dataless_not_found:
                md = MessageDialog(self)
                md.set_info_message("Metadata for {} not found.".format(file))

        self.dataless_not_found.clear()


    def open_parameters_settings(self):
        self.parameters.show()

    def time_frequency_advance(self):
        self._time_frequency_advance = TimeFrequencyAdvance(self.tr1, self.tr2)
        self._time_frequency_advance.show()


    def validate_file(self):
        if not MseedUtil.is_valid_mseed(self.file_selector.file_path):
            msg = "The file {} is not a valid mseed. Please, choose a valid format". \
                format(self.file_selector.file_name)
            raise InvalidFile(msg)


    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        self.file_selector.set_new_rootPath(value)

    def onChange_file(self, file_path):
        # Called every time user select a different file
        pass

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        if dir_path:
            bind.value = dir_path

    def onChange_metadata_path(self, value):
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
        except:
            pass

    @property
    def trace(self):
        return ObspyUtil.get_tracer_from_file(self.file_selector.file_path)

    def get_data(self):
        file = self.file_selector.file_path
        starttime = convert_qdatetime_utcdatetime(self.starttime_date)
        endtime = convert_qdatetime_utcdatetime(self.endtime_date)
        diff = endtime - starttime
        parameters = self.parameters.getParameters()
        sd = SeismogramDataAdvanced(file)

        if self.trimCB.isChecked() and diff >= 0:
            tr = sd.get_waveform_advanced(parameters, self.inventory, filter_error_callback=self.filter_error_message,
                                          start_time=starttime, end_time=endtime)
        else:
            tr = sd.get_waveform_advanced(parameters, self.inventory, filter_error_callback=self.filter_error_message)

        t = tr.times()
        return tr, t

    def get_time_window(self):

        t1 = convert_qdatetime_utcdatetime(self.starttime_date)
        t2 = convert_qdatetime_utcdatetime(self.endtime_date)

        return t1, t2

    def plot_seismogram(self):
        selection = self.selectCB.currentText()

        if selection == "Seismogram 1":
            #self.validate_file()
            [self.tr1, t] = self.get_data()
            self.canvas_plot1.plot(t, self.tr1.data, 0, color="black", linewidth=0.5)
            self.canvas_plot1.set_xlabel(1, "Time (s)")
            self.canvas_plot1.set_ylabel(0, "Amplitude ")
            self.canvas_plot1.set_ylabel(1, "Frequency (Hz)")
            info = "{}.{}.{}".format(self.tr1.stats.network, self.tr1.stats.station, self.tr1.stats.channel)
            self.canvas_plot1.set_plot_label(0, info)

            if self.time_frequencyChB.isChecked():
                self.time_frequency(self.tr1, selection)

        if selection == "Seismogram 2":
            #self.validate_file()
            [self.tr2, t] = self.get_data()
            self.canvas_plot2.plot(t, self.tr2.data, 0, color="black", linewidth=0.5)
            self.canvas_plot2.set_xlabel(1, "Time (s)")
            self.canvas_plot2.set_ylabel(0, "Amplitude ")
            self.canvas_plot2.set_ylabel(1, "Frequency (Hz)")
            info = "{}.{}.{}".format(self.tr2.stats.network, self.tr2.stats.station, self.tr2.stats.channel)
            self.canvas_plot2.set_plot_label(0, info)

            if self.time_frequencyChB.isChecked():
                self.time_frequency(self.tr2, selection)

    def time_frequency(self, tr, order):
        selection = self.time_frequencyCB.currentText()
        ts, te = self.get_time_window()
        diff = te-ts
        if selection == "Multitaper Spectrogram":

            win = int(self.mt_window_lengthDB.value() * tr.stats.sampling_rate)
            tbp = self.time_bandwidth_DB.value()
            ntapers = self.number_tapers_mtSB.value()
            f_min = self.freq_min_mtDB.value()
            f_max = self.freq_max_mtDB.value()
            mtspectrogram = MTspectrogram(self.file_selector.file_path, win, tbp, ntapers, f_min, f_max)

            if self.trimCB.isChecked() and diff >= 0:
                x, y, log_spectrogram = mtspectrogram.compute_spectrogram(tr, start_time=ts, end_time=te)
            else:
                x, y, log_spectrogram = mtspectrogram.compute_spectrogram(tr)

            if order == "Seismogram 1":
                self.canvas_plot1.plot_contour(x, y, log_spectrogram, axes_index=1, clabel="Power [dB]",
                                         cmap=plt.get_cmap("jet"))
                self.canvas_plot1.set_xlabel(1, "Time (s)")
                self.canvas_plot1.set_ylabel(0, "Amplitude ")
                self.canvas_plot1.set_ylabel(1, "Frequency (Hz)")

            elif order == "Seismogram 2":

                self.canvas_plot2.plot_contour(x, y, log_spectrogram, axes_index=1, clabel="Power [dB]",
                                               cmap=plt.get_cmap("jet"))
                self.canvas_plot2.set_xlabel(1, "Time (s)")
                self.canvas_plot2.set_ylabel(0, "Amplitude ")
                self.canvas_plot2.set_ylabel(1, "Frequency (Hz)")

        elif selection == "Wigner Spectrogram":

            win = int(self.mt_window_lengthDB.value() * tr.stats.sampling_rate)
            tbp = self.time_bandwidth_DB.value()
            ntapers = self.number_tapers_mtSB.value()
            f_min = self.freq_min_mtDB.value()
            f_max = self.freq_max_mtDB.value()
            wignerspec = WignerVille(self.file_selector.file_path, win, tbp, ntapers, f_min, f_max)

            if self.trimCB.isChecked() and diff >= 0:
                x, y, log_spectrogram = wignerspec.compute_wigner_spectrogram(tr, start_time=ts, end_time=te)
            else:
                x, y, log_spectrogram = wignerspec.compute_spectrogram(tr)

            if order == "Seismogram 1":
                self.canvas_plot1.plot_contour(x, y, log_spectrogram, axes_index=1, clabel="Rel Power ",
                                         cmap=plt.get_cmap("jet"))
                self.canvas_plot1.set_xlabel(1, "Time (s)")
                self.canvas_plot1.set_ylabel(0, "Amplitude ")
                self.canvas_plot1.set_ylabel(1, "Frequency (Hz)")

            elif order == "Seismogram 2":

                self.canvas_plot2.plot_contour(x, y, log_spectrogram, axes_index=1, clabel="Power [dB]",
                                               cmap=plt.get_cmap("jet"))
                self.canvas_plot2.set_xlabel(1, "Time (s)")
                self.canvas_plot2.set_ylabel(0, "Amplitude ")
                self.canvas_plot2.set_ylabel(1, "Frequency (Hz)")



        elif selection == "Continuous Wavelet Transform":
            #if self.trimCB.isChecked() and diff >= 0:
            #    tr.trim(starttime=ts, endtime=te)

            fs = tr.stats.sampling_rate
            nf = 40
            f_min = self.freq_min_cwtDB.value()
            f_max = self.freq_max_cwtDB.value()
            wmin = self.wminSB.value()
            wmax = self.wminSB.value()
            tt = int( self.wavelet_lenghtDB.value()*fs)
            npts = len(tr.data)
            # Old version
            #[ba, nConv, frex, half_wave] = ccwt_ba_fast(npts, fs, f_min, f_max, wmin, wmax,tt, nf)
            #cf, sc, scalogram = cwt_fast(tr.data, ba, nConv, frex, half_wave, fs)
            #scalogram = np.abs(scalogram) ** 2
            # scalogram2 = 10 * (np.log10(scalogram / np.max(scalogram)))
            #
            t = np.linspace(0, tr.stats.delta * npts, npts)
            cw = ConvolveWaveletScipy(self.file_selector.file_path)
            if self.trimCB.isChecked() and diff >= 0:
                cw.setup_wavelet(ts, te, wmin=wmin, wmax=wmax, tt=tt, fmin=f_min, fmax=f_max, nf=nf, use_rfft=False,
                                 decimate=False)
            else:
                cw.setup_wavelet(wmin=wmin, wmax=wmax, tt=tt, fmin=f_min, fmax=f_max, nf=nf, use_rfft=False,
                                 decimate=False)

            scalogram2 = cw.scalogram_in_dbs()


            scalogram2 = np.around(scalogram2, decimals=2)
            scalogram2 = np.clip(scalogram2, a_min=-120, a_max=0)
            cf = cw.cf_lowpass()
            #start =time.time()
            x, y = np.meshgrid(t, np.linspace(f_min, f_max, scalogram2.shape[0]))
            #print(time.time()-start)
            c_f = wmin / 2 * math.pi
            f = np.linspace(f_min, f_max, scalogram2.shape[0])
            pred = (math.sqrt(2) * c_f / f) * fs - (math.sqrt(2) * c_f / f_max) * fs

            pred_comp = t[len(t)-1]-pred
            min_cwt= -120
            max_cwt = 0

            norm = Normalize(vmin=min_cwt, vmax=max_cwt)

            if order == "Seismogram 1":
                self.canvas_plot1.plot(t[0:len(t) - 1], cf, 0, clear_plot=False, is_twinx=True, color="red",
                                       linewidth=0.5)

                self.canvas_plot1.plot_contour(x, y, scalogram2, axes_index=1, clabel="Power [dB]", cmap=plt.get_cmap("jet"), vmin= min_cwt, vmax=max_cwt)

                #ax_test = self.canvas_plot1.get_axe(1)
                #ax_test.imshow(scalogram2, extent=[0, max(t), min(f), max(f)],cmap = "jet")
                #ax_test.pcolormesh(t, f, scalogram2, cmap=plt.get_cmap("jet"))
                #ax_test.set_xlim(t[0],max(t))
                #ax_test.set_ylim(f[0], max(f))


                ax_cone = self.canvas_plot1.get_axe(1)
                ax_cone.fill_between(pred, f, 0, color= "black", edgecolor="red", alpha=0.3)
                ax_cone.fill_between(pred_comp, f, 0, color="black", edgecolor="red", alpha=0.3)
                self.canvas_plot1.set_xlabel(1, "Time (s)")
                self.canvas_plot1.set_ylabel(0, "Amplitude ")
                self.canvas_plot1.set_ylabel(1, "Frequency (Hz)")

            if order == "Seismogram 2":
                self.canvas_plot2.plot(t[0:len(t) - 1], cf, 0, clear_plot=False, is_twinx=True, color="red",
                                       linewidth=0.5)
                self.canvas_plot2.plot_contour(x, y, scalogram2, axes_index=1, clabel="Power [dB]",
                                           cmap=plt.get_cmap("jet"), norm = norm)

                ax_cone2 = self.canvas_plot2.get_axe(1)
                ax_cone2.fill_between(pred, f, 0, color="black", edgecolor="red", alpha=0.3)
                ax_cone2.fill_between(pred_comp, f, 0, color="black", edgecolor="red", alpha=0.3)
                self.canvas_plot2.set_xlabel(1, "Time (s)")
                self.canvas_plot2.set_ylabel(0, "Amplitude ")
                self.canvas_plot2.set_ylabel(1, "Frequency (Hz)")

        else:
            pass
