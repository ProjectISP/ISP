import math
from PyQt5.QtCore import Qt
from obspy import UTCDateTime
from scipy import ndimage
import pandas as pd
from isp import ROOT_DIR
from isp.DataProcessing import SeismogramDataAdvanced, ConvolveWaveletScipy
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import InvalidFile
from isp.Gui import pw, pyc
from isp.Gui.Frames import BaseFrame, UiTimeFrequencyFrame, FilesView, \
    MatplotlibCanvas, MessageDialog
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.time_frequency_advance_frame import TimeFrequencyAdvance
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, add_save_load, convert_qdatetime_utcdatetime, set_qdatetime
from isp.Utils import MseedUtil, ObspyUtil, AsycTime
from isp.Utils.subprocess_utils import open_url
from isp.seismogramInspector.MTspectrogram import MTspectrogram
import numpy as np
import os
from sys import platform
import matplotlib.dates as mdt
import matplotlib.dates as mdates

@add_save_load()
class TimeFrequencyFrame(BaseFrame, UiTimeFrequencyFrame):

    def __init__(self):

        super(TimeFrequencyFrame, self).__init__()
        self.setupUi(self)
        self.__stations_dir = None
        self.__metadata_manager = None
        self.url="https://projectisp.github.io/ISP_tutorial.github.io/tf/"
        # picks
        self.picks = {'ID': [], 'Time': [], 'Amplitude': [], 'Phase': []}
        self.inventory = {}
        self._stations_info = {}
        self.tr1 = []
        self.tr2 = []
        self.tr3 = []
        self.canvas_plot1 = MatplotlibCanvas(self.widget_plot_up, sharex=True, constrained_layout=True, nrows=2)
        self.canvas_plot2 = MatplotlibCanvas(self.widget_plot_down, sharex=True, constrained_layout=True, nrows=2)
        self.canvas_plot3 = MatplotlibCanvas(self.widget_plot_3, sharex=True, constrained_layout=True, nrows=3)
        #self.canvas_plot3.figure.subplots_adjust(left=0.046, bottom=0.070, right=0.976, top=0.975, wspace=0.2,
        #                                         hspace=0.0)
        self.canvas_plot3.set_xlabel(2, "Time (s)")

        # Binding
        self.canvas_plot1.mpl_connect('key_press_event', self.key_pressed)
        self.canvas_plot2.mpl_connect('key_press_event', self.key_pressed)
        self.canvas_plot3.mpl_connect('key_press_event', self.key_pressed)
        self.canvas_plot1.on_double_click(self.on_click_matplotlib)
        self.canvas_plot2.on_double_click(self.on_click_matplotlib)
        self.canvas_plot3.on_double_click(self.on_click_matplotlib)

        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        # Add file selector to the widget
        self.file_selector = FilesView(self.root_path_bind.value, parent=self.fileSelectorWidget,
                                       on_change_file_callback=lambda file_path: self.onChange_file(file_path))
        # Binds
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.datalessBtn.clicked.connect(lambda: self.on_click_select_file(self.dataless_path_bind))
        # Action Buttons
        self.actionOpen_Help.triggered.connect(lambda: self.open_help())
        self.macroBtn.clicked.connect(lambda: self.open_parameters_settings())
        self.actionOpen_Spectral_Analysis.triggered.connect(self.time_frequency_advance)
        self.advanceBtn.clicked.connect(self.time_frequency_advance)
        self.macroBtn.clicked.connect(lambda: self.open_parameters_settings())
        self.plotBtn.clicked.connect(self.plot_seismogram)
        self.stationsBtn.clicked.connect(self.stations_info)


        # Parameters settings
        self.parameters = ParametersSettings()

        # Time Frequency Advance
        #self.time_frequency_advance = TimeFrequencyAdvance()
        self.horizontalSlider.valueChanged.connect(self.valueChanged)
        self.activate_xaxis = True

        #self.radioBtnTime.clicked.connect(lambda self: self.activate_xaxis=True)
        self.radioBtnTime.clicked.connect(lambda: setattr(self, 'activate_xaxis', True))
        self.radioBtnDate.clicked.connect(lambda: setattr(self, 'activate_xaxis', False))

        self.time_frequencyCB.currentTextChanged.connect(self.hilight_params)
        # Resolution
        # Based on 100 Hz,
        self.very_low_res = 100*3600*0.5
        self.low_res = 100*3600 * 1
        self.high_res = 100 * 3600 * 3
        self.very_high_res = 100 * 3600 * 6

        # Dialog
        self.progress_dialog = pw.QProgressDialog(self)
        self.progress_dialog.setRange(0, 0)
        self.progress_dialog.setWindowTitle('Processing.....')
        self.progress_dialog.setLabelText('Please Wait')
        self.progress_dialog.setWindowIcon(self.windowIcon())
        self.progress_dialog.setWindowTitle(self.windowTitle())
        self.progress_dialog.close()

    def hilight_params(self):
        selection = self.time_frequencyCB.currentText()
        if selection == "Multitaper Spectrogram":
            self.tabWidget.setCurrentIndex(0)
        elif selection == "Continuous Wavelet Transform":
            self.tabWidget.setCurrentIndex(1)

    def valueChanged(self):
        self.resSB.setValue(self.horizontalSlider.value())

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

    def find_nearest(self, a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return a.flat[idx], idx

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
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path

    def on_click_select_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]


    def onChange_metadata_path(self, value):
        md = MessageDialog(self)
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
            md.set_info_message("Loaded Metadata, please check your terminal for further details")
        except:
            md.set_error_message("Something went wrong. Please check your metada file is a correct one")

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

    def stations_info(self):
        obsfiles = MseedUtil.get_mseed_files(self.root_path_bind.value)
        obsfiles.sort()
        sd = []
        for file in obsfiles:
            st = SeismogramDataAdvanced(file)
            station = [st.stats.Network, st.stats.Station, st.stats.Location, st.stats.Channel, st.stats.StartTime,
                       st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]
            sd.append(station)
        self._stations_info = StationsInfo(sd, check= False)
        self._stations_info.show()

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

        elif self.low_res < npts <= self.high_res:
            self.res_factor = 40
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

    def plot_seismogram(self):
        selection = self.selectCB.currentText()

        if selection == "Seismogram 1":
            #self.validate_file()
            self.tabWidget_TF.setCurrentIndex(0)
            self.canvas_plot1.clear()
            [self.tr1, t] = self.get_data()
            if not self.activate_xaxis:
                self.canvas_plot1.plot_date(self.tr1.times("matplotlib"), self.tr1.data, 0, fmt='-', clear_plot=True, color="black", linewidth=0.5)
                self.canvas_plot1.set_xlabel(1, "Date")
            else:
                self.canvas_plot1.plot(t, self.tr1.data, 0, clear_plot=True, color="black", linewidth=0.5)
                self.canvas_plot1.set_xlabel(1, "Time (s)")
            self.canvas_plot1.set_ylabel(0, "Amplitude ")
            self.canvas_plot1.set_ylabel(1, "Frequency (Hz)")
            info = "{}.{}.{}".format(self.tr1.stats.network, self.tr1.stats.station, self.tr1.stats.channel)
            self.canvas_plot1.set_plot_label(0, info)
            self.__estimate_res(self.tr1.stats.npts)
            if not self.activate_xaxis:
                formatter = mdt.DateFormatter('%H:%M:%S')
                ax1 = self.canvas_plot1.get_axe(0)
                ax1.xaxis.set_major_formatter(formatter)
                ax2 = self.canvas_plot1.get_axe(1)
                ax2.xaxis.set_major_formatter(formatter)
            if self.time_frequencyChB.isChecked():
                self.time_frequency(self.tr1, selection)

        if selection == "Seismogram 2":
            #self.validate_file()
            self.tabWidget_TF.setCurrentIndex(0)
            self.canvas_plot2.clear()
            [self.tr2, t] = self.get_data()
            if not self.activate_xaxis:
                self.canvas_plot2.plot_date(self.tr2.times("matplotlib"), self.tr2.data, 0, fmt='-', clear_plot=True,
                                            color="black", linewidth=0.5)
                self.canvas_plot2.set_xlabel(1, "Date")
            else:
                self.canvas_plot2.plot(t, self.tr2.data, 0, clear_plot=True, color="black", linewidth=0.5)
                self.canvas_plot2.set_xlabel(1, "Time (s)")
            self.canvas_plot2.set_xlabel(1, "Time (s)")
            self.canvas_plot2.set_ylabel(0, "Amplitude ")
            self.canvas_plot2.set_ylabel(1, "Frequency (Hz)")
            info = "{}.{}.{}".format(self.tr2.stats.network, self.tr2.stats.station, self.tr2.stats.channel)
            self.canvas_plot2.set_plot_label(0, info)
            self.__estimate_res(self.tr2.stats.npts)
            if not self.activate_xaxis:
                formatter = mdt.DateFormatter('%H:%M:%S')
                ax1 = self.canvas_plot2.get_axe(0)
                ax1.xaxis.set_major_formatter(formatter)
                ax2 = self.canvas_plot2.get_axe(1)
                ax2.xaxis.set_major_formatter(formatter)
            if self.time_frequencyChB.isChecked():
                self.time_frequency(self.tr2, selection)


        if selection == "Seismogram 3":

            self.tabWidget_TF.setCurrentIndex(1)
            self.canvas_plot3.clear()
            [self.tr3, t] = self.get_data()
            if not self.activate_xaxis:
                self.canvas_plot3.plot_date(self.tr3.times("matplotlib"), self.tr3.data, 0, fmt='-', clear_plot=True,
                                            color="black", linewidth=0.5)
                self.canvas_plot3.set_xlabel(1, "Date")
            else:
                self.canvas_plot3.plot(t, self.tr3.data, 0, clear_plot=True, color="black", linewidth=0.5)
                self.canvas_plot3.set_xlabel(1, "Time (s)")
            self.canvas_plot3.set_ylabel(0, "Amplitude ")
            self.canvas_plot3.set_ylabel(1, "Frequency (Hz)")
            self.canvas_plot3.set_ylabel(2, "Period (s)")
            info = "{}.{}.{}".format(self.tr3.stats.network, self.tr3.stats.station, self.tr3.stats.channel)
            self.canvas_plot3.set_plot_label(0, info)
            self.__estimate_res(self.tr3.stats.npts)
            if not self.activate_xaxis:
                formatter = mdt.DateFormatter('%H:%M:%S')
                ax1 = self.canvas_plot3.get_axe(0)
                ax1.xaxis.set_major_formatter(formatter)
                ax2 = self.canvas_plot3.get_axe(1)
                ax2.xaxis.set_major_formatter(formatter)
            if self.time_frequencyChB.isChecked():
                self.time_frequency_full(self.tr3, selection)

    def process_import_trace(self, tr, phases=None, travel_times=None):
        self.tabWidget_TF.setCurrentIndex(1)
        self.canvas_plot3.clear()
        self.tr3 = tr
        t = self.tr3.times()
        fn = int(tr.stats.sampling_rate/2)
        duration = (len(tr.data)*tr.stats.delta)
        minf_freq = np.min([0.01, 4/duration])
        print("trace imported")
        print(self.tr3)
        set_qdatetime(tr.stats.starttime, self.starttime_date)
        set_qdatetime(tr.stats.endtime, self.endtime_date)
        self.trimCB.setChecked(True)
        self.time_frequencyCB.setCurrentIndex(1)
        self.time_frequencyChB.setChecked(True)
        self.freq_max_cwtDB.setValue(fn)
        self.freq_min_cwtDB.setValue(minf_freq)
        self.atomsSB.setValue(40)
        self.wminSB.setValue(6)
        self.wmaxSB.setValue(6)
        self.canvas_plot3.plot(t, self.tr3.data, 0, clear_plot=True, color="black", linewidth=0.5)
        # plot arrivals
        if phases!=None and travel_times!=None:
            for phase, time in zip(phases, travel_times):
                self.canvas_plot3.draw_arrow(time, axe_index=0, arrow_label=phase, draw_arrow=False, color="green")
        self.canvas_plot3.set_xlabel(2, "Time (s)")
        self.canvas_plot3.set_ylabel(0, "Amplitude ")
        self.canvas_plot3.set_ylabel(1, "Frequency (Hz)")
        self.canvas_plot3.set_ylabel(2, "Period (s)")
        info = "{}.{}.{}".format(self.tr3.stats.network, self.tr3.stats.station, self.tr3.stats.channel)
        self.canvas_plot3.set_plot_label(0, info)
        self.__estimate_res(self.tr3.stats.npts)
        if self.time_frequencyChB.isChecked():
             selection = self.selectCB.currentText()
             self.time_frequency_full(self.tr3, selection)

    def time_frequency(self, tr, order):
        self._time_frequency(tr, order)
        self.progress_dialog.exec()


    @AsycTime.run_async()
    def _time_frequency(self, tr, order):
        selection = self.time_frequencyCB.currentText()
        ts, te = self.get_time_window()
        diff = te-ts

        if self.activate_xaxis:
            dates=False
        else:
            dates=True

        if selection == "Multitaper Spectrogram":
            self.res_factor = 1
            win = int(self.mt_window_lengthDB.value() * tr.stats.sampling_rate)
            overlap= 1-(self.overlapSB.value())/100
            f_min = self.freq_min_mtDB.value()
            f_max = self.freq_max_mtDB.value()
            mtspectrogram = MTspectrogram(self.file_selector.file_path, win, f_min, f_max, overlap)

            if self.trimCB.isChecked() and diff >= 0:
                x, y, log_spectrogram = mtspectrogram.compute_spectrogram(tr, start_time=ts, end_time=te,
                                                                          res = self.res_factor, dates=dates)
            else:
                x, y, log_spectrogram = mtspectrogram.compute_spectrogram(tr, res=self.res_factor)

            log_spectrogram = np.clip(log_spectrogram, a_min=self.minlevelCB.value(), a_max=0)
            min_log_spectrogram = self.minlevelCB.value()
            max_log_spectrogram = 0

            if order == "Seismogram 1":

                if self.res_factor <= 1:
                    self.canvas_plot1.plot_contour(x, y, log_spectrogram, axes_index=1, clabel="Power [dB]",
                                         cmap=self.colourCB.currentText(), vmin= min_log_spectrogram,
                                                   vmax=max_log_spectrogram)

                elif self.res_factor > 1:
                    self.canvas_plot1.pcolormesh(x, y, log_spectrogram, axes_index=1, clabel="Power [dB]",
                            cmap=self.colourCB.currentText(), vmin=min_log_spectrogram, vmax=max_log_spectrogram,
                                                 shading='nearest')

                #elif self.typeCB.currentText() == 'imshow':
                #    self.canvas_plot1.image(x, y, log_spectrogram, axes_index=1, clear_plot=True, clabel="Power [dB]",
                #                         cmap=self.colourCB.currentText())
                if not self.activate_xaxis:
                    formatter = mdt.DateFormatter('%H:%M:%S')
                    ax2 = self.canvas_plot1.get_axe(1)
                    ax2.xaxis.set_major_formatter(formatter)
                    self.canvas_plot1.set_xlabel(1, "Date")
                else:
                    self.canvas_plot1.set_xlabel(1, "Time (s)")
                self.canvas_plot1.set_ylabel(0, "Amplitude ")
                self.canvas_plot1.set_ylabel(1, "Frequency (Hz)")

            elif order == "Seismogram 2":
                if self.res_factor <= 1:

                    self.canvas_plot2.plot_contour(x, y, log_spectrogram, axes_index=1, clear_plot=True, clabel="Power [dB]",
                                               cmap=self.colourCB.currentText(), vmin=min_log_spectrogram, vmax=max_log_spectrogram)
                elif self.res_factor > 1:

                    self.canvas_plot2.pcolormesh(x, y, log_spectrogram, axes_index=1, clear_plot=True, clabel="Power [dB]",
                            cmap=self.colourCB.currentText(), vmin=min_log_spectrogram, vmax=max_log_spectrogram,
                                                 shading='nearest')

                if not self.activate_xaxis:
                    formatter = mdt.DateFormatter('%H:%M:%S')
                    ax2 = self.canvas_plot2.get_axe(1)
                    ax2.xaxis.set_major_formatter(formatter)
                    self.canvas_plot2.set_xlabel(1, "Date")
                else:
                    self.canvas_plot2.set_xlabel(1, "Time (s)")

                self.canvas_plot2.set_ylabel(0, "Amplitude ")
                self.canvas_plot2.set_ylabel(1, "Frequency (Hz)")

        elif selection == "Continuous Wavelet Transform":

            fs = tr.stats.sampling_rate
            nf = self.atomsSB.value()
            f_min = self.freq_min_cwtDB.value()
            f_max = self.freq_max_cwtDB.value()
            wmin = self.wminSB.value()
            wmax = self.wminSB.value()
            #tt = int( self.wavelet_lenghtDB.value()*fs)
            npts = len(tr.data)
            t = np.linspace(0, tr.stats.delta * npts, npts)
            #cw = ConvolveWaveletScipy(self.file_selector.file_path)
            cw = ConvolveWaveletScipy(tr)
            wavelet = self.wavelet_typeCB.currentText()

            m = self.wavelets_param.value()
            if self.trimCB.isChecked() and diff >= 0:

                cw.setup_wavelet(ts, te, wmin=wmin, wmax=wmax, tt=int(fs/f_min), fmin=f_min, fmax=f_max, nf=nf,
                                 use_wavelet = wavelet, m = m, decimate=False)
            else:
                cw.setup_wavelet(wmin=wmin, wmax=wmax, tt=int(fs/f_min), fmin=f_min, fmax=f_max, nf=nf,
                                 use_wavelet = wavelet, m = m, decimate=False)

            scalogram2 = cw.scalogram_in_dbs()
            scalogram2 = np.clip(scalogram2, a_min=self.minlevelCB.value(), a_max=0)
            #cf = cw.cf_lowpass()

            #freq = np.linspace(f_min, f_max, scalogram2.shape[0])


            if self.res_factor > 1:

                scalogram2 = ndimage.zoom(scalogram2, (1.0, 1/self.res_factor))
                #tt = t[::factor]
                tt = np.linspace(0, self.res_factor * tr.stats.delta * scalogram2.shape[1], scalogram2.shape[1])
                freq = np.logspace(np.log10(f_min), np.log10(f_max), scalogram2.shape[0])

                if dates:
                    x_datetimes = [tr.stats.starttime + s for s in tt]
                    x_datetimes = [dt.datetime for dt in x_datetimes]
                    x, y = np.meshgrid(mdates.date2num(x_datetimes), freq)
                else:

                    x, y = np.meshgrid(tt, freq)
            else:

                freq = np.logspace(np.log10(f_min), np.log10(f_max), scalogram2.shape[0])

                if dates:
                    x_datetimes = [tr.stats.starttime + s for s in t]
                    x_datetimes = [dt.datetime for dt in x_datetimes]
                    x, y = np.meshgrid(mdates.date2num(x_datetimes), freq)
                else:

                    x, y = np.meshgrid(t, freq)

            k = wmin / (2 * np.pi * freq)
            # delay = int(fs*np.mean(k))
            c_f = wmin / 2 * math.pi
            f = np.linspace(f_min, f_max, scalogram2.shape[0])
            pred = (math.sqrt(2) * c_f / f) - (math.sqrt(2) * c_f / f_max)

            pred_comp = t[len(t)-1]-pred
            min_cwt=self.minlevelCB.value()
            max_cwt = 0

            #norm = Normalize(vmin=min_cwt, vmax=max_cwt)

            #tf=t[delay:len(t)]
            #cf = cf[0:len(tf)]
            if order == "Seismogram 1":

                if self.res_factor <= 1:
                     self.canvas_plot1.plot_contour(x, y, scalogram2, axes_index=1, clear_plot=True, clabel="Power [dB]",
                                                    cmap=self.colourCB.currentText(), vmin= min_cwt, vmax=max_cwt)
                elif self.res_factor > 1:
                     self.canvas_plot1.pcolormesh(x, y, scalogram2, axes_index=1, clear_plot=True, clabel="Power [dB]",
                                                  cmap=self.colourCB.currentText(), vmin=min_cwt, vmax=max_cwt,
                                                  shading='nearest')


                ax_cone = self.canvas_plot1.get_axe(1)
                ax_cone.fill_between(pred, f, 0, color= "black", edgecolor="red", alpha=0.3)
                ax_cone.fill_between(pred_comp, f, 0, color="black", edgecolor="red", alpha=0.3)
                if not self.activate_xaxis:
                    formatter = mdt.DateFormatter('%H:%M:%S')
                    ax2 = self.canvas_plot1.get_axe(1)
                    ax2.xaxis.set_major_formatter(formatter)
                    self.canvas_plot1.set_xlabel(1, "Date")
                else:
                    self.canvas_plot1.set_xlabel(1, "Time (s)")
                self.canvas_plot1.set_ylabel(0, "Amplitude ")
                self.canvas_plot1.set_ylabel(1, "Frequency (Hz)")

            if order == "Seismogram 2":

                if self.res_factor <= 1:
                     self.canvas_plot2.plot_contour(x, y, scalogram2, axes_index=1, clear_plot=True, clabel="Power [dB]",
                                                    cmap=self.colourCB.currentText(), vmin= min_cwt, vmax=max_cwt)
                elif self.res_factor > 1:
                     self.canvas_plot2.pcolormesh(x, y, scalogram2, axes_index=1, clear_plot=True, clabel="Power [dB]",
                                                  cmap=self.colourCB.currentText(), vmin= min_cwt, vmax=max_cwt,
                                                  shading='nearest')
                # elif self.typeCB.currentText() == 'imshow':
                #     self.canvas_plot2.image(x, y, scalogram2, axes_index=1, clear_plot=True, clabel="Power [dB]",
                #                              cmap=self.colourCB.currentText(), vmin=min_cwt, vmax=max_cwt, yscale = 'log')
                ax_cone2 = self.canvas_plot2.get_axe(1)
                ax_cone2.fill_between(pred, f, 0, color="black", edgecolor="red", alpha=0.3)
                ax_cone2.fill_between(pred_comp, f, 0, color="black", edgecolor="red", alpha=0.3)
                if not self.activate_xaxis:
                    formatter = mdt.DateFormatter('%H:%M:%S')
                    ax2 = self.canvas_plot2.get_axe(1)
                    ax2.xaxis.set_major_formatter(formatter)
                    self.canvas_plot2.set_xlabel(1, "Date")
                else:
                    self.canvas_plot2.set_xlabel(1, "Time (s)")
                self.canvas_plot2.set_ylabel(0, "Amplitude ")
                self.canvas_plot2.set_ylabel(1, "Frequency (Hz)")

        else:
            pass

        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)


    def time_frequency_full(self, tr, order):
        self._time_frequency_full(tr, order)
        self.progress_dialog.exec()

    @AsycTime.run_async()
    def _time_frequency_full(self, tr, order):
        selection = self.time_frequencyCB.currentText()
        ts, te = self.get_time_window()
        diff = te - ts
        if self.activate_xaxis:
            dates=False
        else:
            dates=True
        if selection == "Continuous Wavelet Transform":

            fs = tr.stats.sampling_rate
            nf = self.atomsSB.value()
            f_min = self.freq_min_cwtDB.value()
            f_max = self.freq_max_cwtDB.value()
            if f_max < 1:
                f_max = 1
            wmin = self.wminSB.value()
            wmax = self.wminSB.value()
            npts = len(tr.data)
            t = np.linspace(0, tr.stats.delta * npts, npts)
            cw = ConvolveWaveletScipy(tr)
            wavelet = self.wavelet_typeCB.currentText()

            m = self.wavelets_param.value()
            if self.trimCB.isChecked() and diff >= 0:

                cw.setup_wavelet(ts, te, wmin=wmin, wmax=wmax, tt=int(fs / f_min), fmin=f_min, fmax=f_max, nf=nf,
                                 use_wavelet=wavelet, m=m, decimate=False)
            else:
                cw.setup_wavelet(wmin=wmin, wmax=wmax, tt=int(fs / f_min), fmin=f_min, fmax=f_max, nf=nf,
                                 use_wavelet=wavelet, m=m, decimate=False)

            scalogram2 = cw.scalogram_in_dbs()
            scalogram2 = np.clip(scalogram2, a_min=self.minlevelCB.value(), a_max=0)

            if self.res_factor > 1:

                scalogram2 = ndimage.zoom(scalogram2, (1.0, 1 / self.res_factor))
                tt = np.linspace(0, self.res_factor * tr.stats.delta * scalogram2.shape[1], scalogram2.shape[1])
                freq = np.logspace(np.log10(f_min), np.log10(f_max), scalogram2.shape[0])
                if dates:
                    x_datetimes = [tr.stats.starttime + s for s in tt]
                    x_datetimes = [dt.datetime for dt in x_datetimes]
                    x, y = np.meshgrid(mdates.date2num(x_datetimes), freq)
                else:
                    x, y = np.meshgrid(tt, freq)
            else:
                freq = np.logspace(np.log10(f_min), np.log10(f_max), scalogram2.shape[0])
                if dates:
                    x_datetimes = [tr.stats.starttime + s for s in t]
                    x_datetimes = [dt.datetime for dt in x_datetimes]
                    x, y = np.meshgrid(mdates.date2num(x_datetimes), freq)
                else:

                    x, y = np.meshgrid(t, freq)


            min_cwt = self.minlevelCB.value()
            max_cwt = 0

            value, idx = self.find_nearest(y[:,1], 1.0)
            scalogram_period = scalogram2[0:idx, :]

            scalogram2 = scalogram2[idx:, :]

            x_period = x[0:idx,:]
            y_period = 1/(y[0:idx, :])

            x_freq = x[idx:, :]
            y_freq = y[idx:, :]


            if self.res_factor <= 1:
                self.canvas_plot3.plot_contour(x_freq, y_freq, scalogram2, axes_index=1, levels=20, clear_plot=True,
                                               clabel="Power [dB]",
                                               cmap=self.colourCB.currentText(), vmin=min_cwt, vmax=max_cwt)

                self.canvas_plot3.plot_contour(x_period, 10*np.log(y_period), scalogram_period, axes_index=2, levels=20,
                                               clear_plot=True, clabel="Power [dB]",
                                               cmap=self.colourCB.currentText(), vmin=min_cwt, vmax=max_cwt)

            elif self.res_factor > 1:
                self.canvas_plot3.pcolormesh(x_freq, y_freq, scalogram2, axes_index=1, clear_plot=True, clabel="Power [dB]",
                                             cmap=self.colourCB.currentText(), vmin=min_cwt, vmax=max_cwt,
                                             shading='nearest')

                self.canvas_plot3.pcolormesh(x_period, y_period, scalogram_period, axes_index=2, clear_plot=True,
                                             clabel ="Power [dB]", cmap=self.colourCB.currentText(), vmin=min_cwt,
                                             vmax=max_cwt, shading='nearest')

            ax_period = self.canvas_plot3.get_axe(1)
            ax_period.set_yscale('log')
            ax_period = self.canvas_plot3.get_axe(2)
            ax_period.invert_yaxis()
            ax_period.set_yscale('log')

            if not self.activate_xaxis:
                formatter = mdt.DateFormatter('%H:%M:%S')
                ax2 = self.canvas_plot3.get_axe(2)
                ax2.xaxis.set_major_formatter(formatter)
                self.canvas_plot3.set_xlabel(2, "Date")
            else:
                self.canvas_plot3.set_xlabel(2, "Time (s)")
            self.canvas_plot3.set_ylabel(0, "Amplitude ")
            self.canvas_plot3.set_ylabel(1, "Frequency (Hz)")
            self.canvas_plot3.set_ylabel(2, "Period (s)")

        else:
            pass

        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)

    def key_pressed(self, event):
        selection = self.selectCB.currentText()

        if event.key == 'w':
            self.plot_seismogram()

        if event.key == 'q':
            if selection == "Seismogram 1":
                [tr, t] = self.get_data()
                x1, y1 = event.xdata, event.ydata
                if not self.activate_xaxis:
                    tt = UTCDateTime(mdt.num2date(x1))
                else:
                    tt = tr.stats.starttime + x1
                set_qdatetime(tt, self.starttime_date)
                self.canvas_plot1.draw_arrow(x1, 0, arrow_label="st", color="purple", linestyles='--', picker=False)
            elif selection == "Seismogram 2":
                [tr, t] = self.get_data()
                x1, y1 = event.xdata, event.ydata
                if not self.activate_xaxis:
                    tt = UTCDateTime(mdt.num2date(x1))
                else:
                    tt = tr.stats.starttime + x1
                set_qdatetime(tt, self.starttime_date)
                self.canvas_plot2.draw_arrow(x1, 0, arrow_label="st", color="purple", linestyles='--', picker=False)

            elif selection == "Seismogram 3":
                x1, y1 = event.xdata, event.ydata
                [tr, t] = self.get_data()
                if not self.activate_xaxis:
                    tt = UTCDateTime(mdt.num2date(x1))
                else:
                    tt = tr.stats.starttime + x1
                set_qdatetime(tt, self.starttime_date)
                self.canvas_plot3.draw_arrow(x1, 0, arrow_label="st", color="purple", linestyles='--', picker=False)

        if event.key == 'e':

            if selection == "Seismogram 1":
                [tr, t] = self.get_data()
                x1, y1 = event.xdata, event.ydata
                if not self.activate_xaxis:
                    tt = UTCDateTime(mdt.num2date(x1))
                else:
                    tt = tr.stats.starttime + x1
                set_qdatetime(tt, self.endtime_date)
                self.canvas_plot1.draw_arrow(x1, 0, arrow_label="et", color="purple", linestyles='--',
                                              picker=False)
            elif selection == "Seismogram 2":
                [tr, t] = self.get_data()
                x1, y1 = event.xdata, event.ydata
                if not self.activate_xaxis:
                    tt = UTCDateTime(mdt.num2date(x1))
                else:
                    tt = tr.stats.starttime + x1
                set_qdatetime(tt, self.endtime_date)
                self.canvas_plot2.draw_arrow(x1, 0, arrow_label="et", color="purple", linestyles='--',
                                              picker=False)

            elif selection == "Seismogram 3":
                x1, y1 = event.xdata, event.ydata
                [tr, t] = self.get_data()
                if not self.activate_xaxis:
                    tt = UTCDateTime(mdt.num2date(x1))
                else:
                    tt = tr.stats.starttime + x1
                set_qdatetime(tt, self.endtime_date)
                self.canvas_plot3.draw_arrow(x1, 0, arrow_label="et", color="purple", linestyles='--', picker=False)



    def open_help(self):
        open_url(self.url)

    # mouse actions

    def on_click_matplotlib(self, event, canvas):
        #self.picks = {'ID': [], 'Time': [], 'Amplitude': [], 'Phase': []}
        selection = self.selectCB.currentText()
        output_dir = os.path.join(ROOT_DIR, 'picks_timefreq')
        output_path = os.path.join(ROOT_DIR, 'picks_timefreq', 'picks.txt')

        if os.path.isdir(output_dir):
            pass
        else:
            os.makedirs(output_dir)

        if isinstance(canvas, MatplotlibCanvas):
            x1, y1 = event.xdata, event.ydata

            [tr, t] = self.get_data()
            # Save in a dataframe the pick value
            if selection == "Seismogram 1":
                self.canvas_plot1.draw_arrow(x1, 0, arrow_label=self.pickCB.currentText(), color="red", linestyles='--', picker=False)

            elif selection == "Seismogram 2":
                self.canvas_plot2.draw_arrow(x1, 0, arrow_label=self.pickCB.currentText(), color="red", linestyles='--', picker=False)

            elif selection == "Seismogram 3":
                self.canvas_plot3.draw_arrow(x1, 0, arrow_label=self.pickCB.currentText(), color="red", linestyles='--', picker=False)

            if not self.activate_xaxis:
                tt = UTCDateTime(mdt.num2date(x1))
            else:
                tt = tr.stats.starttime + x1

            diff = tt - tr.stats.starttime #

            idx_amplitude = int(tr.stats.sampling_rate * diff)
            amplitude = tr.data[idx_amplitude]

            self.picks['ID'].append(tr.id)
            self.picks['Time'].append(tt.isoformat()[:-5])
            self.picks['Amplitude'].append(amplitude)
            self.picks['Phase'].append(self.pickCB.currentText())

            df = pd.DataFrame(self.picks)
            print("Output Pick at: ", output_path)
            print(df)
            df.to_csv(output_path, index=False, header=True)



        # elif selection == "Wigner Spectrogram":
        #
        #     win = int(self.mt_window_lengthDB.value() * tr.stats.sampling_rate)
        #     tbp = self.time_bandwidth_DB.value()
        #     ntapers = self.number_tapers_mtSB.value()
        #     f_min = self.freq_min_mtDB.value()
        #     f_max = self.freq_max_mtDB.value()
        #     wignerspec = WignerVille(self.file_selector.file_path, win, tbp, ntapers, f_min, f_max)
        #
        #     if self.trimCB.isChecked() and diff >= 0:
        #         x, y, log_spectrogram = wignerspec.compute_wigner_spectrogram(tr, start_time=ts, end_time=te, res = self.res_factor)
        #     else:
        #         x, y, log_spectrogram = wignerspec.compute_wigner_spectrogram(tr, res = self.res_factor)
        #
        #     if order == "Seismogram 1":
        #
        #         if self.res_factor <= 1:
        #             self.canvas_plot1.plot_contour(x, y, log_spectrogram, axes_index=1, clear_plot=True,
        #                                            clabel="Rel Power ",cmap=self.colourCB.currentText())
        #         elif self.res_factor > 1:
        #             self.canvas_plot1.pcolormesh(x, y, log_spectrogram, axes_index=1, clear_plot=True,
        #                                          clabel="Rel Power ", cmap=self.colourCB.currentText())
        #         # elif self.typeCB.currentText() == 'imshow':
        #         #     self.canvas_plot1.image(x, y, log_spectrogram, axes_index=1, clear_plot=True, clabel="Power [dB]",
        #         #                          cmap=self.colourCB.currentText())
        #         self.canvas_plot1.set_xlabel(1, "Time (s)")
        #         self.canvas_plot1.set_ylabel(0, "Amplitude ")
        #         self.canvas_plot1.set_ylabel(1, "Frequency (Hz)")
        #
        #     elif order == "Seismogram 2":
        #         if self.res_factor <= 1:
        #             self.canvas_plot2.plot_contour(x, y, log_spectrogram, axes_index=1, clear_plot=True, clabel="Power [dB]",
        #              cmap=self.colourCB.currentText())
        #         elif self.res_factor > 1:
        #             self.canvas_plot2.pcolormesh(x, y, log_spectrogram, axes_index=1, clear_plot=True, clabel="Power [dB]",
        #                                            cmap=self.colourCB.currentText())
        #         # elif self.typeCB.currentText() == 'imshow':
        #         #     self.canvas_plot2.image(x, y, log_spectrogram, axes_index=1, clear_plot=True, clabel="Power [dB]",
        #         #                              cmap=self.colourCB.currentText())
        #         self.canvas_plot2.set_xlabel(1, "Time (s)")
        #         self.canvas_plot2.set_ylabel(0, "Amplitude ")
        #         self.canvas_plot2.set_ylabel(1, "Frequency (Hz)")
        #