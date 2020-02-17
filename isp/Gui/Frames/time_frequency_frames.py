from enum import Enum, unique

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from obspy import read
from scipy.signal import hilbert

from isp.DataProcessing import SeismogramData
from isp.Exceptions import InvalidFile
from isp.Gui import pw
from isp.Gui.Frames import MatplotlibFrame, BaseFrame, UiTimeFrequencyFrame, FilesView, MessageDialog, \
    MatplotlibCanvas, TimeSelectorBox, FilterBox, SpectrumBox, UiTimeAnalysisWidget
from isp.Gui.Frames.qt_components import ParentWidget, StationInfoBox, EventInfoBox
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, add_save_load
from isp.Structures.structures import StationsStats
from isp.Utils import MseedUtil, ObspyUtil
from isp.seismogramInspector.MTspectrogram import MTspectrogram
from isp.seismogramInspector.ba_fast import ccwt_ba_fast
from isp.seismogramInspector.CWT_fast import cwt_fast

@unique
class Phases(Enum):

    Default = "Phase"
    PPhase = "P"
    PnPhase = "Pn"
    PgPhase = "Pg"
    SpPhase = "Sp"
    SnPhase = "Sn"
    SgPhase = "Sg"
    LgPhase = "Lg"

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    @classmethod
    def get_phases(cls):
        return [item.value for item in cls.__members__.values()]


@add_save_load()
class TimeAnalysisWidget(pw.QFrame, UiTimeAnalysisWidget):

    def __init__(self, parent, current_index=-1, parent_name=None):

        super(TimeAnalysisWidget, self).__init__()
        self.setupUi(self)
        ParentWidget.set_parent(parent, self, current_index)

        if parent_name:  # add parent name if given. Helps to make difference between the same widget
            self.parent_name = parent_name

        # embed matplotlib to qt widget
        self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=2)
        self.canvas.set_xlabel(1, "Time (s)")
        self.canvas.on_double_click(self.on_double_click_matplotlib)

        self.time_selector = TimeSelectorBox(self.PlotToolsWidget, 0)
        self.filter = FilterBox(self.PlotToolsWidget, 1)
        self.spectrum_box = SpectrumBox(self.PlotToolsWidget, 3)
        self.station_info = StationInfoBox(self.PlotToolsWidget, 4)

        # bind buttons
        self.plotSeismogramBtn.clicked.connect(lambda: self.on_click_plot_seismogram(self.canvas))
        self.plotArrivalsBtn.clicked.connect(lambda: self.on_click_plot_arrivals(self.canvas))
        self.spectrum_box.register_plot_mtp(lambda: self.on_click_mtp(self.canvas))
        self.spectrum_box.register_plot_cwt(lambda: self.on_click_cwt(self.canvas))

        self.__file_selector = None
        self.__event_info = None
        self.is_envelop_checked = False

    def register_file_selector(self, file_selector: FilesView):
        self.__file_selector = file_selector

    def set_event_info(self, event_into: EventInfoBox):
        self.__event_info = event_into

    @property
    def file_selector(self) -> FilesView:
        return self.__file_selector

    @property
    def event_info(self) -> EventInfoBox:
        return self.__event_info

    @property
    def tracer_stats(self):
        return ObspyUtil.get_stats(self.file_selector.file_path)

    @property
    def stream(self):
        return read(self.file_selector.file_path)

    @property
    def trace(self):
        return ObspyUtil.get_tracer_from_file(self.file_selector.file_path)

    def validate_file(self):
        if not MseedUtil.is_valid_mseed(self.file_selector.file_path):
            msg = "The file {} is not a valid mseed. Please, choose a valid format".\
                format(self.file_selector.file_name)
            md = MessageDialog(self)
            md.set_info_message(msg)
            raise InvalidFile(msg)

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def get_data(self):

        filter_value = self.filter.filter_value

        f_min = self.filter.min_freq
        f_max = self.filter.max_freq

        t1 = self.time_selector.start_time
        t2 = self.time_selector.end_time

        sd = SeismogramData.from_tracer(self.trace)
        return sd.get_waveform(filter_error_callback=self.filter_error_message,
                               filter_value=filter_value, f_min=f_min, f_max=f_max, start_time=t1, end_time=t2)

    def get_time_window(self):
        t1 = self.time_selector.start_time
        t2 = self.time_selector.end_time

        return t1, t2

    def plot_seismogram(self, canvas):
        t, s1 = self.get_data()
        canvas.plot(t, s1, 0, color="black",linewidth=0.5)

        if self.is_envelop_checked:
            analytic_sygnal = hilbert(s1)
            envelope = np.abs(analytic_sygnal)
            canvas.plot(t, envelope, 0, clear_plot=False, linewidth=0.5, color='sandybrown')

    def plot_mt_spectrogram(self, canvas: MatplotlibCanvas):
        win = int(self.spectrum_box.win_bind.value * self.tracer_stats.Sampling_rate)
        tbp = self.spectrum_box.tw_bind.value
        ntapers = self.spectrum_box.ntapers_bind.value
        f_min = self.filter.min_freq
        f_max = self.filter.max_freq
        ts, te = self.get_time_window()

        mtspectrogram = MTspectrogram(self.file_selector.file_path, win, tbp, ntapers, f_min, f_max)
        x, y, log_spectrogram = mtspectrogram.compute_spectrogram(start_time=ts, end_time=te,
                                                                  trace_filter=self.filter.filter_value)
        canvas.plot_contour(x, y, log_spectrogram, axes_index=1, clabel="Power [dB]",  cmap=plt.get_cmap("jet"))
        canvas.set_xlabel(1, "Time (s)")

    def plot_cwt_spectrogram(self, canvas: MatplotlibCanvas):
        tr = ObspyUtil.get_tracer_from_file(self.file_selector.file_path)
        ts, te = self.get_time_window()
        tr.trim(starttime=ts, endtime=te)
        tr.detrend(type="demean")

        f_min = 1. / self.spectrum_box.win_bind.value if self.filter.min_freq == 0 else self.filter.min_freq
        f_max = self.filter.max_freq
        ObspyUtil.filter_trace(tr, self.filter.filter_value, f_min, f_max)
        nf = 40
        tt = int(self.spectrum_box.win_bind.value * self.tracer_stats.Sampling_rate)
        wmin = self.spectrum_box.w1_bind.value
        wmax = self.spectrum_box.w2_bind.value
        npts = len(tr.data)
        [ba, nConv, frex, half_wave] = ccwt_ba_fast(npts, self.tracer_stats.Sampling_rate, f_min, f_max, wmin, wmax, tt, nf)
        cf, sc, scalogram = cwt_fast(tr.data, ba, nConv, frex, half_wave)
        #scalogram = ccwt(tr.data, self.tracer_stats.Sampling_rate, f_min, f_max, wmin, wmax, tt, nf)
      
        scalogram = np.abs(scalogram) ** 2

        t = np.linspace(0, self.tracer_stats.Delta * npts, npts)
        scalogram2 = 10 * (np.log10(scalogram / np.max(scalogram)))
        x, y = np.meshgrid(t, np.linspace(f_min, f_max, scalogram2.shape[0]))

        max_cwt = np.max(scalogram2)
        min_cwt = np.min(scalogram2)
        norm = Normalize(vmin=min_cwt, vmax=max_cwt)
        canvas.plot_contour(x, y, scalogram2, axes_index=1, clabel="Power [dB]", levels=100,
                            cmap=plt.get_cmap("jet"), norm=norm)
        canvas.set_xlabel(1, "Time (s)")

    def on_click_plot_seismogram(self, canvas):
        try:
            self.validate_file()
            self.update_station_info()
            self.plot_seismogram(canvas)
        except InvalidFile:
            pass

    def on_click_mtp(self, canvas):
        try:
            self.validate_file()
            self.update_station_info()
            self.plot_mt_spectrogram(canvas)
        except InvalidFile:
            pass

    def on_click_cwt(self, canvas):
        try:
            self.validate_file()
            self.update_station_info()
            self.plot_cwt_spectrogram(canvas)
        except InvalidFile:
            pass

    def on_double_click_matplotlib(self, event, canvas):
        pass

    def on_click_plot_arrivals(self, canvas):
        self.update_station_info()
        self.plot_arrivals(canvas)

    def update_station_info(self):
        self.station_info.set_basic_info(self.tracer_stats)

    def plot_arrivals(self, canvas):
        self.event_info.set_canvas(canvas)
        station_stats = StationsStats(self.tracer_stats.Station,
                                      self.station_info.latitude, self.station_info.longitude, 0.)
        self.event_info.plot_arrivals(0, self.time_selector.start_time, station_stats)


class TimeFrequencyFrame(BaseFrame, UiTimeFrequencyFrame):

    def __init__(self, ):
        super(TimeFrequencyFrame, self).__init__()
        self.setupUi(self)

        self.dayplot_frame = None

        # Bind buttons
        self.selectDirBtn.clicked.connect(self.on_click_select_directory)
        self.dayPlotBtn.clicked.connect(self.on_click_dayplot)

        # Bind qt objects
        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)

        self.event_info = EventInfoBox(self.mainToolsWidget, None)
        self.event_info.set_buttons_visibility(False)

        # Add file selector to the widget
        self.file_selector = FilesView(self.root_path_bind.value, parent=self.fileSelectorWidget,
                                       on_change_file_callback=lambda file_path: self.onChange_file(file_path))

        self.time_analysis_widget = TimeAnalysisWidget(self.canvasWidget, parent_name="time_analysis_0")
        self.time_analysis_widget2 = TimeAnalysisWidget(self.canvasWidget, parent_name="time_analysis_1")

        self.time_analysis_windows = [self.time_analysis_widget,  self.time_analysis_widget2]
        for taw in self.time_analysis_windows:
            taw.register_file_selector(self.file_selector)
            taw.set_event_info(self.event_info)

        self.actionPlotEnvelope.toggled.connect(self.on_envelop_toggle)

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

    def on_envelop_toggle(self, checked: bool):
        for taw in self.time_analysis_windows:
            taw.is_envelop_checked = checked

    def validate_file(self):
        if not MseedUtil.is_valid_mseed(self.file_selector.file_path):
            msg = "The file {} is not a valid mseed. Please, choose a valid format".\
                format(self.file_selector.file_name)
            md = MessageDialog(self)
            md.set_info_message(msg)
            raise InvalidFile(msg)

    def on_click_select_directory(self):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind.value)

        if dir_path:
            self.root_path_bind.value = dir_path

    def plot_day_view(self):
        self.dayplot_frame = MatplotlibFrame(self.stream, type='dayplot')
        self.dayplot_frame.show()

    def on_click_dayplot(self):
        try:
            self.validate_file()
            self.plot_day_view()
        except InvalidFile:
            pass