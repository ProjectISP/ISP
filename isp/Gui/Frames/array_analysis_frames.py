import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.DataProcessing.seismogram_analysis import SeismogramDataAdvanced
from isp.Gui.Frames import BaseFrame, \
MatplotlibCanvas, UiArrayAnalysisFrame, CartopyCanvas, MatplotlibFrame, MessageDialog
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_coordinates import StationsCoords
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, convert_qdatetime_utcdatetime
from isp.Gui import pw
import os
import matplotlib.dates as mdt

from isp.Utils import MseedUtil
from isp.arrayanalysis import array_analysis


class ArrayAnalysisFrame(BaseFrame, UiArrayAnalysisFrame):

    def __init__(self):
        super(ArrayAnalysisFrame, self).__init__()
        self.setupUi(self)
        self.__stations_dir = None
        self.stream_frame = None
        self.__metadata_manager = None
        self.inventory = {}
        self._stations_info = {}
        self._stations_coords = {}
        self.canvas = MatplotlibCanvas(self.responseMatWidget)
        self.canvas_fk = MatplotlibCanvas(self.widget_fk,nrows=4)
        self.canvas_slow_map = MatplotlibCanvas(self.widget_slow_map)
        self.canvas_fk.on_double_click(self.on_click_matplotlib)
        self.canvas_stack = MatplotlibCanvas(self.widget_stack)
        self.canvas_stack.figure.subplots_adjust(left=0.080, bottom=0.374, right=0.970, top=0.990, wspace=0.2, hspace=0.0)
        self.cartopy_canvas = CartopyCanvas(self.widget_map)
        self.canvas.set_new_subplot(1, ncols=1)

        #Binding
        self.root_path_bind = BindPyqtObject(self.rootPathForm)
        self.root_pathFK_bind = BindPyqtObject(self.rootPathFormFK)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.fmin_bind = BindPyqtObject(self.fminSB)
        self.fmax_bind = BindPyqtObject(self.fmaxSB)
        self.grid_bind = BindPyqtObject(self.gridSB)
        self.smax_bind = BindPyqtObject(self.smaxSB)


        self.fminFK_bind = BindPyqtObject(self.fminFKSB)
        self.fmaxFK_bind = BindPyqtObject(self.fmaxFKSB)
        self.overlap_bind = BindPyqtObject(self.overlapSB)
        self.timewindow_bind = BindPyqtObject(self.timewindowSB)
        self.smaxFK_bind = BindPyqtObject(self.slowFKSB)
        self.slow_grid_bind = BindPyqtObject(self.gridFKSB)

        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.selectDirBtnFK.clicked.connect(lambda: self.on_click_select_directory(self.root_pathFK_bind))
        self.datalessBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))

        #Action Buttons
        self.arfBtn.clicked.connect(lambda: self.arf())
        self.runFKBtn.clicked.connect(lambda: self.FK_plot())
        self.plotBtn.clicked.connect(lambda: self.plot_seismograms())
        self.actionSettings.triggered.connect(lambda: self.open_parameters_settings())
        self.actionWrite.triggered.connect(self.write)

        self.stationsBtn.clicked.connect(self.stationsInfo)
        self.stations_coordsBtn.clicked.connect(self.stations_coordinates)

        # Parameters settings
        self.__parameters = ParametersSettings()

        # Stations Coordinates
        self.__stations_coords = StationsCoords()

    def open_parameters_settings(self):
        self.__parameters.show()

    def stations_coordinates(self):
        self.__stations_coords.show()


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

    def arf(self):
        # Test get coordinates#
        coords = self.__stations_coords.getCoordinates()
        print(coords)

        coords_path = os.path.join(self.root_path_bind.value, "coords.txt")
        wavenumber = array_analysis.array()
        arf,coords = wavenumber.arf(coords_path, self.fmin_bind.value, self.fmax_bind.value, self.smax_bind.value)
        slim = self.smax_bind.value
        sstep = slim / len(arf)
        x = np.linspace(-1 * slim, slim, (slim - (-1 * slim) / sstep))
        y = np.linspace(-1 * slim, slim, (slim - (-1 * slim) / sstep))
        self.canvas.plot_contour(x, y, arf, axes_index=0, clabel="Power [dB]", cmap=plt.get_cmap("jet"))
        self.canvas.set_xlabel(0, "Sx (s/km)")
        self.canvas.set_ylabel(0, "Sy (s/km)")
        #lon=coords[:,1]
        #lat=coords[:,0]
        #depth=coords[:,2]
        #self.cartopy_canvas.plot_stations(lon, lat, depth, 0)

    def FK_plot(self):

        starttime = convert_qdatetime_utcdatetime(self.starttime_date)
        endtime = convert_qdatetime_utcdatetime(self.endtime_date)
        selection = self.inventory.select(station=self.stationLE.text(), channel = self.channelLE.text())
        wavenumber = array_analysis.array()
        relpower,abspower, AZ, Slowness, T = wavenumber.FK(self.st, selection, starttime, endtime,
        self.fminFK_bind.value, self.fmaxFK_bind.value, self.smaxFK_bind.value, self.slow_grid_bind.value,
        self.timewindow_bind.value, self.overlap_bind.value)
        self.canvas_fk.scatter3d(T, relpower,relpower, axes_index=0, clabel="Power [dB]")
        self.canvas_fk.scatter3d(T, abspower, relpower, axes_index=1, clabel="Power [dB]")
        self.canvas_fk.scatter3d(T, AZ, relpower, axes_index=2, clabel="Power [dB]")
        self.canvas_fk.scatter3d(T, Slowness, relpower, axes_index=3, clabel="Power [dB]")
        self.canvas_fk.set_ylabel(0, " Rel Power ")
        self.canvas_fk.set_ylabel(1, " Absolute Power ")
        self.canvas_fk.set_ylabel(2, " Back Azimuth ")
        self.canvas_fk.set_ylabel(3, " Slowness ")
        self.canvas_fk.set_xlabel(3, "Time [s]")
        ax = self.canvas_fk.get_axe(3)
        formatter = mdt.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation = 30)

    def on_click_matplotlib(self, event, canvas):
        if isinstance(canvas, MatplotlibCanvas):
            st = self.st.copy()
            wavenumber = array_analysis.array()
            selection = self.inventory.select(station=self.stationLE.text(), channel=self.channelLE.text())
            x1, y1 = event.xdata, event.ydata
            DT = x1
            Z, Sxpow, Sypow, coord = wavenumber.FKCoherence(st, selection, DT,
            self.fminFK_bind.value, self.fmaxFK_bind.value, self.smaxFK_bind.value, self.timewindow_bind.value,
                                   self.slow_grid_bind.value, self.methodSB.currentText())
            if self.methodSB.currentText() == "FK":
                clabel="Power"
            elif self.methodSB.currentText() == "MTP.COHERENCE":
                clabel = "Magnitude Coherence"

            Sx = np.arange(-1*self.smaxFK_bind.value, self.smaxFK_bind.value, self.slow_grid_bind.value)[np.newaxis]
            nx = len(Sx[0])
            x = y = np.linspace(-1*self.smaxFK_bind.value, self.smaxFK_bind.value, nx)
            X, Y = np.meshgrid(x, y)
            self.canvas_slow_map.plot_contour(X, Y, Z, axes_index=0, clabel=clabel, cmap=plt.get_cmap("jet"))
            self.canvas_slow_map.set_xlabel(0, "Sx [s/km]")
            self.canvas_slow_map.set_ylabel(0, "Sy [s/km]")
            # Call Stack and Plot###
            stream_stack, time = wavenumber.stack_stream(self.root_pathFK_bind.value, Sxpow, Sypow, coord)
            stack = wavenumber.stack(stream_stack)
            self.canvas_stack.plot(time, stack, axes_index = 0)
            self.canvas_stack.set_xlabel(0, " Time [s] ")
            self.canvas_stack.set_ylabel(0, "Stack Amplitude")

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def plot_seismograms(self):

        starttime = convert_qdatetime_utcdatetime(self.starttime_date)
        endtime = convert_qdatetime_utcdatetime(self.endtime_date)
        diff = endtime - starttime
        file_path = self.root_pathFK_bind.value
        obsfiles = []

        for dirpath, _, filenames in os.walk(file_path):
            for f in filenames:
                 if f != ".DS_Store":
                    obsfiles.append(os.path.abspath(os.path.join(dirpath, f)))
        obsfiles.sort()
        parameters = self.__parameters.getParameters()
        all_traces =[]
        trace_number = 0
        for file in obsfiles:
            sd = SeismogramDataAdvanced(file)
            if self.trimCB.isChecked() and diff >= 0:
                tr = sd.get_waveform_advanced(parameters, self.inventory, filter_error_callback=self.filter_error_message,
                    start_time=starttime, end_time=endtime, trace_number=trace_number)
            else:
                tr = sd.get_waveform_advanced(parameters, self.inventory, filter_error_callback=self.filter_error_message,
                                              trace_number=trace_number)

            all_traces.append(tr)
            trace_number = trace_number  + 1

        self.st = Stream(traces=all_traces)

        if self.selectCB.isChecked():
            self.st = self.st.select( station=self.stationLE.text(), channel=self.channelLE.text())
        self.stream_frame = MatplotlibFrame(self.st, type='normal')
        self.stream_frame.show()

    def stationsInfo(self):
        obsfiles = MseedUtil.get_mseed_files(self.root_pathFK_bind.value)
        obsfiles.sort()
        sd = []
        for file in obsfiles:
            st = SeismogramDataAdvanced(file)
            station = [st.stats.Network, st.stats.Station, st.stats.Location, st.stats.Channel, st.stats.StartTime,
                       st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]
            sd.append(station)
        self._stations_info = StationsInfo(sd, check=True)
        self._stations_info.show()

    def write(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        n=len(self.st)
        for j in range(n):
            tr=self.st[j]
            print(tr.id, "Writing data processed")
            path_output =  os.path.join(dir_path, tr.id)
            tr.write(path_output, format="MSEED")



