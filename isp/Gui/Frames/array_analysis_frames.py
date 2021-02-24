import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, UTCDateTime, Trace
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.DataProcessing.seismogram_analysis import SeismogramDataAdvanced
from isp.Gui.Frames import BaseFrame, \
MatplotlibCanvas, UiArrayAnalysisFrame, CartopyCanvas, MatplotlibFrame, MessageDialog
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_coordinates import StationsCoords
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.vespagram import Vespagram
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, convert_qdatetime_utcdatetime
from isp.Gui import pw, pqg
import os
import matplotlib.dates as mdt
from datetime import date
import pandas as pd
from isp.Utils import MseedUtil, AsycTime
from isp.arrayanalysis import array_analysis
from isp import ROOT_DIR
from isp.Utils.subprocess_utils import exc_cmd

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

        self.root_pathFK_bind = BindPyqtObject(self.rootPathFormFK)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.fmin_bind = BindPyqtObject(self.fminSB)
        self.fmax_bind = BindPyqtObject(self.fmaxSB)
        self.grid_bind = BindPyqtObject(self.gridSB)
        self.smax_bind = BindPyqtObject(self.smaxSB)

        # On select

        self.canvas_fk.register_on_select(self.on_select, rectprops=dict(alpha=0.2, facecolor='red'))


        self.fminFK_bind = BindPyqtObject(self.fminFKSB)
        self.fmaxFK_bind = BindPyqtObject(self.fmaxFKSB)
        self.overlap_bind = BindPyqtObject(self.overlapSB)
        self.timewindow_bind = BindPyqtObject(self.timewindowSB)
        self.smaxFK_bind = BindPyqtObject(self.slowFKSB)
        self.slow_grid_bind = BindPyqtObject(self.gridFKSB)

        # Bind buttons

        self.selectDirBtnFK.clicked.connect(lambda: self.on_click_select_directory(self.root_pathFK_bind))
        self.datalessBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))

        #Action Buttons
        self.arfBtn.clicked.connect(lambda: self.arf())
        self.runFKBtn.clicked.connect(lambda: self.FK_plot())
        self.plotBtn.clicked.connect(lambda: self.plot_seismograms())
        self.actionSettings.triggered.connect(lambda: self.open_parameters_settings())
        self.actionProcessed_Seimograms.triggered.connect(self.write)
        self.actionStacked_Seismograms.triggered.connect(self.write_stack)

        self.stationsBtn.clicked.connect(self.stationsInfo)
        self.mapBtn.clicked.connect(self.stations_map)
        self.actionCreate_Stations_File.triggered.connect(self.stations_coordinates)
        self.actionLoad_Stations_File.triggered.connect(self.load_path)
        self.actionRunVespagram.triggered.connect(self.open_vespagram)
        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+O'), self)
        self.shortcut_open.activated.connect(self.open_solutions)

        # Parameters settings
        self.__parameters = ParametersSettings()

        # Stations Coordinates
        self.__stations_coords = StationsCoords()
        # picks
        self.picks =  {'Time': [], 'Phase': [], 'BackAzimuth':[], 'Slowness':[], 'Power':[]}

    def open_parameters_settings(self):
        self.__parameters.show()

    def stations_coordinates(self):
        self.__stations_coords.show()


    def open_vespagram(self):
        if self.st and self.inventory and self.t1 and self.t2:
            self.__vespagram = Vespagram(self.st, self.inventory, self.t1, self.t2)
            self.__vespagram.show()


    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        if dir_path:
            bind.value = dir_path

    @AsycTime.run_async()
    def onChange_metadata_path(self, value):
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
        except:
            pass

    def load_path(self):
        selected_file = pw.QFileDialog.getOpenFileName(self, "Select Stations Coordinates file")
        self.path_file = selected_file[0]
        df = pd.read_csv(self.path_file, delim_whitespace=True)
        n = len(df)
        self.coords = np.zeros([n, 3])
        for i in range(n):
            #coords[i]=data[i]
            self.coords[i] = np.array([df['Lon'][i], df['Lat'][i], df['Depth'][i]])

    def arf(self):
        try:
            if self.coords.all():

                wavenumber = array_analysis.array()
                arf = wavenumber.arf(self.coords, self.fmin_bind.value, self.fmax_bind.value,
                                             self.smax_bind.value, self.grid_bind.value)

                slim = self.smax_bind.value
                x = y = np.linspace(-1 * slim, slim, len(arf))
                self.canvas.plot_contour(x, y, arf, axes_index=0, clabel="Power [dB]", cmap=plt.get_cmap("jet"))
                self.canvas.set_xlabel(0, "Sx (s/km)")
                self.canvas.set_ylabel(0, "Sy (s/km)")
        except:
            md = MessageDialog(self)
            md.set_error_message("Couldn't compute ARF, please check if you have loaded stations coords")


    def stations_map(self):
        coords = {}

        if self.path_file:
            df = pd.read_csv(self.path_file, delim_whitespace=True)
            n = len(df)
            self.coords = np.zeros([n, 3])
            for i in range(n):
                 coords[df['Name'][i]]=[df['Lon'][i], df['Lat'][i]]
        try:
            self.cartopy_canvas.plot_map(df['Lon'][0], df['Lat'][0], 0, 0, 0, 0, resolution = "low",
                                     stations = coords)
        except:
            pass

    def FK_plot(self):
        self.canvas_stack.set_new_subplot(nrows=1, ncols=1)
        starttime = convert_qdatetime_utcdatetime(self.starttime_date)
        endtime = convert_qdatetime_utcdatetime(self.endtime_date)
        selection = self.inventory.select(station=self.stationLE.text(), channel = self.channelLE.text())
        if self.trimCB.isChecked():
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
            self.canvas_fk.set_ylabel(3, " Slowness [s/km] ")
            self.canvas_fk.set_xlabel(3, " Time [s] ")
            ax = self.canvas_fk.get_axe(3)
            formatter = mdt.DateFormatter('%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_tick_params(rotation = 30)
        else:
            md = MessageDialog(self)
            md.set_info_message("Please select dates and then check Trim box")


    def on_click_matplotlib(self, event, canvas):
        output_path = os.path.join(ROOT_DIR, 'arrayanalysis', 'dataframe.csv')

        if isinstance(canvas, MatplotlibCanvas):
            st = self.st.copy()
            wavenumber = array_analysis.array()
            selection = self.inventory.select(station=self.stationLE.text(), channel=self.channelLE.text())
            x1, y1 = event.xdata, event.ydata
            DT = x1
            Z, Sxpow, Sypow, coord = wavenumber.FKCoherence(st, selection, DT,
            self.fminFK_bind.value, self.fmaxFK_bind.value, self.smaxFK_bind.value, self.timewindow_bind.value,
                                   self.slow_grid_bind.value, self.methodSB.currentText())

            backacimuth = wavenumber.azimuth2mathangle(np.arctan2(Sypow, Sxpow) * 180 / np.pi)
            slowness = np.abs(Sxpow, Sypow)
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

            # Save in a dataframe the pick value
            x1 = wavenumber.gregorian2date(x1)

            self.picks['Time'].append(x1.isoformat())
            self.picks['Phase'].append(self.phaseCB.currentText())
            self.picks['BackAzimuth'].append(backacimuth[0])
            self.picks['Slowness'].append(slowness[0])
            self.picks['Power'].append(np.max(Z))
            df = pd.DataFrame(self.picks)
            df.to_csv(output_path, index = False, header=True)

            # Call Stack and Plot###
            #stream_stack, time = wavenumber.stack_stream(self.root_pathFK_bind.value, Sxpow, Sypow, coord)

            if st:
                st2 = self.st.copy()
                # Align for the maximum power and give the data of the traces
                stream_stack, self.time, self.stats = wavenumber.stack_stream(st2, Sxpow, Sypow, coord)
                # stack the traces
                self.stack = wavenumber.stack(stream_stack, stack_type = self.stackCB.currentText())
                self.canvas_stack.plot(self.time, self.stack, axes_index = 0, linewidth = 0.75)
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
        if dir_path:
            n=len(self.st)
            try:
                if len(n)>0:
                    for j in range(n):
                        tr = self.st[j]
                        t1 = tr.stats.starttime
                        id = tr.id+"."+"D"+"."+str(t1.year)+"."+str(t1.julday)
                        print(tr.id, "Writing data processed")
                        path_output = os.path.join(dir_path, id)
                        tr.write(path_output, format="MSEED")
                else:
                    md = MessageDialog(self)
                    md.set_info_message("Nothing to write")
            except:
                pass

    def write_stack(self):
            if len(self.stack)>0:
                root_path = os.path.dirname(os.path.abspath(__file__))
                dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
                if dir_path:
                    tr = Trace(data=self.stack, header=self.stats)
                    file = os.path.join(dir_path,tr.id)
                    tr.write(file, format="MSEED")
            else:
                md = MessageDialog(self)
                md.set_info_message("Nothing to write")



    def __to_UTC(self, DT):

        # Convert start from Greogorian to actual date
        Time = DT
        Time = Time - int(Time)
        d = date.fromordinal(int(DT))
        date1 = d.isoformat()
        H = (Time * 24)
        H1 = int(H)  # Horas
        minutes = (H - int(H)) * 60
        minutes1 = int(minutes)
        seconds = (minutes - int(minutes)) * 60
        H1 = str(H1).zfill(2)
        minutes1 = str(minutes1).zfill(2)
        seconds = "%.2f" % seconds
        seconds = str(seconds).zfill(2)
        DATE = date1 + "T" + str(H1) + minutes1 + seconds

        t1 = UTCDateTime(DATE)
        return t1

    def on_select(self, ax_index, xmin, xmax):
        self.t1 = self.__to_UTC(xmin)
        self.t2 = self.__to_UTC(xmax)

    def open_solutions(self):
        output_path = os.path.join(ROOT_DIR,'arrayanalysis','dataframe.csv')
        try:
            command = "{} {}".format('open', output_path)
            exc_cmd(command, cwd = ROOT_DIR)
        except:
            md = MessageDialog(self)
            md.set_error_message("Coundn't open solutions file")
