import pickle
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, UTCDateTime, Trace, Inventory
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Gui.Frames import BaseFrame, \
MatplotlibCanvas, UiArrayAnalysisFrame, CartopyCanvas, MatplotlibFrame, MessageDialog
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_coordinates import StationsCoords
from isp.Gui.Frames.vespagram import Vespagram
from isp.Gui.Frames.slowness_map import SlownessMap
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, convert_qdatetime_utcdatetime, set_qdatetime
from isp.Gui import pw, pqg, pyc
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QStyle
import os
import matplotlib.dates as mdt
from datetime import date
import pandas as pd
from isp.Utils import MseedUtil
from isp.arrayanalysis import array_analysis
from isp import ROOT_DIR
from isp.Utils.subprocess_utils import exc_cmd
from isp.Gui.Frames.help_frame import HelpDoc
from sys import platform
from isp.arrayanalysis.backprojection_tools import back_proj_organize, backproj
from isp.arrayanalysis.plot_bp import plot_bp


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
        self.stack = None
        self.canvas = MatplotlibCanvas(self.responseMatWidget)
        self.canvas_fk = MatplotlibCanvas(self.widget_fk,nrows=4)
        self.canvas_slow_map = MatplotlibCanvas(self.widget_slow_map)
        self.canvas_fk.on_double_click(self.on_click_matplotlib)
        self.canvas_stack = MatplotlibCanvas(self.widget_stack)
        self.cartopy_canvas = CartopyCanvas(self.widget_map)
        self.canvas.set_new_subplot(1, ncols=1)

        #Binding

        self.root_pathBP_bind = BindPyqtObject(self.rootPathFormBP)

        self.metadata_path_bindBP = BindPyqtObject(self.datalessPathFormBP, self.onChange_metadata_path)
        self.output_path_bindBP = BindPyqtObject(self.outputPathFormBP, self.onChange_metadata_path)
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


        # Bind buttons BackProjection
        self.selectDirBtnBP.clicked.connect(lambda: self.on_click_select_directory(self.root_pathBP_bind))
        self.datalessBtnBP.clicked.connect(lambda: self.on_click_select_metadata_file(self.metadata_path_bindBP))
        self.outputBtn.clicked.connect(lambda: self.on_click_select_directory(self.output_path_bindBP))

        #Action Buttons
        self.arfBtn.clicked.connect(lambda: self.arf())
        self.runFKBtn.clicked.connect(lambda: self.FK_plot())
        self.plotBtnBP.clicked.connect(lambda: self.plot_seismograms(FK=False))
        self.actionSettings.triggered.connect(lambda: self.open_parameters_settings())
        self.actionProcessed_Seimograms.triggered.connect(self.write)
        self.actionStacked_Seismograms.triggered.connect(self.write_stack)
        self.stationsBtnBP.clicked.connect(lambda: self.stationsInfo(FK=False))
        self.mapBtn.clicked.connect(self.stations_map)
        self.actionCreate_Stations_File.triggered.connect(self.stations_coordinates)
        self.arf_write_coordsBtn.clicked.connect(self.stations_coordinates)
        self.arfLoad_coordsBtn.clicked.connect(self.load_path)
        self.actionLoad_Stations_File.triggered.connect(self.load_path)
        self.actionRunVespagram.triggered.connect(self.open_vespagram)
        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+O'), self)
        self.shortcut_open.activated.connect(self.open_solutions)
        self.create_gridBtn.clicked.connect(self.create_grid)
        self.actionOpen_Help.triggered.connect(lambda: self.open_help())
        self.load_videoBtn.clicked.connect(self.loadvideoBP)
        self.actionOpenSlowness.triggered.connect(lambda: self.open_slownessMap())

        # help Documentation
        self.help = HelpDoc()

        # Slowness Map
        self.__slownessMap = SlownessMap()

        # Parameters settings
        self.__parameters = ParametersSettings()

        # Stations Coordinates
        self.__stations_coords = StationsCoords()

        # picks
        self.picks =  {'Time': [], 'Phase': [], 'BackAzimuth':[], 'Slowness':[], 'Power':[]}

        # video
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.backprojection_widget)
        self.player.stateChanged.connect(self.mediaStateChanged)
        self.player.positionChanged.connect(self.positionChanged)
        self.player.durationChanged.connect(self.durationChanged)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_bp)
        self.positionSlider.sliderMoved.connect(self.setPosition)


    def mediaStateChanged(self):
         if self.player.state() == QMediaPlayer.PlayingState:
             self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
         else:
             self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
         self.positionSlider.setValue(position)

    def durationChanged(self, duration):
         self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
         self.player.setPosition(position)

    def open_slownessMap(self):
        self.__slownessMap.show()

    def stations_coordinates(self):
        self.__stations_coords.show()

    def open_vespagram(self):
        if self.st and self.inventory and self.t1 and self.t2:
            self.__vespagram = Vespagram(self.st, self.inventory, self.t1, self.t2)
            self.__vespagram.show()


    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path


    def onChange_metadata_path(self, value):

        md = MessageDialog(self)
        try:

            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
            md.set_info_message("Loaded Metadata, please check your terminal for further details")

        except:

            md.set_error_message("Something went wrong. Please check your metada file is a correct one")


    def on_click_select_metadata_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]

    def load_path(self):
        selected_file = pw.QFileDialog.getOpenFileName(self, "Select Stations Coordinates file")
        self.path_file = selected_file[0]
        df = pd.read_csv(self.path_file, delim_whitespace=True)
        n = len(df)
        self.coords = np.zeros([n, 3])
        for i in range(n):
            #coords[i]=data[i]
            self.coords[i] = np.array([df['Lon'][i], df['Lat'][i], df['Depth'][i]])
        print(self.coords)

    def arf(self):
        try:
            if self.coords.all():

                wavenumber = array_analysis.array()
                arf = wavenumber.arf(self.coords, self.fmin_bind.value, self.fmax_bind.value, self.smax_bind.value, self.grid_bind.value)

                slim = self.smax_bind.value
                x = y = np.linspace(-1 * slim, slim, len(arf))

                self.canvas.clear()
                #self.canvas.set_new_subplot(nrows=1, ncols=1)
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
            Lat_mean = sum(df["Lat"].tolist())/len(df["Lat"].tolist())
            Long_mean = sum(df["Lon"].tolist())/len(df["Lon"].tolist())
            n = len(df)
            self.coords = np.zeros([n, 3])
            for i in range(n):
                 coords[df['Name'][i]]=[df['Lon'][i],df['Lat'][i]]
        try:
            self.cartopy_canvas.plot_map(Long_mean, Lat_mean, 0, 0, 0, 0, resolution="low", stations=coords)
        except:
            md = MessageDialog(self)
            md.set_info_message("Please load a stations file with array coordinates")

    def FK_plot(self):
        self.canvas_stack.set_new_subplot(nrows=1, ncols=1)
        starttime = convert_qdatetime_utcdatetime(self.starttime_date)
        endtime = convert_qdatetime_utcdatetime(self.endtime_date)
        selection = MseedUtil.filter_inventory_by_stream(self.st, self.inventory)

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
            #selection = self.inventory.select(station=self.stationLE.text(), channel=self.channelLE.text())
            selection = MseedUtil.filter_inventory_by_stream(self.st, self.inventory)
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


    def process_fk(self, stream: Stream, inventory:Inventory, starttime:UTCDateTime, endtime:UTCDateTime):

        self.bpWidget.setCurrentIndex(1)
        self.st = stream
        print(self.st)
        self.inventory = inventory
        # set times
        set_qdatetime(starttime, self.starttime_date)
        set_qdatetime(endtime, self.endtime_date)
        self.trimCB.setChecked(True)
        self.stream_frame = MatplotlibFrame(self.st, type='normal')
        self.stream_frame.show()


    def write(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                           pw.QFileDialog.DontUseNativeDialog)
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
        if self.stack is not None and len(self.stack) > 0:
            root_path = os.path.dirname(os.path.abspath(__file__))
            if "darwin" == platform:
                dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
            else:
                dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                               pw.QFileDialog.DontUseNativeDialog)
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

    ### New part back-projection

    def create_grid(self):
        area_coords = [self.minLonBP, self.maxLonBP,self.minLatBP, self.maxLatBP]
        bp = back_proj_organize(self, self.rootPathFormBP, self.datalessPathFormBP, area_coords, self.sxSB.value,
                                self.sxSB.value, self.depthSB.value)

        mapping = bp.create_dict()

        try:
            self.path_file = os.path.join(self.output_path_bindBP.value,"mapping.pkl")
            file_to_store = open(self.path_file, "wb")
            pickle.dump(mapping, file_to_store)
            md = MessageDialog(self)
            md.set_info_message("BackProjection grid created succesfully!!!")

        except:
            md = MessageDialog(self)
            md.set_error_message("Coundn't create a BackProjection grid")


    def run_bp(self):

        try:
            if os.path.exists(self.path_file):
                with open(self.path_file, 'rb') as handle:
                    mapping = pickle.load(handle)
        except:
            md = MessageDialog(self)
            md.set_error_message("Please you need try to previously create a BackProjection grid")

        power = backproj.run_back(self.st, mapping, self.time_winBP.value, self.stepBP.value,
                                  window=self.slide_winBP.value, multichannel=self.mcccCB.isChecked(),
                                  stack_process = self.methodBP.currentText())

        #plot_cum(power, mapping['area_coords'], self.cum_sumBP.value, self.st)
        plot_bp(power, mapping['area_coords'], self.cum_sumBP.value, self.st, output=self.output_path_bindBP.value)

        fname = os.path.join(self.output_path_bindBP.value, "power.pkl")

        file_to_store = open(fname, "wb")
        pickle.dump(power, file_to_store)

    def loadvideoBP(self):
        self.path_video, _ = pw.QFileDialog.getOpenFileName(self, "Choose your BackProjection",
                                                       ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        if self.path_video != '':

            self.player.setVideoOutput(self.backprojection_widget)
            self.player.setMedia(QMediaContent(pyc.QUrl.fromLocalFile(self.path_video)))
            md = MessageDialog(self)
            md.set_info_message("Video containing BackProjection succesfully loaded")
        else:
            md = MessageDialog(self)
            md.set_error_message("Video containing BackProjection couldn't be loaded")


    def play_bp(self):

        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def open_help(self):
        self.help.show()
