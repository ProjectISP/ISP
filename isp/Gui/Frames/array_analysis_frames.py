import pickle
import subprocess
import platform
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from obspy import Stream, UTCDateTime, Trace, Inventory
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.DataProcessing.plot_tools_manager import PlotToolsManager
from isp.Gui.Frames import BaseFrame, \
    MatplotlibCanvas, UiArrayAnalysisFrame, CartopyCanvas, MatplotlibFrame, MessageDialog
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_coordinates import StationsCoords
from isp.Gui.Frames.vespagram import Vespagram
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from isp.Gui import pw, pqg, pyc
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QStyle
import os
import matplotlib.dates as mdt
import pandas as pd
from isp.Utils import MseedUtil, AsycTime
from isp.Utils.subprocess_utils import open_url
from isp.arrayanalysis import array_analysis
from isp import ROOT_DIR
from isp.arrayanalysis.backprojection_tools import back_proj_organize, backproj
from isp.arrayanalysis.plot_bp import plot_bp
from isp.seismogramInspector.signal_processing_advanced import find_nearest
from PyQt5.QtCore import Qt

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
        self.url='https://projectisp.github.io/ISP_tutorial.github.io/aa/'
        self.canvas = MatplotlibCanvas(self.responseMatWidget)
        self.canvas_fk = MatplotlibCanvas(self.widget_fk, nrows=4)
        self.canvas_slow_map = MatplotlibCanvas(self.widget_slow_map)
        self.canvas_fk.on_double_click(self.on_click_matplotlib)
        self.cartopy_canvas = CartopyCanvas(self.widget_map)
        self.canvas.set_new_subplot(1, ncols=1)

        # Binding

        self.root_pathBP_bind = BindPyqtObject(self.rootPathFormBP)
        self.metadata_path_bindBP = BindPyqtObject(self.datalessPathFormBP, self.onChange_metadata_path)
        self.output_path_bindBP = BindPyqtObject(self.outputPathFormBP, self.onChange_metadata_path)
        self.fmin_bind = BindPyqtObject(self.fminSB)
        self.fmax_bind = BindPyqtObject(self.fmaxSB)
        self.grid_bind = BindPyqtObject(self.gridSB)
        self.smax_bind = BindPyqtObject(self.smaxSB)

        # On select
        self.canvas_fk.register_on_select(self.on_multiple_select,
                                          button=MouseButton.RIGHT, sharex=True, rectprops=dict(alpha=0.2,
                                                                                                facecolor='blue'))
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

        # Action Buttons
        self.arfBtn.clicked.connect(lambda: self.arf())
        self.runFKBtn.clicked.connect(lambda: self.FK_plot())
        self.plotBtnBP.clicked.connect(lambda: self.plot_seismograms(FK=False))
        self.actionStacked_Seismograms.triggered.connect(self.write_stack)
        self.stationsBtnBP.clicked.connect(lambda: self.stationsInfo(FK=False))
        self.mapBtn.clicked.connect(self.stations_map)
        self.actionCreate_Stations_File.triggered.connect(self.stations_coordinates)
        self.arf_write_coordsBtn.clicked.connect(self.stations_coordinates)
        self.arfLoad_coordsBtn.clicked.connect(self.load_path)
        self.actionLoad_Stations_File.triggered.connect(self.load_path)
        self.actionRunVespagram.triggered.connect(self.open_vespagram)
        self.runVespaBtn.clicked.connect(self.open_vespagram)
        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+O'), self)
        self.shortcut_open.activated.connect(self.open_solutions)
        self.create_gridBtn.clicked.connect(self.create_grid)
        self.actionOpen_Help.triggered.connect(lambda: self.open_help())
        self.load_videoBtn.clicked.connect(self.loadvideoBP)

        # Parameters settings
        self.__parameters = ParametersSettings()

        # Stations Coordinates
        self.__stations_coords = StationsCoords()

        # picks
        self.picks = {'Time': [], 'Phase': [], 'BackAzimuth': [], 'Slowness': [], 'Power': []}

        # video
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.backprojection_widget)
        self.player.stateChanged.connect(self.mediaStateChanged)
        self.player.positionChanged.connect(self.positionChanged)
        self.player.durationChanged.connect(self.durationChanged)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_bp)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        # Dialog
        self.progress_dialog = pw.QProgressDialog(self)
        self.progress_dialog.setRange(0, 0)
        self.progress_dialog.setWindowTitle('Processing.....')
        self.progress_dialog.setLabelText('Please Wait')
        self.progress_dialog.setWindowIcon(self.windowIcon())
        self.progress_dialog.setWindowTitle(self.windowTitle())
        self.progress_dialog.close()

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

    def stations_coordinates(self):
        self.__stations_coords.show()

    def open_vespagram(self):
        if self.st and self.inventory and self.t1 and self.t2:
            max_values = self.__get_max_values()
            self.__vespagram = Vespagram(self.st, self.inventory, self.t1, self.t2, max_values, self.fminFK_bind.value,
                                         self.fmaxFK_bind.value, self.timewindow_bind.value)
            self.__vespagram.show()

    def __get_max_values(self):

        max_values = {}
        max_values["max_abs_power"] = np.max(self.abspower)
        max_values["max_relpower"] = np.max(self.relpower)
        idx_max_relpower, val = find_nearest(self.relpower, max_values["max_relpower"])

        max_values["max_slowness"] = self.Slowness[idx_max_relpower]
        max_values["max_azimuth"] = self.AZ[idx_max_relpower]

        return max_values

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
        self.df_stations = pd.read_csv(self.path_file, delim_whitespace=True)
        # List of columns to check
        columns_to_check = ['Name', 'Lon', 'Lat', 'Depth']

        # Check if each column exists in the DataFrame
        exists = {col: col in self.df_stations.columns for col in columns_to_check}

        if exists:

            n = len(self.df_stations)
            self.coords = np.zeros([n, 3])
            self.coords_map = {}

            for i in range(n):
                self.coords[i] = np.array([self.df_stations['Lon'][i], self.df_stations['Lat'][i],
                                           self.df_stations['Depth'][i]])
                self.coords_map[self.df_stations['Name'][i]] = [self.df_stations['Lon'][i],
                                                                self.df_stations['Lat'][i]]

            print(self.coords_map)
            self.mapBtn.setEnabled(True)
            md = MessageDialog(self)
            md.set_info_message("Ready to compute the Array Response Funtion", "Fill the parameters according to "
                                                                               "your array aperture and interdistance, "
                                                                               "Then click at ARF")
        else:

            md = MessageDialog(self)
            md.set_error_message("Please check tat your file contains four columns", "Name, Lon, Lat and Depth")

    def arf(self):
        try:
            if self.coords.all():
                wavenumber = array_analysis.array()
                arf = wavenumber.arf(self.coords, self.fmin_bind.value, self.fmax_bind.value, self.smax_bind.value,
                                     self.grid_bind.value)

                slim = self.smax_bind.value
                x = y = np.linspace(-1 * slim, slim, len(arf))

                self.canvas.clear()

                self.canvas.plot_contour(x, y, arf, axes_index=0, clabel="Power [dB]", cmap=plt.get_cmap("jet"))
                self.canvas.set_xlabel(0, "Sx (s/km)")
                self.canvas.set_ylabel(0, "Sy (s/km)")
        except:
            md = MessageDialog(self)
            md.set_error_message("Couldn't compute ARF, please check if you have loaded stations coords")

    def stations_map(self):

        Lat_mean = sum(self.df_stations["Lat"].tolist()) / len(self.df_stations["Lat"].tolist())
        Long_mean = sum(self.df_stations["Lon"].tolist()) / len(self.df_stations["Lon"].tolist())

        try:
            self.cartopy_canvas.plot_map(Long_mean, Lat_mean, 0, 0, 0, 0, resolution="low", stations=self.coords_map)
        except:
            md = MessageDialog(self)
            md.set_info_message("Please load a stations file with array coordinates")




    def FK_plot(self):
        self.__FK_plot()
        self.progress_dialog.exec()
        md = MessageDialog(self)
        md.set_info_message("FK analysis finished", "Double click to see the slowness map / "
                                                    "Drag holding right button to select time spam to analyze in "
                                                    "Vespagram module")

    @AsycTime.run_async()
    def __FK_plot(self):
        print("Starting FK analysis")
        # starttime = convert_qdatetime_utcdatetime(self.starttime_date)
        # endtime = convert_qdatetime_utcdatetime(self.endtime_date)
        self.phaseLE.clear()
        self.trace_stack = None
        self.phaseLE.setPlainText("Ready to save your picks, at isp/arrayanalysis/array_picks.txt")
        selection = MseedUtil.filter_inventory_by_stream(self.st, self.inventory)

        wavenumber = array_analysis.array()
        self.relpower, self.abspower, self.AZ, self.Slowness, self.T = wavenumber.FK(self.st, selection, self.starttime,
                                                                                     self.endttime,
                                                                                     self.fminFK_bind.value,
                                                                                     self.fmaxFK_bind.value,
                                                                                     self.smaxFK_bind.value,
                                                                                     self.slow_grid_bind.value,
                                                                                     self.timewindow_bind.value,
                                                                                     self.overlap_bind.value)

        print("Plotting FK analysis")
        self.canvas_fk.scatter3d(self.T, self.relpower, self.relpower, axes_index=0, clabel="Power [dB]")
        self.canvas_fk.scatter3d(self.T, self.abspower, self.relpower, axes_index=1, clabel="Power [dB]")
        self.canvas_fk.scatter3d(self.T, self.AZ, self.relpower, axes_index=2, clabel="Power [dB]")
        self.canvas_fk.scatter3d(self.T, self.Slowness, self.relpower, axes_index=3, clabel="Power [dB]")
        self.canvas_fk.set_ylabel(0, " Rel Power ")
        self.canvas_fk.set_ylabel(1, " Absolute Power ")
        self.canvas_fk.set_ylabel(2, " Back Azimuth ")
        self.canvas_fk.set_ylabel(3, " Slowness [s/km] ")
        self.canvas_fk.set_xlabel(3, " Time [s] ")
        ax = self.canvas_fk.get_axe(3)
        formatter = mdt.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=30)
        print("Finished FK analysis")
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)

    def on_click_matplotlib(self, event, canvas):
        output_path = os.path.join(ROOT_DIR, 'arrayanalysis', 'array_picks.txt"')

        if isinstance(canvas, MatplotlibCanvas):
            st = self.st.copy()
            wavenumber = array_analysis.array()
            # selection = self.inventory.select(station=self.stationLE.text(), channel=self.channelLE.text())
            selection = MseedUtil.filter_inventory_by_stream(self.st, self.inventory)
            x1, y1 = event.xdata, event.ydata
            if self.methodSB.currentText() == "FK" or self.methodSB.currentText() == "MTP.COHERENCE" \
                    or self.methodSB.currentText() == "CAPON":
                Z, Sxpow, Sypow, coord = wavenumber.FKCoherence(st, selection, x1,
                                                                self.fminFK_bind.value, self.fmaxFK_bind.value,
                                                                self.smaxFK_bind.value, self.timewindow_bind.value,
                                                                self.slow_grid_bind.value, self.methodSB.currentText())
            else:

                Z, Sxpow, Sypow, coord = wavenumber.run_music(st, selection, x1,
                                                                self.fminFK_bind.value, self.fmaxFK_bind.value,
                                                                self.smaxFK_bind.value, self.timewindow_bind.value,
                                                                self.slow_grid_bind.value, self.methodSB.currentText())

            backacimuth = wavenumber.azimuth2mathangle(np.arctan2(Sypow, Sxpow) * 180 / np.pi)
            slowness = np.abs(Sxpow, Sypow)
            if self.methodSB.currentText() == "FK" or self.methodSB.currentText() == "CAPON":
                clabel = "Power"
            elif self.methodSB.currentText() == "MTP.COHERENCE":
                clabel = "Magnitude Coherence"
            elif self.methodSB.currentText() == "MUSIC":
                clabel = "MUSIC Pseudospectrum"



            Sx = np.arange(-1 * self.smaxFK_bind.value, self.smaxFK_bind.value, self.slow_grid_bind.value)[np.newaxis]
            nx = len(Sx[0])
            x = y = np.linspace(-1 * self.smaxFK_bind.value, self.smaxFK_bind.value, nx)
            X, Y = np.meshgrid(x, y)
            self.canvas_slow_map.plot_contour(X, Y, Z, axes_index=0, clabel=clabel, cmap=plt.get_cmap("jet"))
            self.canvas_slow_map.set_xlabel(0, "Sx [s/km]")
            self.canvas_slow_map.set_ylabel(0, "Sy [s/km]")

            # Save in a dataframe the pick value
            x1 = UTCDateTime(mdt.num2date(x1))

            self.picks['Time'].append(x1.isoformat())
            self.picks['Phase'].append(self.phaseCB.currentText())
            self.picks['BackAzimuth'].append(backacimuth[0])
            self.picks['Slowness'].append(slowness[0])
            self.picks['Power'].append(np.max(Z))
            df = pd.DataFrame(self.picks)
            df.to_csv(output_path, index=False, header=True)

            # write the pick

            self.phaseLE.appendPlainText("{phase}"
                                                    " Time {time} Slowness: {slowness:.2f} Azimuth: {azimuth:.2f} Power: "
                                                    "{power:.2f} ".
                                                    format(phase=self.phaseCB.currentText(), time=x1.isoformat()[:-5],
                                                           slowness=slowness[0], azimuth=backacimuth[0],
                                                           power=np.max(Z)))


            # Call Stack and Plot###
            if st and self.showStackCB.isChecked():
                st2 = self.st.copy()

                # Align for the maximum power and give the data of the traces
                self.st_shift, self.trace_stack = wavenumber.stack_stream(st2, Sxpow, Sypow,
                                                                coord, stack_type=self.stackCB.currentText())
                self.pt = PlotToolsManager(" ")
                self.pt.multiple_shifts_array(self.trace_stack, self.st_shift)

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def process_fk(self, stream: Stream, inventory: Inventory, starttime: UTCDateTime, endtime: UTCDateTime):

        self.bpWidget.setCurrentIndex(1)
        self.st = stream
        print(self.st)
        self.inventory = inventory
        self.starttime = starttime
        self.endttime = endtime
        self.stream_frame = MatplotlibFrame(self.st, type='normal')
        self.stream_frame.show()

    def write_stack(self):
        if isinstance(self.trace_stack, Trace):
            root_path = os.path.dirname(os.path.abspath(__file__))
            if "darwin" == platform:
                dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
            else:
                dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                               pw.QFileDialog.DontUseNativeDialog)

            if dir_path:
                file = os.path.join(dir_path, self.trace_stack.id)
                self.trace_stack.write(file, format="MSEED")
                md = MessageDialog(self)
                md.set_info_message("Written stack trace at: "+dir_path)
        else:
            md = MessageDialog(self)
            md.set_info_message("Nothing to write")

    def __to_UTC(self, t1):

        return UTCDateTime(mdt.num2date(t1))

    def on_multiple_select(self, ax_index, xmin, xmax):

        self.t1 = self.__to_UTC(xmin)
        self.t2 = self.__to_UTC(xmax)

        idx1, val = find_nearest(self.T, xmin)
        idx2, val = find_nearest(self.T, xmax)
        T = self.T[idx1:idx2]

        self.canvas_fk.plot_date(T, self.relpower[idx1:idx2], 0, color="purple", clear_plot=False, fmt='.',
                                 markeredgecolor='black', markeredgewidth=0.5, alpha=0.75,
                                 linewidth=0.5, label="selected for Vespagram")

        self.canvas_fk.plot_date(T, self.abspower[idx1:idx2], 1, color="purple", clear_plot=False, fmt='.',
                                 markeredgecolor='black', markeredgewidth=0.5, alpha=0.75,
                                 linewidth=0.5, label="selected for Vespagram")

        self.canvas_fk.plot_date(T, self.AZ[idx1:idx2], 2, color="purple", clear_plot=False, fmt='.',
                                 markeredgecolor='black', markeredgewidth=0.5, alpha=0.75,
                                 linewidth=0.5, label="selected for Vespagram")

        self.canvas_fk.plot_date(T, self.Slowness[idx1:idx2], 3, color="purple", clear_plot=False, fmt='.',
                                 markeredgecolor='black', markeredgewidth=0.5, alpha=0.75,
                                 linewidth=0.5, label="selected for Vespagram")

        ax = self.canvas_fk.get_axe(3)
        formatter = mdt.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=30)
        ax.legend()
        self.runVespaBtn.setEnabled(True)

    def open_solutions(self):
        output_path = os.path.join(ROOT_DIR, 'arrayanalysis', 'dataframe.csv')
        try:
            # Determine the appropriate command based on the OS
            if platform.system() == 'Darwin':  # macOS
                command = ["open", output_path]
            elif platform.system() == 'Linux':  # Linux
                command = ["xdg-open", output_path]
            else:
                raise OSError("Unsupported operating system")

            # Execute the command
            subprocess.run(command, cwd=ROOT_DIR, check=True)
        except Exception as e:
            md = MessageDialog(self)
            md.set_error_message(f"Couldn't open solutions file: {str(e)}")

    ### New part back-projection

    def create_grid(self):
        area_coords = [self.minLonBP, self.maxLonBP, self.minLatBP, self.maxLatBP]
        bp = back_proj_organize(self, self.rootPathFormBP, self.datalessPathFormBP, area_coords, self.sxSB.value,
                                self.sxSB.value, self.depthSB.value)

        mapping = bp.create_dict()

        try:
            self.path_file = os.path.join(self.output_path_bindBP.value, "mapping.pkl")
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
                                  stack_process=self.methodBP.currentText())

        # plot_cum(power, mapping['area_coords'], self.cum_sumBP.value, self.st)
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
        open_url(self.url)
