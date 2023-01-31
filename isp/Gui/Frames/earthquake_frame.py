import shutil
from concurrent.futures.thread import ThreadPoolExecutor
import matplotlib.dates as mdt
from matplotlib.backend_bases import MouseButton
from obspy import UTCDateTime, Stream, Trace
from obspy.core.event import Origin
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.signal.trigger import coincidence_trigger
from isp.DataProcessing import DatalessManager, SeismogramDataAdvanced, ConvolveWaveletScipy
from isp.DataProcessing.NeuralNetwork import CNNPicker
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.DataProcessing.plot_tools_manager import PlotToolsManager
from isp.Exceptions import parse_excepts
from isp.Gui import pw, pqg, pyc, qt
from isp.Gui.Frames import BaseFrame, UiEarthquakeAnalysisFrame, Pagination, MessageDialog, EventInfoBox, \
    MatplotlibCanvas 
from isp.Gui.Frames.earthquake_frame_tabs import Earthquake3CFrame, EarthquakeLocationFrame
from isp.Gui.Frames.help_frame import HelpDoc
from isp.Gui.Frames.open_magnitudes_calc import MagnitudeCalc
from isp.Gui.Frames.earth_model_viewer import EarthModelViewer
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.uncertainity import UncertainityInfo
from isp.Gui.Frames.project_frame import Project
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.settings_dialog import SettingsDialog
from isp.Gui.Frames.search_catalog_frame import SearchCatalogViewer
from isp.Gui.Utils import map_polarity_from_pressed_key, ParallelWorkers
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, convert_qdatetime_utcdatetime, set_qdatetime, parallel_progress_run
from isp.Structures.structures import PickerStructure
from isp.Utils import MseedUtil, ObspyUtil, AsycTime
from isp.arrayanalysis import array_analysis
from isp.arrayanalysis.backprojection_tools import backproj
from isp.db.models import EventLocationModel
from isp.earthquakeAnalisysis import PickerManager, NllManager
import numpy as np
import os
import json
from isp.Utils.subprocess_utils import exc_cmd
from isp import ROOT_DIR, EVENTS_DETECTED, AUTOMATIC_PHASES
import matplotlib.pyplot as plt
from isp.earthquakeAnalisysis.stations_map import StationsMap
from isp.seismogramInspector.signal_processing_advanced import spectrumelement, sta_lta, envelope, Entropydetect, \
    correlate_maxlag, get_lags
from sys import platform

class EarthquakeAnalysisFrame(BaseFrame, UiEarthquakeAnalysisFrame):

    value_entropy_init = pyc.pyqtSignal(int)
    plot_progress = pyc.pyqtSignal()

    def __init__(self):
        super(EarthquakeAnalysisFrame, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Earthquake Location')
        self.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))

        #try:
        self.cnn = CNNPicker()
        #finally:
        #print("Neural Network cannot be loaded")
        self.zoom_diff = None
        self.phases = None
        self.travel_times = None
        self.cancelled = False
        self.aligned_checked = False
        self.aligned_picks = False
        self.aligned_fixed = False
        self.shift_times = None
        self.dist_all = []
        self.baz_all = []
        self.special_selection = []
        self.lines = []
        self.progressbar = pw.QProgressDialog(self)
        self.progressbar.setWindowTitle('Earthquake Location')
        self.progressbar.setLabelText(" Computing Auto-Picking ")
        self.progressbar.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.path_phases = os.path.join(AUTOMATIC_PHASES, "phases_autodetected.txt")
        self.path_detection = os.path.join(EVENTS_DETECTED, "event_autodetects.txt")
        self.progressbar.close()
        self.settings_dialog = SettingsDialog(self)
        self.inventory = {}
        self.files = []
        self.files_path = []
        self.events_times = []
        self.check_start_time = None
        self.check_end_time = None
        self.total_items = 0
        self.items_per_page = 1
        # dict to keep track of picks-> dict(key: PickerStructure) as key we use the drawn line.
        self.picked_at = {}
        self.__dataless_manager = None
        self.__metadata_manager = None
        self.st = None
        self.cf = None
        self.chop = {'Body waves': {}, 'Surf Waves': {}, 'Coda': {}, 'Noise': {}}
        self.color = {'Body waves': 'orangered', 'Surf Waves': 'blue', 'Coda': 'purple', 'Noise': 'green'}
        self.dataless_not_found = set()  # a set of mseed files that the dataless couldn't find.
        self.pagination = Pagination(self.pagination_widget, self.total_items, self.items_per_page)
        self.pagination.set_total_items(0)
        self.pagination.bind_onPage_changed(self.onChange_page)
        self.pagination.bind_onItemPerPageChange_callback(self.onChange_items_per_page)

        self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=self.items_per_page, constrained_layout=False)
        self.canvas.set_xlabel(0, "Time (s)")
        # self.canvas.figure.subplots_adjust(left=0.065, bottom=0.1440, right=0.9, top=0.990, wspace=0.2, hspace=0.0)
        self.canvas.figure.tight_layout()

        self.canvas.on_double_click(self.on_click_matplotlib)
        self.canvas.on_pick(self.on_pick)
        self.canvas.register_on_select(self.on_select, rectprops=dict(alpha=0.2, facecolor='red'))
        # That's how you can register a right click selector
        self.canvas.register_on_select(self.on_multiple_select,
                                       button=MouseButton.RIGHT, sharex=True, rectprops=dict(alpha=0.2, facecolor='blue'))

        self.canvas.mpl_connect('key_press_event', self.key_pressed)
        self.canvas.mpl_connect('axes_enter_event', self.enter_axes)
        self.event_info = EventInfoBox(self.eventInfoWidget, self.canvas)
        self.event_info.register_plot_arrivals_click(self.on_click_plot_arrivals)
        self.earthquake_3c_frame = Earthquake3CFrame(self.parentWidget3C)
        self.earthquake_location_frame = EarthquakeLocationFrame(self.parentWidgetLocation)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)

        # Bind buttons

        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_file(self.metadata_path_bind))
        self.selectDirBtn.clicked.connect(lambda: self.load_project())
        self.updateBtn.clicked.connect(self.plot_seismogram)
        self.stations_infoBtn.clicked.connect(self.stationsInfo)
        self.rotateBtn.clicked.connect(self.rotate)
        self.mapBtn.clicked.connect(self.plot_map_stations)
        self.crossBtn.clicked.connect(self.cross)
        self.macroBtn.clicked.connect(self.open_parameters_settings)
        self.__metadata_manager = MetadataManager(self.metadata_path_bind.value)
        self.actionSet_Parameters.triggered.connect(lambda: self.open_parameters_settings())
        self.actionOpen_Earth_Model_Viewer.triggered.connect(lambda: self.open_earth_model_viewer())
        self.actionWrite_Current_Page_2.triggered.connect(self.write_files_page)
        self.actionArray_Anlysis.triggered.connect(self.open_array_analysis)
        self.actionMoment_Tensor_Inversion.triggered.connect(self.open_moment_tensor)
        self.actionTime_Frequency_Analysis.triggered.connect(self.time_frequency_analysis)
        self.actionOpen_Magnitudes_Calculator.triggered.connect(self.open_magnitudes_calculator)
        self.actionSTA_LTA.triggered.connect(self.run_STA_LTA)
        self.actionCWT_CF.triggered.connect(self.cwt_cf)
        self.actionEnvelope.triggered.connect(self.envelope)
        self.actionReceiver_Functions.triggered.connect(self.open_receiver_functions)
        self.actionRun_picker.triggered.connect(self._picker_thread)
        self.actionRun_Event_Detector.triggered.connect(self.detect_events)
        self.actionOpen_Settings.triggered.connect(lambda: self.settings_dialog.show())
        self.actionSearch_in_Catalog.triggered.connect(lambda: self.open_catalog_viewer())
        self.actionStack.triggered.connect(lambda: self.stack_all_seismograms())
        #self.actionSpectral_Entropy.triggered.connect(lambda : self.spectral_entropy())
        self.actionSpectral_Entropy.triggered.connect(lambda: self.spectral_entropy_progress())
        self.actionRemove_all_selections.triggered.connect(lambda : self.clean_all_chop())
        self.actionClean_selection.triggered.connect(lambda : self.clean_chop_at_page())
        self.actionClean_Events_Detected.triggered.connect(lambda : self.clean_events_detected())
        self.actionPlot_All_Seismograms.triggered.connect(lambda : self.plot_all_seismograms())
        self.actionOpen_Help.triggered.connect(lambda: self.open_help())
        self.actionOnly_a_folder.triggered.connect(lambda: self.availability())
        self.actionAll_tree.triggered.connect(lambda: self.availability_all_tree())
        self.actionOpen_picksnew.triggered.connect(lambda: self.open_solutions())
        self.actionRemove_picks.triggered.connect(lambda: self.remove_picks())
        self.actionNew_location.triggered.connect(lambda: self.start_location())
        self.actionRun_autoloc.triggered.connect(lambda: self.picker_all())
        self.actionFrom_Phase_Pick.triggered.connect(lambda: self.alaign_picks())
        self.actionUsing_MCCC.triggered.connect(lambda: self.alaign_mccc())
        self.actionPicks_from_file.triggered.connect(lambda: self.import_pick_from_file())
        self.actionNew_Project.triggered.connect(lambda: self.new_project())
        self.newProjectBtn.clicked.connect(lambda: self.new_project())
        self.actionLoad_Project.triggered.connect(lambda: self.load_project())
        self.actionPlot_Record_Section.triggered.connect(lambda: self.plot_prs())
        self.pm = PickerManager()  # start PickerManager to save pick location to csv file.

        # Parameters settings

        self.parameters = ParametersSettings()

        # Uncertainity pick
        self.uncertainities = UncertainityInfo()

        # Project

        self.project_dialog = Project()

        # Earth Model Viewer

        self.earthmodel = EarthModelViewer()

        # help Documentation

        self.help = HelpDoc()

        # catalog viewer

        self.catalog = SearchCatalogViewer()

        # shortcuts
        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+U'), self)
        self.shortcut_open.activated.connect(self.open_solutions)

        # shortcuts
        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+Y'), self)
        self.shortcut_open.activated.connect(self.open_events)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+L'), self)
        self.shortcut_open.activated.connect(self.open_parameters_settings)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+I'),self)
        self.shortcut_open.activated.connect(self.clean_chop_at_page)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+C'), self)
        self.shortcut_open.activated.connect(self.clean_events_detected)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+M'), self)
        self.shortcut_open.activated.connect(self.open_magnitudes_calculator)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+P'),self)
        self.shortcut_open.activated.connect(self.comboBox_phases.showPopup)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('P'), self)
        self.shortcut_open.activated.connect(self.plot_prs)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+K'), self)
        self.shortcut_open.activated.connect(self.save_cf)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+N'), self)
        self.shortcut_open.activated.connect(self.plot_all_seismograms)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+B'), self)
        self.shortcut_open.activated.connect(self.stack_all_seismograms)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+J'), self)
        self.shortcut_open.activated.connect(self.clean_all_chop)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('W'), self)
        self.shortcut_open.activated.connect(self.plot_seismogram)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+W'), self)
        self.shortcut_open.activated.connect(self.get_now_files)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+R'), self)
        self.shortcut_open.activated.connect(self.detect_events)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+T'), self)
        self.shortcut_open.activated.connect(self._picker_thread)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+F'), self)
        self.shortcut_open.activated.connect(self.picker_all)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('U'), self)
        self.shortcut_open.activated.connect(self.open_uncertainity_settings)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('I'), self)
        self.shortcut_open.activated.connect(self.multi_cursor_on)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('O'), self)
        self.shortcut_open.activated.connect(self.multi_cursor_off)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('R'), self)
        self.shortcut_open.activated.connect(self.reload_current_project)

    # def on_xlims_change(self, event_ax):
    #
    #     self.zoom = event_ax.get_xlim()
    #     t1 = UTCDateTime(mdt.num2date(self.zoom[0]))
    #     t2 = UTCDateTime(mdt.num2date(self.zoom[1]))
    #     self.zoom_diff = (t2-t1)

    def cancelled_callback(self):
        self.cancelled = True

    def multi_cursor_on(self):
        self.canvas.activate_multi_cursor()


    def multi_cursor_off(self):
        self.canvas.deactivate_multi_cursor()


    def open_help(self):
        self.help.show()

    def open_parameters_settings(self):
        self.parameters.show()

    def open_uncertainity_settings(self):
        self.uncertainities.show()

    def new_project(self):
        self.loaded_project = False
        self.setEnabled(False)
        self.project_dialog.exec()
        self.setEnabled(True)
        self.project = self.project_dialog.project

        self.get_now_files()
        # now we can access to #self.project_dialog.project

    def load_project(self):
        self.loaded_project = True
        selected = pw.QFileDialog.getOpenFileName(self, "Select Project", ROOT_DIR)

        md = MessageDialog(self)

        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            try:
                self.current_project_file = selected[0]
                self.project = MseedUtil.load_project(file = selected[0])
                project_name = os.path.basename(selected[0])
                md.set_info_message("Project {} loaded  ".format(project_name))
            except:
                md.set_error_message("Project couldn't be loaded ")
        else:
            md.set_error_message("Project couldn't be loaded ")
        self.get_now_files()
        # now we can access to #self.project_dialog.project

    def reload_current_project(self):

        md = MessageDialog(self)
        if self.loaded_project:
            if isinstance(self.current_project_file, str) and os.path.isfile(self.current_project_file):
                try:
                    self.project = MseedUtil.load_project(file = self.current_project_file)
                    project_name = os.path.basename(self.current_project_file)
                    md.set_info_message("Project {} reloaded  ".format(project_name))
                except:
                    md.set_error_message("Project couldn't be reloaded ")
            else:
                md.set_error_message("Project couldn't be reloaded ")
            self.get_now_files()

        else:
            try:
                if isinstance(self.project, dict):
                    self.get_now_files()
                    md.set_info_message("New Project reloaded")
            except:
                md.set_error_message("Project couldn't be reloaded ")


    def open_earth_model_viewer(self):
        self.earthmodel.show()

    def open_catalog_viewer(self):
        self.catalog.show()
    #@property
    #def event_info(self) -> EventInfoBox:
    #    return self.__event_info
    @pyc.Slot()
    def _increase_progress(self):
        self.progressbar.setValue(self.progressbar.value() + 1)

    def _process_station(self, station, index):
        st2 = self.st.select(station=station)
        try:
            maxstart = np.max([tr.stats.starttime for tr in st2])
            minend = np.min([tr.stats.endtime for tr in st2])
            st2.trim(maxstart, minend)
            self.cnn.setup_stream(st2)  # set stream to use in prediction.
            self.cnn.predict()
            arrivals = self.cnn.get_arrivals()
            for k, times in arrivals.items():
                for t in times:
                    if k == "p":
                        line = self.canvas.draw_arrow(t.matplotlib_date, index + 2,
                                               "P", color="blue", linestyles='--', picker=True)
                        self.lines.append(line)

                        # TODO needs to be reviewed
                        #with open(self.path_phases, "a+") as f:
                        #    f.write(station + " " + k.upper() + " " + t.strftime(format="%Y-%m-%dT%H:%M:%S.%f") + "\n")

                        self.picked_at[str(line)] = PickerStructure(t, st2[2].stats.station, t.matplotlib_date,
                            0.2, 0, "blue", "P", self.get_file_at_index(index + 2))

                        self.pm.add_data(t, 0.2, 0, st2[2].stats.station, "P", Component=st2[2].stats.channel,
                                         First_Motion="?")
                        self.pm.save()

                    if k == "s":

                        line1s = self.canvas.draw_arrow(t.matplotlib_date, index + 0,
                                               "S", color="purple", linestyles='--', picker=True)
                        line2s = self.canvas.draw_arrow(t.matplotlib_date, index + 1,
                                               "S", color="purple", linestyles='--', picker=True)

                        self.lines.append(line1s)
                        self.lines.append(line2s)

                        # TODO needs to be reviewed
                        #with open(self.path_phases, "a+") as f:
                        #    f.write(station+" "+k.upper()+" "+t.strftime(format="%Y-%m-%dT%H:%M:%S.%f") + "\n")

                        self.picked_at[str(line1s)] = PickerStructure(t, st2[1].stats.station, t.matplotlib_date,
                             0.2, 0, "blue", "S", self.get_file_at_index(index + 0))

                        self.picked_at[str(line2s)] = PickerStructure(t, st2[1].stats.station, t.matplotlib_date,
                             0.2, 0, "blue", "S", self.get_file_at_index(index + 1))

                        self.pm.add_data(t, 0.2, 0, st2[1].stats.station, "S", Component=st2[1].stats.channel,
                                         First_Motion="?")
                        self.pm.save()

        except ValueError as e:
            # TODO: summarize errors and show eventually
            # md = MessageDialog(self)
            # md.set_info_message("Prediction failed for station {}\n{}".format(station,e))
            pass
        pyc.QMetaObject.invokeMethod(self, '_increase_progress', qt.AutoConnection)

    def _run_picker(self):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # Creates a new file
        #with open(self.path_phases, 'w') as fp:
        #    pass
        if self.st:
            stations = ObspyUtil.get_stations_from_stream(self.st)
            N = len(stations)
            pyc.QMetaObject.invokeMethod(self.progressbar, 'setMaximum', qt.AutoConnection, pyc.Q_ARG(int, N))
            pyc.QMetaObject.invokeMethod(self.progressbar, 'setValue', qt.AutoConnection, pyc.Q_ARG(int, 0))
            base_indexes = [*map(lambda x: 3 * x, [*range(N)])]
            # TODO OPEN MP FOR MULTIPLE STATIONS
            for station, base_index in zip(stations, base_indexes):
                self._process_station(station, base_index)
        else:
            pyc.QMetaObject.invokeMethod(self.progressbar, 'reset', qt.AutoConnection)

    def _picker_thread(self):
        self.progressbar.reset()
        self.progressbar.setLabelText(" Computing Auto-Picking ")
        with ThreadPoolExecutor(1) as executor:
            f = executor.submit(self._run_picker)
            self.progressbar.exec()
            f.cancel()


    def picker_all(self):
        # delta is the trim of the data
        IND_OUTPUT_PATH = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'loc', 'last.hyp')
        LOC_OUTPUT_PATH = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'all_locations')
        params = self.settings_dialog.getParameters()
        delta = params["window pick"]
        transform =params["transform"]
        pick_output_path = PickerManager.get_default_output_path()
        self.nll_manager = NllManager(pick_output_path, self.metadata_path_bind.value)
        st_detect_all = self.st.copy()

        self.detect_events()
        events_path = self.path_detection
        picks_path = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'obs', 'output.txt')


        with open(events_path, 'rb') as handle:
            events = json.load(handle)

        for k in range(len(events)):
            start = UTCDateTime(events[k]) - delta
            end = UTCDateTime(events[k]) + delta
            self.st.trim(starttime = start , endtime = end)
            # clean previous picks
            os.remove(picks_path)
            # Picking on detected event
            self._run_picker()
            # Locate based on prevous detected and picked event
            std_out = self.nll_manager.run_nlloc(0, 0, 0, transform = transform)

            # restore it
            self.st = st_detect_all
            # copy output to /location_output/all_locations
            LOC_OUTPUT_PATH_TEMP  = os.path.join(LOC_OUTPUT_PATH,events[k]+".hyp")
            shutil.copyfile(IND_OUTPUT_PATH, LOC_OUTPUT_PATH_TEMP)

            #md = MessageDialog(self)
            #md.set_info_message("Location complete. Check details for earthquake located at "+events[k]
            #                    , std_out)
        md = MessageDialog(self)
        md.set_info_message("Location complete. Check details for earthquake located in "+LOC_OUTPUT_PATH)


    def alaign_picks(self):
            self.aligned_picks = True
            phase = self.comboBox_phases.currentText()
            self.aligned_checked = True
            self.pick_times = MseedUtil.get_NLL_phase_picks(phase)
            self.plot_seismogram()

    def import_pick_from_file(self):

        selected = pw.QFileDialog.getOpenFileName(self, "Select picking file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            self.pick_times_imported = MseedUtil.get_NLL_phase_picks2(input_file = selected[0])
            stations_info = self.__stations_info_list()
            for count, station in enumerate(stations_info):
                id = station[1] + "." + station[3]
                if id in self.pick_times_imported:
                    pick = self.pick_times_imported[id] #in UTCDatetime needs to be in samples
                    for j in range(len(pick)):
                        pick_value = pick[j][1].matplotlib_date
                        # build the label properly

                        if pick[j][3] == "U":
                            label = pick[j][0] + " +"
                        elif pick[j][3] == "D":
                            label = pick[j][0] + " -"
                        else:
                            label = pick[j][0] + " ?"

                        line = self.canvas.draw_arrow(pick_value, count, arrow_label=label, amplitude=pick[j][7],
                                                      color="green", picker=True)
                        self.lines.append(line)
                        self.picked_at[str(line)] = PickerStructure(pick[j][1], id.split(".")[0], pick_value, pick[j][4],
                             pick[j][7], "green", label, self.get_file_at_index(count))

                        self.pm.add_data(pick_value, pick[j][4], pick[j][7], id.split(".")[0], pick[j][0],
                                         First_Motion = pick[j][3], Component = pick[j][2])
                        self.pm.save()

    def __stations_info_list(self):
        files_at_page = self.get_files_at_page()
        sd = []

        for file in files_at_page:
            st = SeismogramDataAdvanced(file)

            station = [st.stats.Network, st.stats.Station, st.stats.Location, st.stats.Channel, st.stats.StartTime,
                       st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]

            sd.append(station)

        return sd

    @property
    def dataless_manager(self):
        if not self.__dataless_manager:
            self.__dataless_manager = DatalessManager(self.dataless_path_bind.value)
        return self.__dataless_manager


    def message_dataless_not_found(self):
        if len(self.dataless_not_found) > 1:
            md = MessageDialog(self)
            md.set_info_message("Metadata not found.")
        else:
            for file in self.dataless_not_found:
                md = MessageDialog(self)
                md.set_info_message("Metadata for {} not found.".format(file))

        self.dataless_not_found.clear()

    def get_files_at_page(self):
        n_0 = (self.pagination.current_page - 1) * self.pagination.items_per_page
        n_f = n_0 + self.pagination.items_per_page
        return self.files[n_0:n_f]

    def get_file_at_index(self, index):
        files_at_page = self.get_files_at_page()
        return files_at_page[index]

    def onChange_page(self, page):
        self.plot_seismogram()

    def onChange_items_per_page(self, items_per_page):
        self.items_per_page = items_per_page
        #self.plot_seismogram()

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        pass
        #self.get_now_files()
        #self.set_pagination_files(self.files_path)

        # self.plot_seismogram()

    def set_pagination_files(self, files_path):
        self.files = files_path
        self.total_items = len(self.files)
        self.pagination.set_total_items(self.total_items)


    def get_now_files(self):

        selection = [self.netForm.text(), self.stationForm.text(), self.channelForm.text()]
        _, self.files_path = MseedUtil.filter_project_keys(self.project, net=selection[0], station=selection[1], channel=selection[2])

        if self.trimCB.isChecked():
            start = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
            end = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
            diff = end-start
            if diff > 0:
                self.files_path = MseedUtil.filter_time(list_files=self.files_path, starttime=start, endtime=end)
        else:
            self.files_path = MseedUtil.filter_time(list_files=self.files_path)


        self.set_pagination_files(self.files_path)




    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    @AsycTime.run_async()
    def onChange_metadata_path(self, value):

        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
        except:
            raise FileNotFoundError("The metadata is not valid")



    def subprocess_feedback(self, err_msg: str, set_default_complete=True):
        """
        This method is used as a subprocess feedback. It runs when a raise expect is detected.

        :param err_msg: The error message from the except.
        :param set_default_complete: If True it will set a completed successfully message. Otherwise nothing will
            be displayed.
        :return:
        """
        if err_msg:
            md = MessageDialog(self)
            if "Error code" in err_msg:
                md.set_error_message("Click in show details detail for more info.", err_msg)
            else:
                md.set_warning_message("Click in show details for more info.", err_msg)
        else:
            if set_default_complete:
                md = MessageDialog(self)
                md.set_info_message("Loaded Metadata, please check your terminal for further details")

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


    def alaign_mccc(self):
        if self.st:
            bp = backproj()
            self.st, self.shift_times = bp.multichanel(self.st, resample = True)
            self.plot_seismogram()



    def sort_by_distance_traces(self, trace):

        st_stats = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, trace)
        if st_stats:

            dist, _, _ = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                          self.event_info.longitude)
            self.dist_all.append(dist/1000)

            return dist
        else:
            self.dataless_not_found.add(trace)
            print("No Metadata found for {} file.".format(trace))
            return 0.

    def sort_by_baz_traces(self, trace):
        st_stats = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, trace)

        if st_stats:

            _, _, az_from_epi = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                                 self.event_info.longitude)
            self.baz_all.append(az_from_epi)
            return az_from_epi
        else:

            self.dataless_not_found.add(trace)
            print("No Metadata found for {} file.".format(trace))
            return 0.


    def sort_by_distance_advance(self, file):

         self.dist_all = []
         st_stats = self.__metadata_manager.extract_coordinates(self.inventory, file)
         if st_stats:

             dist, _, _ = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                           self.event_info.longitude)
             self.dist_all.append(dist/1000)
             return dist
         else:
             self.dataless_not_found.add(file)
             print("No Metadata found for {} file.".format(file))
             self.dist_all.append(-100)
             return 0.

    def sort_by_baz_advance(self, file):

         self.baz_all = []
         st_stats = self.__metadata_manager.extract_coordinates(self.inventory, file)

         if st_stats:

             _, _, az_from_epi = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                          self.event_info.longitude)
             self.baz_all.append(az_from_epi)
             return az_from_epi
         else:

             self.dataless_not_found.add(file)
             print("No Metadata found for {} file.".format(file))
             return 0.

    def plot_seismogram(self):
        self.workers = ParallelWorkers(os.cpu_count())
        # Here we can disabled thing or make additional staff
        self.workers.job(self.__plot_seismogram)

        self.decimator = [None, False]
        if self.st:
            del self.st

        self.canvas.clear()

        #self.nums_clicks = 0

        if self.trimCB.isChecked() and self.check_start_time != None and self.check_end_time != None:
            if self.check_start_time != convert_qdatetime_utcdatetime(self.dateTimeEdit_1) and \
                    self.check_end_time != convert_qdatetime_utcdatetime(self.dateTimeEdit_2):
                self.get_now_files()

        if self.sortCB.isChecked():
            if self.comboBox_sort.currentText() == "Distance":
                self.files_path.sort(key=self.sort_by_distance_advance)
                self.actionPlot_Record_Section.setEnabled(True)
                self.message_dataless_not_found()

            elif self.comboBox_sort.currentText() == "Back Azimuth":
                self.files_path.sort(key=self.sort_by_baz_advance)
                self.message_dataless_not_found()


        if len(self.special_selection)> 0:
            self.set_pagination_files(self.special_selection)
            self.files_at_page = self.get_files_at_page()
        else:
            self.set_pagination_files(self.files_path)
            self.files_at_page = self.get_files_at_page()

        ##
        self.start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        self.end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        self.check_start_time = self.start_time
        self.check_end_time = self.end_time
        ##
        self.diff = self.end_time - self.start_time
        if len(self.canvas.axes) != len(self.files_at_page) or self.autorefreshCB.isChecked():
            self.canvas.set_new_subplot(nrows=len(self.files_at_page), ncols=1)
        self.last_index = 0
        self.min_starttime = []
        self.max_endtime = []

        tuple_files = [(i, file) for i, file in enumerate(self.files_at_page)]

        self.all_traces = [None for i in range(len(self.files_at_page))]

        self.parameters_list = self.parameters.getParameters()
        self.min_starttime = [None for i in range(len(self.files_at_page))]
        self.max_endtime = [None for i in range(len(self.files_at_page))]

        prog_dialog = pw.QProgressDialog()
        prog_dialog.setLabelText("Process and Plot")
        prog_dialog.setValue(0)
        prog_dialog.setRange(0, 0)

        def prog_callback():
            pyc.QMetaObject.invokeMethod(prog_dialog, "accept")

        self.plot_progress.connect(prog_callback)
        self.workers.start(tuple_files)

        prog_dialog.exec()
        for tuple_ind in tuple_files:
            self.redraw_event_times(tuple_ind[0])
            self.redraw_pickers(tuple_ind[1], tuple_ind[0])

        self.st = Stream(traces=self.all_traces)

        self.shift_times = None

        try:
            if self.min_starttime and self.max_endtime is not None:
                auto_start = min(self.min_starttime)
                auto_end = max(self.max_endtime)
                self.auto_start = auto_start
                self.auto_end = auto_end

            ax = self.canvas.get_axe(len(self.files_at_page)-1)
            # ax.callbacks.connect('xlim_changed', self.on_xlims_change)
            if self.trimCB.isChecked() and self.aligned_picks == False:

                ax.set_xlim(self.start_time.matplotlib_date, self.end_time.matplotlib_date)
            else:

                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))

            self.canvas.set_xlabel(len(self.files_at_page)-1, "Date")

        except:
            pass
        self.special_selection = []
        self.aligned_checked = False
        self.aligned_picks = False


    def __plot_seismogram(self, tuple_files):

        index = tuple_files[0]
        file_path = tuple_files[1]

        sd = SeismogramDataAdvanced(file_path)

        if self.trimCB.isChecked() and self.diff >= 0 and self.fastCB.isChecked():

            self.decimator = sd.resample_check(start_time=self.start_time, end_time=self.end_time)

        elif self.trimCB.isChecked() == False and self.fastCB.isChecked() == True:

            self.decimator = sd.resample_check()

        if self.decimator[1]:
            self.parameters_list.insert(0, ['resample', self.decimator[0], True])

        if self.trimCB.isChecked() and self.diff >= 0:

            tr = sd.get_waveform_advanced(self.parameters_list, self.inventory,
                                          filter_error_callback=self.filter_error_message,
                                          start_time=self.start_time, end_time=self.end_time, trace_number=index)
        else:

            tr = sd.get_waveform_advanced(self.parameters_list, self.inventory,
                                               filter_error_callback=self.filter_error_message, trace_number=index)
        if len(tr) > 0:

            if self.aligned_checked:
                try:
                    pick_reference = self.pick_times[tr.stats.station+"."+tr.stats.channel]
                    shift_time = pick_reference[1] - tr.stats.starttime
                    tr.stats.starttime = UTCDateTime("2000-01-01T00:00:00") - shift_time
                except:
                    tr.stats.starttime = UTCDateTime("2000-01-01T00:00:00")


            if self.actionFrom_StartT.isChecked():
                tr.stats.starttime = UTCDateTime("2000-01-01T00:00:00")

            if self.shift_times is not None:
                tr.stats.starttime = tr.stats.starttime + self.shift_times[index][0]

            t = tr.times("matplotlib")
            s = tr.data

            self.canvas.plot_date(t, s, index, color="black", fmt= '-', linewidth=0.5)

            ax = self.canvas.get_axe(index)
            try:
                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(top=False)
                ax.tick_params(labeltop=False)
            except:
                pass
            ax.set_ylim(np.min(s), np.max(s))
            #
            if index != (self.pagination.items_per_page-1):
                try:
                    ax.tick_params(bottom=False)
                except:
                    pass

            st_stats = ObspyUtil.get_stats(file_path)
            self.redraw_chop(tr, s, index)

            if self.decimator[1]:
                warning = "Decimated to " + str(self.decimator[0])+"  Hz"
                self.canvas.set_warning_label(index, warning)

            if st_stats and self.sortCB.isChecked() == False:
                info = "{}.{}.{}".format(st_stats.Network, st_stats.Station, st_stats.Channel)
                self.canvas.set_plot_label(index, info)

            elif st_stats and self.sortCB.isChecked() and self.comboBox_sort.currentText() == "Distance":

                dist = self.sort_by_distance_advance(file_path)
                dist = "{:.1f}".format(dist/1000.0)
                info = "{}.{}.{} Distance {} km".format(st_stats.Network, st_stats.Station, st_stats.Channel,
                                                     str(dist))
                self.canvas.set_plot_label(index, info)

            elif st_stats and self.sortCB.isChecked() and self.comboBox_sort.currentText() == "Back Azimuth":

                back = self.sort_by_baz_advance(file_path)
                back = "{:.1f}".format(back)
                info = "{}.{}.{} Back Azimuth {}".format(st_stats.Network, st_stats.Station, st_stats.Channel,
                                                         str(back))
                self.canvas.set_plot_label(index, info)

            try:
                self.min_starttime[index] = min(t)
                self.max_endtime[index] = max(t)
            except:
                print("Empty traces")

            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)

            self.all_traces[index] = tr
            self.plot_progress.emit()


    # Rotate to GAC #
    def rotate(self):

        if self.st:
            self.canvas.clear()
            all_traces_rotated = []
            stations = ObspyUtil.get_stations_from_stream(self.st)
            start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
            end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
            for k in range(len(stations)):

                if self.angCB.isChecked():

                    # Process the data
                    self.plot_seismogram()
                    st1 = self.st.copy()
                    st2 = st1.select(station=stations[k])
                    maxstart = np.max([tr.stats.starttime for tr in st2])
                    minend = np.min([tr.stats.endtime for tr in st2])
                    st2.trim(maxstart, minend)
                    bazim = self.rot_ang.value()

                else:

                    st1 = self.st.copy()
                    st2 = st1.select(station=stations[k])
                    maxstart = np.max([tr.stats.starttime for tr in st2])
                    minend = np.min([tr.stats.endtime for tr in st2])
                    st2.trim(maxstart, minend)
                    tr = st2[0]
                    coordinates = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, tr)
                    [azim, bazim, inci] = ObspyUtil.coords2azbazinc(coordinates.Latitude,coordinates.Longitude,
                    coordinates.Elevation,self.event_info.latitude, self.event_info.longitude, self.event_info.event_depth)
                    print(bazim)
                st2.rotate(method='NE->RT', back_azimuth=bazim)

                print("The GAC Rotation is not posible for",stations[k])
                for tr in st2:
                    all_traces_rotated.append(tr)

            self.st = Stream(traces=all_traces_rotated)
            # plot
            files_at_page = self.get_files_at_page()
            for index, file_path in enumerate(files_at_page):
                tr = all_traces_rotated[index]
                t = tr.times("matplotlib")
                s = tr.data
                if tr.stats.channel[2] == "T" or tr.stats.channel[2] == "R":
                    st_stats = ObspyUtil.get_stats_from_trace(tr)
                    id_new = st_stats['net'] + "." + st_stats['station'] + "." + st_stats['location'] + "." \
                         + st_stats['channel']
                    #change chop_dictionary
                    if tr.stats.channel[2] == "T":
                        id_old = st_stats['net'] + "." + st_stats['station'] + "." + st_stats['location'] + "." \
                         + st_stats['channel'][0:2]+"E"
                        try:
                            for key , value in self.chop.items():
                                if id_new in self.chop[key]:
                                    self.chop[key][id_new] = self.chop[key].pop(id_old)
                        except:
                            pass

                    if tr.stats.channel[2] == "R":
                        id_old = st_stats['net'] + "." + st_stats['station'] + "." + st_stats['location'] + "." \
                         + st_stats['channel'][0:2]+"N"
                        try:
                            for key , value in self.chop.items():
                                if id_new in self.chop[key]:
                                    self.chop[key][id_new] = self.chop[key].pop(id_old)
                        except:
                            pass

                    self.canvas.plot_date(t, s, index, color="steelblue", fmt='-', linewidth=0.5)
                    info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
                    self.canvas.set_plot_label(index, info)
                    self.redraw_pickers(file_path, index)
                # redraw_chop = 1 redraw chopped data, 2 update in case data chopped is modified
                    self.redraw_chop(tr, s, index)
                else:
                    st_stats = ObspyUtil.get_stats_from_trace(tr)
                    self.canvas.plot_date(t, s, index, color="black", fmt='-', linewidth=0.5)
                    info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
                    self.canvas.set_plot_label(index, info)
                    self.redraw_pickers(file_path, index)
                    # redraw_chop = 1 redraw chopped data, 2 update in case data chopped is midified
                    self.redraw_chop(tr, s, index)
                last_index =  index

            ax = self.canvas.get_axe(last_index)
            try:
                if self.trimCB.isChecked():
                    ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
                else:
                    ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
                formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
                ax.xaxis.set_major_formatter(formatter)
                self.canvas.set_xlabel(last_index, "Date")
            except:
                pass

    def run_STA_LTA(self):
        self.cf = []
        cfs = []
        files_at_page = self.get_files_at_page()
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        params = self.settings_dialog.getParameters()
        STA = params["STA"]
        LTA = params["LTA"]
        for index, file_path in enumerate(files_at_page):
            tr = self.st[index]

            if STA < LTA:
                cf = sta_lta(tr.data, tr.stats.sampling_rate, STA = STA, LTA = LTA)
            else:
                cf = sta_lta(tr.data, tr.stats.sampling_rate)

            st_stats = ObspyUtil.get_stats_from_trace(tr)
            # Normalize
            #cf =cf/max(cf)
            # forward to be centered the peak
            t = tr.times("matplotlib")
            self.canvas.plot_date(t, tr.data, index, color="black", fmt='-', linewidth=0.5)
            info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
            self.canvas.set_plot_label(index, info)
            self.redraw_pickers(file_path, index)
            self.redraw_chop(tr, tr.data, index)

            t = t[0:len(cf)]

            ax = self.canvas.get_axe(index)
            ax2 = ax.twinx()
            ax2.plot(t, cf, color="grey", linewidth=0.5)
            last_index = index
            tr_cf = tr.copy()
            tr_cf.data = cf
            tr_cf.times = t
            cfs.append(tr_cf)
        ax = self.canvas.get_axe(last_index)
        try:
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(last_index, "Date")
        except:
            pass

        self.cf = Stream(traces=cfs)



    def cwt_cf(self):
        self.cf = []
        cfs = []
        files_at_page = self.get_files_at_page()
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        params = self.settings_dialog.getParameters()
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        diff = end_time - start_time
        cycles = params["Num Cycles"]
        fmin = params["Fmin"]
        fmax = params["Fmax"]

        for index, file_path in enumerate(files_at_page):
            tr = self.st[index]
            st_stats = ObspyUtil.get_stats_from_trace(tr)
            cw = ConvolveWaveletScipy(tr)
            if self.trimCB.isChecked() and diff >= 0:

                if fmin < fmax and cycles > 5:
                    tt = int(tr.stats.sampling_rate / fmin)
                    cw.setup_wavelet(start_time, end_time, wmin=cycles, wmax=cycles, tt=tt, fmin=fmin, fmax=fmax, nf=40,
                                 use_rfft=False, decimate=False)

                else:
                    cw.setup_wavelet(start_time, end_time,wmin=6, wmax=6, tt=10, fmin=0.2, fmax=10, nf=40, use_rfft=False,
                                     decimate=False)

            else:
                if fmin < fmax and cycles > 5:
                   tt = int(tr.stats.sampling_rate / fmin)
                   cw.setup_wavelet(wmin=cycles, wmax=cycles, tt=tt, fmin=fmin, fmax=fmax, nf=40,
                                     use_rfft=False, decimate=False)
                else:
                   cw.setup_wavelet(wmin=6, wmax=6, tt=10, fmin=0.2, fmax=10, nf=40, use_rfft=False, decimate=False)


            delay = cw.get_time_delay()
            start = tr.stats.starttime + delay
            #f = np.logspace(np.log10(fmin), np.log10(fmax))
            #k = cycles / (2 * np.pi * f) #one standar deviation
            #delay = np.mean(k)

            tr.stats.starttime=start
            t = tr.times("matplotlib")

            self.canvas.plot_date(t, tr.data, index, color="black", fmt='-', linewidth=0.5)
            info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
            self.canvas.set_plot_label(index, info)
            self.redraw_pickers(file_path, index)
            self.redraw_chop(tr, tr.data, index)
            cf = cw.cf_lowpass()
            # Normalize
            #cf = cf / max(cf)
            t=t[0:len(cf)]
            #self.canvas.plot(t, cf, index, is_twinx=True, color="red",linewidth=0.5)
            #self.canvas.set_ylabel_twinx(index, "CWT (CF)")
            ax = self.canvas.get_axe(index)
            ax2 = ax.twinx()
            ax2.plot(t, cf, color="red", linewidth=0.5, alpha = 0.5)
            #ax2.set_ylim(-1.05, 1.05)
            last_index = index
            tr_cf = tr.copy()
            tr_cf.data = cf
            tr_cf.times = t
            cfs.append(tr_cf)
        ax = self.canvas.get_axe(last_index)
        try:
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(last_index, "Date")
        except:
            pass
        self.cf = Stream(traces=cfs)

    def envelope(self):
        self.cf = []
        cfs = []
        files_at_page = self.get_files_at_page()
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        for index, file_path in enumerate(files_at_page):
            tr = self.st[index]
            t = tr.times("matplotlib")
            st_stats = ObspyUtil.get_stats_from_trace(tr)
            cf = envelope(tr.data, tr.stats.sampling_rate)
            self.canvas.plot_date(t, tr.data, index, color="black", fmt='-', linewidth=0.5)
            self.canvas.plot_date(t, cf, index, color="blue", clear_plot=False, fmt='-', linewidth=0.5, alpha = 0.5)
            info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
            self.canvas.set_plot_label(index, info)
            self.redraw_pickers(file_path, index)
            self.redraw_chop(tr, tr.data, index)
            last_index = index
            tr_cf = tr.copy()
            tr_cf.data = cf
            tr_cf.times = t
            cfs.append(tr_cf)
        ax = self.canvas.get_axe(last_index)
        try:
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(last_index, "Date")
        except:
            pass

        self.cf = Stream(traces=cfs)

    def spectral_entropy(self):

        params = self.settings_dialog.getParameters()
        win = params["win_entropy"]

        self.cf = []
        cfs = []
        files_at_page = self.get_files_at_page()

        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        for index, file_path in enumerate(files_at_page):
            if self.cancelled:
                return

            tr = self.st[index]
            t = tr.times("matplotlib")

            delta =tr.stats.delta
            st_stats = ObspyUtil.get_stats_from_trace(tr)
            cf = Entropydetect(tr.data, win, delta)
            t_entropy = t[0:len(cf)]
            self.canvas.plot_date(t, tr.data, index, color="black", fmt='-', linewidth=0.5)
            info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
            self.canvas.set_plot_label(index, info)
            self.redraw_pickers(file_path, index)
            self.redraw_chop(tr, tr.data, index)
            cf = cf / max(cf)
            ax = self.canvas.get_axe(index)
            ax2 = ax.twinx()
            ax2.plot(t_entropy, cf, color="green", linewidth=0.5, alpha=0.5)
            ax2.set_ylim(-1.05, 1.05)
            last_index = index
            tr_cf = tr.copy()
            tr_cf.data = cf
            tr_cf.times = t_entropy
            cfs.append(tr_cf)
            self.value_entropy_init.emit(index + 1)

        ax = self.canvas.get_axe(last_index)
        try:
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(last_index, "Date")
        except:
            pass

        self.cf = Stream(traces=cfs)


    # @AsycTime.run_async()
    def detect_events(self):

        #to make a detection it is needed to trim the data otherwise,
        # is going to take the starttime and endtime of the file

        params = self.settings_dialog.getParameters()
        threshold = params["ThresholdDetect"]
        coincidences = params["Coincidences"]
        cluster  = params["Cluster"]

        standard_deviations = []
        all_traces = []

        starttime = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        endtime = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)


        for tr in self.st:
            cw = ConvolveWaveletScipy(tr)
            if self.trimCB.isChecked():
                cw.setup_wavelet(starttime, endtime, wmin=5, wmax=5, tt=10, fmin=0.5, fmax=10, nf=40, use_rfft=False,
                             decimate=False)
            else:
                cw.setup_wavelet(wmin=5, wmax=5, tt=10, fmin=0.5, fmax=10, nf=40, use_rfft=False,
                                 decimate=False)

            cf = cw.cf_lowpass()
            # Normalize
            #cf = cf / max(cf)
            standard_deviations.append(np.std(cf))

            tr_cf = tr.copy()
            tr_cf.data = cf
            all_traces.append(tr_cf)

        max_threshold = threshold*np.mean(standard_deviations)
        min_threshold = 1*np.mean(standard_deviations)

        self.st = Stream(traces=all_traces)

        trigger = coincidence_trigger(trigger_type=None, thr_on = max_threshold, thr_off = min_threshold,
                                     trigger_off_extension = 0, thr_coincidence_sum = coincidences, stream=self.st,
                                      similarity_threshold = 0.8, details=True)


        for k in range(len(trigger)):
            detection = trigger[k]
            for key in detection:

                if key == 'time':
                    time = detection[key]
                    self.events_times.append(time)
        # calling for 1D clustering more than one detection per earthquake //eps seconds span
        try:
            self.events_times, str_times = MseedUtil.cluster_events(self.events_times, eps=cluster)

            with open(self.path_detection, "w") as fp:
                json.dump(str_times, fp)

            self.plot_seismogram()
            md = MessageDialog(self)
            md.set_info_message("Events Detection done")
        except:
            md = MessageDialog(self)
            md.set_info_message("No Detections")

    def write_files_page(self):

        root_path = os.path.dirname(os.path.abspath(__file__))
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if self.st:
            n = len(self.st)
            for j in range(n):
                tr = self.st[j]
                t1 = tr.stats.starttime
                id = tr.id+"."+"D"+"."+str(t1.year)+"."+str(t1.julday)
                print(tr.id, "Writing data processed")
                path_output = os.path.join(dir_path, id)
                tr.write(path_output, format="MSEED")

    def save_cf(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if self.cf:
            n = len(self.cf)
            for j in range(n):
                tr = self.cf[j]
                t1 = tr.stats.starttime
                id = tr.id + "." + "M" + "." + str(t1.year) + "." + str(t1.julday)
                print(tr.id, "Writing data processed")
                path_output = os.path.join(dir_path, id)
                tr.write(path_output, format="MSEED")

    def stack_all_seismograms(self):
        params = self.settings_dialog.getParameters()

        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=1, ncols=1)
        index = 0
        ##
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        wavenumber = array_analysis.array()
        stream_stack, t, stats, time = wavenumber.stack_seismograms(self.st)
        stack = wavenumber.stack(stream_stack,stack_type=params["stack type"])

        self.canvas.plot_date(time, stack, index, clear_plot=True, color='steelblue', fmt='-', linewidth=0.5)
        info = "{}".format(stats['station'])
        self.canvas.set_plot_label(index, info)
        try:

            self.cf = Stream([Trace(data=stack, header=stats)])
            ax = self.canvas.get_axe(0)
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(0, "Date")
        except:
            pass

    def plot_all_seismograms(self):
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=1, ncols=1)
        index = 0
        colors = ['black','indianred','chocolate','darkorange','olivedrab','lightseagreen',
                  'royalblue','darkorchid','magenta']
        ##
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)

        if len(self.st)>0:
            i = 0
            for tr in self.st:
                if len(tr) > 0:
                    t = tr.times("matplotlib")
                    s = tr.data
                    if len(self.st)<10:
                        self.canvas.plot_date(t, s, index, clear_plot=False, color=colors[i], fmt='-', alpha = 0.5,
                                          linewidth=0.5, label= tr.id)

                    else:
                        self.canvas.plot_date(t, s, index, clear_plot=False, color='black', fmt='-', alpha=0.5,
                                              linewidth=0.5)
                    i = i + 1
        try:
            ax = self.canvas.get_axe(0)
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(0, "Date")
            ax.legend()
        except:
            pass

    def plot_prs(self):

        dist_all = []
        baz_all = []
        distance_in_km = True
        depth = self.event_info.event_depth
        otime = self.event_info.event_time

        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=1, ncols=1)
        index = 0
        ##
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)


        if len(self.st) > 0:

            self.st.detrend(type="simple")
            self.st.reverse()
            if self.sortCB.isChecked():
                if self.comboBox_sort.currentText() == "Distance":
                    for tr in self.st:
                        st_stats = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, tr)
                        if st_stats:
                            dist, _, _ = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                                          self.event_info.longitude)
                            dist_all.append(dist / 1000)

                    max_dist = kilometers2degrees(max(dist_all))
                    min_dist = kilometers2degrees(min(dist_all))
                    dist_all.sort()
                    dist_all.reverse()

                    if sum(dist_all)/len(dist_all) > 700:
                        distance_in_km = False
                        for index, dist in enumerate(dist_all):
                            dist_all[index] = kilometers2degrees(dist)
                        max_dist = max(dist_all)
                        min_dist = min(dist_all)
                    self.message_dataless_not_found()

                    arrivals = ObspyUtil.get_trip_times(source_depth=depth, min_dist = min_dist, max_dist=max_dist)
                    all_arrivals = ObspyUtil.convert_travel_times(arrivals, otime,
                                                                  dist_km = distance_in_km)

                # elif self.comboBox_sort.currentText() == "Back Azimuth":
                #     for tr in self.st:
                #         st_stats = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, tr)
                #
                #         if st_stats:
                #             _, _, az_from_epi = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude,
                #                                                  self.event_info.latitude,
                #                                                  self.event_info.longitude)
                #             baz_all.append(az_from_epi)
                #
                #     baz_all.sort()
                #     baz_all.reverse()
                #    self.message_dataless_not_found()

            i = 0
            for tr in self.st:
                if len(tr) > 0:
                    try:
                        t = tr.times("matplotlib")
                        if distance_in_km:
                            s = 5*(tr.data/np.max(tr.data)) + dist_all[i]
                        else:
                            s = 0.5 * (tr.data / np.max(tr.data)) + dist_all[i]

                        self.canvas.plot_date(t, s, 0, clear_plot=False, color="black", fmt='-', alpha=0.5,
                                                      linewidth=0.5, label="")

                    except:
                        pass
                    i = i + 1

        if distance_in_km:
            self.canvas.plot_date(all_arrivals["P"]["times"], all_arrivals["P"]["distances"], 0, clear_plot=False, fmt='-', alpha=0.5,
                              linewidth=1.0, label="P")

            self.canvas.plot_date(all_arrivals["S"]["times"], all_arrivals["S"]["distances"], 0, clear_plot=False, fmt='-', alpha=0.5,
                              linewidth=1.0, label="S")

        else:
            for key in all_arrivals:
                self.canvas.plot_date(all_arrivals[key]["times"], all_arrivals[key]["distances"], 0, clear_plot=False, fmt='-', alpha=0.5,
                                  linewidth=1.0, label=str(key))
        try:
            ax = self.canvas.get_axe(0)
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(0, "Date")
            ax.legend()
        except:
            pass

        self.actionPlot_Record_Section.setDisabled(True)

    def plot_map_stations(self):

        try:
            [lat,lon] = [self.event_info.latitude, self.event_info.longitude]
            #obsfiles = self.files_path

            map_dict={}
            sd = []

            for tr in self.st:
                #st = SeismogramDataAdvanced(file)

                #name = st.stats.Network+"."+st.stats.Station
                name = tr.stats.network+"."+tr.stats.station
                sd.append(name)

                #st_coordinates = self.__metadata_manager.extract_coordinates(self.inventory, file)
                st_coordinates = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, tr)
                map_dict[name] = [st_coordinates.Latitude, st_coordinates.Longitude]

            self.map_stations = StationsMap(map_dict)
            self.map_stations.plot_stations_map(latitude = lat, longitude=lon)

        except:
            md = MessageDialog(self)
            md.set_error_message("couldn't plot stations map, please check your metadata and the trace headers")

    def redraw_pickers(self, file_name, axe_index):

        picked_at = {key: values for key, values in self.picked_at.items()}  # copy the dictionary.
        for key, value in picked_at.items():
            ps: PickerStructure = value
            if file_name == ps.FileName:
                #new_line = self.canvas.draw_arrow(ps.XPosition, axe_index, ps.Label,
                #                                  amplitude=ps.Amplitude, color=ps.Color, picker=True)
                new_line = self.canvas.draw_arrow(ps.XPosition, axe_index, ps.Label, amplitude=ps.Amplitude,
                                                  color=ps.Color, picker=True)
                #picked_at.pop(key)
                self.picked_at.pop(key)
                self.picked_at[str(new_line)] = ps

    def redraw_event_times(self, index):
        if len(self.events_times)>0:
            for k in self.events_times:
                k = k.matplotlib_date
                self.canvas.draw_arrow(k, index, "Event Detected", color="blue", linestyles='--', picker=False)

    def redraw_chop(self, tr, s, ax_index):
       self.kind_wave = self.ChopCB.currentText()
       for key, value in self.chop.items():
           if tr.id in self.chop[key]:
                t = self.chop[key][tr.id][1]
                xmin_index = self.chop[key][tr.id][3]
                xmax_index = self.chop[key][tr.id][4]
                data = s[xmin_index:xmax_index]
                self.chop[key][tr.id][2] = data
                self.canvas.plot_date(t, data, ax_index, clear_plot=False, color=self.color[key],
                                 fmt='-', linewidth=0.5)

    def on_click_matplotlib(self, event, canvas):
        if isinstance(canvas, MatplotlibCanvas):
            polarity, color = map_polarity_from_pressed_key(event.key)
            phase = self.comboBox_phases.currentText()
            #click_at_index = event.inaxes.rowNum
            click_at_index = self.ax_num
            x1, y1 = event.xdata, event.ydata
            #x2, y2 = event.x, event.y
            stats = ObspyUtil.get_stats(self.get_file_at_index(click_at_index))
            # Get amplitude from index
            #x_index = int(round(x1 * stats.Sampling_rate))  # index of x-axes time * sample_rate.
            #amplitude = canvas.get_ydata(click_at_index).item(x_index)  # get y-data from index.
            #amplitude = y1
            label = "{} {}".format(phase, polarity)
            tt = UTCDateTime(mdt.num2date(x1))
            diff = tt - self.st[self.ax_num].stats.starttime
            t = stats.StartTime + diff
            if self.decimator[0] is not None:
                idx_amplitude = int(self.decimator[0]*diff)
            else:
                idx_amplitude = int(stats.Sampling_rate * diff)
            amplitudes = self.st[self.ax_num].data
            amplitude = amplitudes[idx_amplitude]
            uncertainty = self.uncertainities.getUncertainity()

            line = canvas.draw_arrow(x1, click_at_index, label, amplitude=amplitude, color=color, picker=True)
            self.lines.append(line)
            self.picked_at[str(line)] = PickerStructure(tt, stats.Station, x1, uncertainty, amplitude, color, label,
                                                        self.get_file_at_index(click_at_index))
            #print(self.picked_at)
            # Add pick data to file.
            self.pm.add_data(tt, uncertainty, amplitude, stats.Station, phase, Component=stats.Channel,
                             First_Motion=polarity)
            self.pm.save()  # maybe we can move this to when you press locate.


    def on_pick(self, event):
        line = event.artist
        self.canvas.remove_arrow(line)
        picker_structure: PickerStructure = self.picked_at.pop(str(line), None)
        if picker_structure:
            self.pm.remove_data(picker_structure.Time, picker_structure.Station)

    def on_click_plot_arrivals(self, event_time: UTCDateTime, lat: float, long: float, depth: float):
        self.event_info.clear_arrivals()
        for index, file_path in enumerate(self.get_files_at_page()):
            #st_stats = self.dataless_manager.get_station_stats_by_mseed_file(file_path)
            st_stats = self.__metadata_manager.extract_coordinates(self.inventory, file_path)
            #stats = ObspyUtil.get_stats(file_path)
            # TODO remove stats.StartTime and use the picked one from UI.
            self.event_info.plot_arrivals2(index, st_stats)

    def get_arrivals_tf(self):

        try:
            st_stats = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, self.tr_tf)
            self.phases, self.travel_times = self.event_info.get_station_travel_times(st_stats)
            self.phases = list(self.phases)
            self.travel_times = list(self.travel_times)
            eventtime = self.event_info.event_time
            for index, time in enumerate(self.travel_times):
                self.travel_times[index] = (eventtime-self.tr_tf.stats.starttime)+self.travel_times[index]
        except:
            pass

    def stationsInfo(self):

        if self.sortCB.isChecked():
            if self.comboBox_sort.currentText() == "Distance":
                self.files_path.sort(key=self.sort_by_distance_advance)
                self.message_dataless_not_found()

            elif self.comboBox_sort.currentText() == "Back Azimuth":
                self.files_path.sort(key=self.sort_by_baz_advance)
                self.message_dataless_not_found()

        files_at_page = self.get_files_at_page()
        sd = []

        for file in files_at_page:

            st = SeismogramDataAdvanced(file)

            station = [st.stats.Network,st.stats.Station,st.stats.Location,st.stats.Channel,st.stats.StartTime,
                       st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]

            sd.append(station)

        self._stations_info = StationsInfo(sd)
        self._stations_info.show()


    def cross(self):
        self.cf = []
        cfs = []
        max_values = []
        files_at_page = self.get_files_at_page()
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)

        try:
            if len(self.st)>0 and self.trimCB.isChecked():
                num = self.crossSB.value()

                if num<=len(files_at_page):
                    template = self.st[num-1]

                else:
                    template = self.st[0]

                for j, tr in enumerate(self.st):
                    sampling_rates = []
                    if self.crossCB.currentText() == "Auto":
                        template=tr
                        temp_stats = ObspyUtil.get_stats_from_trace(template)
                        st_stats = ObspyUtil.get_stats_from_trace(tr)
                        info = "Auto-Correlation {}.{}.{}".format(st_stats['net'], st_stats['station'],
                                                                            st_stats['channel'])
                    else:
                        st_stats = ObspyUtil.get_stats_from_trace(tr)
                        temp_stats = ObspyUtil.get_stats_from_trace(template)
                        info = "Cross-Correlation {}.{}.{} --> {}.{}.{}".format(st_stats['net'], st_stats['station'],
                                                                            st_stats['channel'],
                                                                            temp_stats['net'], temp_stats['station'],
                                                                            temp_stats['channel'])

                    fs1 = tr.stats.sampling_rate
                    fs2 = template.stats.sampling_rate
                    sampling_rates.append(fs1)
                    sampling_rates.append(fs2)
                    max_sampling_rates = np.max(sampling_rates)

                    if fs1 != fs2:
                        self.tr.resample(max_sampling_rates)
                        self.template.resample(max_sampling_rates)

                    cc = correlate_maxlag(tr.data, template.data, maxlag=max([len(tr.data), len(template.data)]))

                    stats = {'network': st_stats['net'], 'station': st_stats['station'], 'location': '',
                             'channel': st_stats['channel'], 'npts': len(cc),
                             'sampling_rate': max_sampling_rates, 'mseed': {'dataquality': 'M'},
                             'starttime': temp_stats['starttime']}

                    values = [np.max(cc),np.min(cc)]
                    values = np.abs(values)

                    if values[0]>values[1]:
                        maximo = np.where(cc == np.max(cc))
                    else:
                        maximo = np.where(cc == np.min(cc))

                    max_values.append(maximo)
                    self.canvas.plot(get_lags(cc) / max_sampling_rates, cc, j, clear_plot=True,
                                     linewidth=0.5, color="black")

                    max_line = ((maximo[0][0])/max_sampling_rates)-0.5*(len(cc)/max_sampling_rates)
                    self.canvas.draw_arrow(max_line, j, "max lag", color="red", linestyles='-', picker=False)
                    self.canvas.set_plot_label(j, info)
                    ax = self.canvas.get_axe(j)
                    ax.set_xlim(min(get_lags(cc) / max_sampling_rates), max(get_lags(cc) / max_sampling_rates))
                    # saving
                    cfs.append(Trace(cc , stats))

                self.canvas.set_xlabel(j, "Time [s] from zero lag")

                self.cf = Stream(cfs)
        except:
            md = MessageDialog(self)
            md.set_warning_message("Check correlation template and trim time")

    # TODO implement your logic here for multiple select
    def on_multiple_select(self, ax_index, xmin, xmax):
        pass

    def on_select(self, ax_index, xmin, xmax):
        self.kind_wave = self.ChopCB.currentText()
        tr = self.st[ax_index]
        t = self.st[ax_index].times("matplotlib")
        y = self.st[ax_index].data
        dic_metadata = ObspyUtil.get_stats_from_trace(tr)
        metadata = [dic_metadata['net'], dic_metadata['station'], dic_metadata['location'], dic_metadata['channel'],
                    dic_metadata['starttime'],dic_metadata['endtime'],dic_metadata['sampling_rate'],
                    dic_metadata['npts']]
        id = tr.id
        self.canvas.plot_date(t, y, ax_index, clear_plot=False, color="black", fmt='-', linewidth=0.5)
        xmin_index = np.max(np.where(t <= xmin))
        xmax_index = np.min(np.where(t >= xmax))
        t = t[xmin_index:xmax_index]
        s = y[xmin_index:xmax_index]
        self.canvas.plot_date(t, s, ax_index, clear_plot=False, color = self.color[self.kind_wave], fmt='-', linewidth=0.5)
        id = {id: [metadata, t, s, xmin_index, xmax_index]}
        self.chop[self.kind_wave].update(id)

    def on_multiple_select(self, ax_index, xmin, xmax):

        self.kind_wave = self.ChopCB.currentText()
        self.set_pagination_files(self.files_path)
        files_at_page = self.get_files_at_page()

        for ax_index, file_path in enumerate(files_at_page):
            tr = self.st[ax_index]
            t = self.st[ax_index].times("matplotlib")
            y = self.st[ax_index].data
            dic_metadata = ObspyUtil.get_stats_from_trace(tr)
            metadata = [dic_metadata['net'], dic_metadata['station'], dic_metadata['location'], dic_metadata['channel'],
                        dic_metadata['starttime'], dic_metadata['endtime'], dic_metadata['sampling_rate'],
                        dic_metadata['npts']]
            id = tr.id
            self.canvas.plot_date(t, y, ax_index, clear_plot=False, color="black", fmt='-', linewidth=0.5)
            xmin_index = np.max(np.where(t <= xmin))
            xmax_index = np.min(np.where(t >= xmax))
            t = t[xmin_index:xmax_index]
            s = y[xmin_index:xmax_index]
            self.canvas.plot_date(t, s, ax_index, clear_plot=False, color=self.color[self.kind_wave], fmt='-',
                                  linewidth=0.5)
            id = {id: [metadata, t, s, xmin_index, xmax_index]}
            self.chop[self.kind_wave].update(id)


    def enter_axes(self, event):
         self.ax_num = self.canvas.figure.axes.index(event.inaxes)


    def find_chop_by_ax(self, ax):
        id = self.st[ax].id
        for key, value in self.chop[self.kind_wave].items():
            if id in self.chop[self.kind_wave]:
                identified_chop = self.chop[self.kind_wave][id]
            else:
                pass
        return identified_chop, id

    def clean_events_detected(self):
        if len(self.events_times)>0:
            self.events_times = []
            self.plot_seismogram()


    def clean_all_chop(self):
        self.chop = {'Body waves': {}, 'Surf Waves': {}, 'Coda': {}, 'Noise': {}}
        self.plot_seismogram()

    def clean_chop_at_page(self):
        if self.st:
            try:
                n = len(self.st)
                self.kind_wave = self.ChopCB.currentText()
                for j in range(n):
                    tr = self.st[j]
                    if tr.id in self.chop[self.kind_wave]:
                        self.chop[self.kind_wave].pop(tr.id)
                        data = tr.data
                        t = tr.times("matplotlib")
                        self.canvas.plot_date(t, data, j, clear_plot=False, color='black', fmt='-', linewidth=0.5)
                        self.redraw_chop(tr, data, j)
            except Exception:
                raise Exception('Nothing to clean')

    def key_pressed(self, event):

        if event.key == 'q':
            files_at_page = self.get_files_at_page()
            x1, y1 = event.xdata, event.ydata
            tt = UTCDateTime(mdt.num2date(x1))
            set_qdatetime(tt, self.dateTimeEdit_1)
            for index, file_path in enumerate(files_at_page):
                self.canvas.draw_arrow(x1, index, arrow_label="st", color="purple", linestyles='--', picker=False)

        if event.key == 'e':
            files_at_page = self.get_files_at_page()
            x1, y1 = event.xdata, event.ydata
            tt = UTCDateTime(mdt.num2date(x1))
            set_qdatetime(tt, self.dateTimeEdit_2)
            for index, file_path in enumerate(files_at_page):
                self.canvas.draw_arrow(x1, index, arrow_label= "et", color="purple", linestyles='--', picker=False)


        if event.key == 'd':
           self.kind_wave = self.ChopCB.currentText()
           [identified_chop, id] = self.find_chop_by_ax(self.ax_num)
           self.chop[self.kind_wave].pop(id)
           tr = self.st[self.ax_num]
           data = tr.data
           t = tr.times("matplotlib")
           self.canvas.plot_date(t, data, self.ax_num, clear_plot=False, color='black', fmt='-', linewidth=0.5)
           self.redraw_chop(tr, data, self.ax_num)

        if event.key == 'a':
            self.kind_wave = self.ChopCB.currentText()
            [identified_chop, id]= self.find_chop_by_ax(self.ax_num)
            data = identified_chop[2]
            delta = 1 / identified_chop[0][6]
            [spec, freq, jackknife_errors] = spectrumelement(data, delta, id)
            self.spectrum = PlotToolsManager(id)
            self.spectrum.plot_spectrum(freq, spec, jackknife_errors)


        if event.key == 'm':
            self.canvas.draw_selection_TF(self.ax_num)
            self.tr_tf = self.st[self.ax_num]
            self.get_arrivals_tf()

        if event.key == 'h':

            self.canvas.draw_selection(self.ax_num)
            self.special_selection.append(self.files_at_page[self.ax_num])


        if event.key == 'j':

            self.canvas.draw_selection(self.ax_num, check = False)
            if self.files_at_page[self.ax_num] in self.special_selection:
                self.special_selection.pop(self.ax_num)


        if event.key == 'f':
            self.kind_wave = self.ChopCB.currentText()
            id = ""
            self.spectrum = PlotToolsManager(id)
            self.spectrum.plot_spectrum_all(self.chop[self.kind_wave].items())


        if event.key == 'z':
            # Compute Multitaper Spectrogram
            params = self.settings_dialog.getParameters()


            start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
            end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
            [identified_chop, id] = self.find_chop_by_ax(self.ax_num)
            tini = identified_chop[3]
            tend = identified_chop[4]
            data = identified_chop[2]
            t = identified_chop[1]
            npts = len(data)
            fs = identified_chop[0][6]
            delta = 1 /fs
            fn = fs/2


            win = int(params["Win"]*fs)
            tbp = params["TW"]
            ntapers = int(params["N Tapers"])
            f_min = 0
            f_max = fn
            print(win,tbp,ntapers)
            self.spectrogram = PlotToolsManager(id)
            if win > 0.5*fs and ntapers > 2 and tbp > 2:
                [x,y,z] = self.spectrogram.compute_spectrogram_plot(data, win, delta, tbp, ntapers, f_min, f_max, t)
            else:
                [x, y, z] = self.spectrogram.compute_spectrogram_plot(data, int(3*fs), delta, 3.5, 3, f_min, f_max, t)
            ax = self.canvas.get_axe(self.ax_num)

            ax2 = ax.twinx()
            z= np.clip(z, a_min=-120, a_max=0)
            cs = ax2.contourf(x, y, z, levels=50, cmap=plt.get_cmap("jet"), alpha = 0.2)
            fig = ax2.get_figure()
            ax2.set_ylim(0, fn)
            t = t[0:len(x)]
            ax2.set_xlim(t[0],t[-1])
            ax2.set_ylabel('Frequency [ Hz]')
            vmin = -120
            vmax = 0
            cs.set_clim(vmin, vmax)
            # TODO COLORBAR STABLE
            #axs = []
            #for j in range(self.items_per_page):
            #    axs.append(self.canvas.get_axe(j))

            #if self.nums_clicks > 0:
            #    pass
            #else:

            #    self.cbar = fig.colorbar(cs, ax=axs[j], extend='both', orientation='horizontal', pad=0.2)
            #    self.cbar.ax.set_ylabel("Power [dB]")

            tr = self.st[self.ax_num]
            tt = tr.times("matplotlib")
            data = tr.data
            self.canvas.plot_date(tt, data, self.ax_num, clear_plot=False, color='black', fmt='-', linewidth=0.5)
            auto_start = min(tt)
            auto_end = max(tt)

            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(auto_start), mdt.num2date(auto_end))

            ax.set_ylim(min(data),max(data))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            #self.nums_clicks = self.nums_clicks+1

    def availability(self):

        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory')
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', "",
                                                           pw.QFileDialog.DontUseNativeDialog)
        if os.path.isdir(dir_path):
            try:

                MseedUtil.data_availability(dir_path)

            except:
                md = MessageDialog(self)
                md.set_warning_message("No data available")

    def availability_all_tree(self):

        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory')
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', "",
                                                           pw.QFileDialog.DontUseNativeDialog)
        if os.path.isdir(dir_path):
            try:
                MseedUtil.data_availability(dir_path, only_this = False)
            except:
                md = MessageDialog(self)
                md.set_warning_message("No data available")

    def open_magnitudes_calculator(self):
        hyp_file = os.path.join(ROOT_DIR, "earthquakeAnalisysis", "location_output", "loc", "last.hyp")
        origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file)
        if isinstance(origin, Origin):
            self._magnitude_calc = MagnitudeCalc(origin, self.inventory, self.chop)
            self._magnitude_calc.show()

    def open_solutions(self):

        output_path = os.path.join(ROOT_DIR,'earthquakeAnalisysis', 'location_output', 'obs', 'output.txt')

        try:

            if platform == "darwin":

                command = "{} {}".format('open', output_path)

            else:

                command = "{} {}".format('xdg - open', output_path)

            exc_cmd(command, cwd = ROOT_DIR)

        except:

            md = MessageDialog(self)
            md.set_error_message("Coundn't open pick file")

    def open_events(self):

        output_path = os.path.join(EVENTS_DETECTED,'event_autodetects.txt')

        try:

            if platform == "darwin":

                command = "{} {}".format('open', output_path)
            else:

                command = "{} {}".format('xdg - open', output_path)


            exc_cmd(command, cwd = ROOT_DIR)

        except:

            md = MessageDialog(self)
            md.set_error_message("Coundn't open pick file")

    def remove_picks(self):
        md = MessageDialog(self)
        output_path = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'obs', 'output.txt')

        try:
            self.pm.clear()
            self.picked_at = {}
            command = "{} {}".format('rm', output_path)
            exc_cmd(command, cwd=ROOT_DIR)
            md.set_info_message("Removed picks from file")
            self.canvas.remove_arrows(self.lines)
            self.pm = PickerManager()
        except:

            md.set_error_message("Coundn't remove pick file")

    def start_location(self):
        import glob

        md = MessageDialog(self)
        output_path = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'obs', 'output.txt')

        try:
            try:
                command = "{} {}".format('rm', output_path)
                exc_cmd(command, cwd=ROOT_DIR)
            except:
                pass

            output_path = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'loc')

            try:
                files = glob.glob(os.path.join(output_path,"*"))
                for f in files:
                    os.remove(f)
            except:
                pass

            output_path = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'first_polarity')
            try:

                os.remove(os.path.join(output_path,"test.inp"))
                os.remove(os.path.join(output_path, "mechanism.out"))
                os.remove(os.path.join(output_path, "focmec.lst"))

            except:
                pass

            md.set_info_message("Ready for new location")
        except:
            md = MessageDialog(self)
            md.set_error_message("Coundn't remove location, please review ", output_path)


    def open_array_analysis(self):
        self.controller().open_array_window()

    def open_moment_tensor(self):
        self.controller().open_momentTensor_window()

    def time_frequency_analysis(self):
        self.controller().open_seismogram_window()
        answer = pw.QMessageBox.question(self, "Export Seismogram to TF Analysis",
            "if you have selected a seismogram this will be automatically processed in TF Analysis module")

        if pw.QMessageBox.Yes == answer:
            if len(self.tr_tf) > 0 and self.phases!=None and self.travel_times!=None:

                self.controller().time_frequency_frame.process_import_trace(self.tr_tf, phases=self.phases,
                                                                            travel_times=self.travel_times)
            else:
                self.controller().time_frequency_frame.process_import_trace(self.tr_tf)

            self.phases = None
            self.travel_times = None

    def open_receiver_functions(self):
        self.controller().open_receiverFunctions()

    def controller(self):
        from isp.Gui.controllers import Controller
        return Controller()

    ######### Progress Bar #############

    def spectral_entropy_progress(self):
        self.cancelled = False
        parallel_progress_run("Current progress: ", 0, len(self.st), self,
                              self.spectral_entropy, self.cancelled_callback,
                              signalValue=self.value_entropy_init)

    # def retrieve_event(self, event_location: EventLocationModel):
    #     print(event_location)

    def set_catalog_values(self, event_catalog):
        print("Set event values", event_catalog)
        otime = event_catalog[0]
        lat = event_catalog[1]
        lon = event_catalog[2]
        depth = event_catalog[3]

        set_qdatetime(otime, self.dateTimeEdit_1)
        set_qdatetime(otime+900, self.dateTimeEdit_2)
        self.event_info.set_time(otime)
        self.event_info.set_coordinates([lat,lon,depth])

    def set_event_download_values(self, event_catalog):
        print("Set event values", event_catalog)
        otime = event_catalog[0]
        lat = event_catalog[1]
        lon = event_catalog[2]
        depth = event_catalog[3]

        set_qdatetime(otime, self.dateTimeEdit_1)
        set_qdatetime(otime+900, self.dateTimeEdit_2)
        self.event_info.set_time(otime)
        self.event_info.set_coordinates([lat,lon,depth])