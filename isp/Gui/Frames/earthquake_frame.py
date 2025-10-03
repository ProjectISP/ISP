import copy
import matplotlib.dates as mdt
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMessageBox
from matplotlib.backend_bases import MouseButton
from obspy import UTCDateTime, Stream, Trace, Inventory
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from surfquakecore.project.surf_project import SurfProject
from isp.DataProcessing import DatalessManager, SeismogramDataAdvanced, ConvolveWaveletScipy
from surfquakecore.coincidence_trigger.cf_kurtosis import CFKurtosis
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.DataProcessing.plot_tools_manager import PlotToolsManager
from isp.Exceptions import parse_excepts
from isp.Gui import pw, pqg, pyc
from isp.Gui.Frames import BaseFrame, UiEarthquakeAnalysisFrame, Pagination, MessageDialog, EventInfoBox, \
    MatplotlibCanvas
from isp.Gui.Frames.autopick_frame import Autopick
from isp.Gui.Frames.earthquake_frame_tabs import Earthquake3CFrame
from isp.Gui.Frames.locate_frame import Locate
#from isp.Gui.Frames.open_magnitudes_calc import MagnitudeCalc
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.plot_polarization import PlotPolarization
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
from isp.earthquakeAnalysis import PickerManager
import numpy as np
import os
from isp.Utils.subprocess_utils import exc_cmd, open_url
from isp import ROOT_DIR, AUTOMATIC_PHASES, PICKING_DIR
import matplotlib.pyplot as plt
from isp.earthquakeAnalysis.stations_map import StationsMap
from isp.seismogramInspector.signal_processing_advanced import spectrumelement, sta_lta, envelope, Entropydetect, \
    correlate_maxlag, get_lags
import subprocess
import platform


class EarthquakeAnalysisFrame(BaseFrame, UiEarthquakeAnalysisFrame):
    value_entropy_init = pyc.pyqtSignal(int)
    plot_progress = pyc.pyqtSignal()

    def __init__(self):
        super(EarthquakeAnalysisFrame, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Earthquake Location')
        self.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))

        self.url = 'https://projectisp.github.io/ISP_tutorial.github.io/el/'
        self.zoom_diff = None
        self.phases = None
        self.travel_times = None
        self.cancelled = False
        self.aligned_checked = False
        self.aligned_picks = False
        self.aligned_fixed = False
        self.shift_times = None
        self.info_proj_reloaded = {}
        self.project = {}
        self.project_filtered = {}
        self.dist_all = []
        self.baz_all = []
        self.special_selection = []
        self.lines = []
        self.progressbar = pw.QProgressDialog(self)
        self.progressbar.setWindowTitle('Earthquake Location')
        self.progressbar.setLabelText(" Computing Auto-Picking ")
        self.progressbar.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.path_phases = os.path.join(AUTOMATIC_PHASES, "phases_autodetected.txt")
        self.progressbar.close()
        self.settings_dialog = SettingsDialog(self)
        self.inventory = {}
        self.files = []
        self.files_path = []
        self.events_times = []
        self.removed_picks = []
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
        self.numbers = ["1", "2", "3", "4", "5", "6"]
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
                                       button=MouseButton.RIGHT, sharex=True, rectprops=dict(alpha=0.2,
                                                                                             facecolor='blue'))

        self.canvas.mpl_connect('key_press_event', self.key_pressed)
        self.canvas.mpl_connect('axes_enter_event', self.enter_axes)

        # Event info register
        self.event_info = EventInfoBox(self.eventInfoWidget, self.canvas)
        self.event_info.register_plot_arrivals_click(self.on_click_plot_arrivals)
        self.event_info.register_plot_record_section_click(self.on_click_plot_record_section)
        self.earthquake_3c_frame = Earthquake3CFrame(self.parentWidget3C)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)

        # Bind buttons
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_file(self.metadata_path_bind))
        self.selectDirBtn.clicked.connect(lambda: self.load_project())
        self.updateBtn.clicked.connect(self.plot_seismogram)
        self.filterProjectBtn.clicked.connect(self.reload_current_project)
        self.stations_infoBtn.clicked.connect(self.stationsInfo)
        self.phaseUncertaintyBtn.clicked.connect(self.open_uncertainity_settings)
        self.rotateBtn.clicked.connect(self.rotate)
        self.particleMotionBtn.clicked.connect(self.plot_particle_motion)
        self.mapBtn.clicked.connect(self.plot_map_stations)
        self.crossBtn.clicked.connect(self.cross)
        self.macroBtn.clicked.connect(self.open_parameters_settings)
        self.__metadata_manager = MetadataManager(self.metadata_path_bind.value)
        self.actionSet_Parameters.triggered.connect(lambda: self.open_parameters_settings())
        self.actionSeismograms.triggered.connect(self.write_files_page)
        self.actionArray_Anlysis.triggered.connect(self.open_array_analysis)
        self.actionMoment_Tensor_Inversion.triggered.connect(self.open_moment_tensor)
        self.actionTime_Frequency_Analysis.triggered.connect(self.time_frequency_analysis)
        #self.actionOpen_Magnitudes_Calculator.triggered.connect(self.open_magnitudes_calculator)
        self.actionSTA_LTA.triggered.connect(self.run_STA_LTA)
        self.actionlogarythmic_dff.triggered.connect(self.cwt_cf)
        self.actionkurtosis.triggered.connect(self.cwt_kurt)
        self.actionEnvelope.triggered.connect(self.envelope)
        self.actionReceiver_Functions.triggered.connect(self.open_receiver_functions)
        self.actionOpen_Settings.triggered.connect(lambda: self.settings_dialog.show())
        self.actionSearch_in_Catalog.triggered.connect(lambda: self.open_catalog_viewer())
        self.actionStack.triggered.connect(lambda: self.stack_all_seismograms())
        # self.actionSpectral_Entropy.triggered.connect(lambda : self.spectral_entropy())
        self.actionSpectral_Entropy.triggered.connect(lambda: self.spectral_entropy_progress())
        self.actionRemove_all_selections.triggered.connect(lambda: self.clean_all_chop())
        self.actionClean_selection.triggered.connect(lambda: self.clean_chop_at_page())
        self.actionClean_Events_Detected.triggered.connect(lambda: self.clean_events_detected())
        self.actionPlot_All_Seismograms.triggered.connect(lambda: self.plot_all_seismograms())
        self.actionOpen_Help.triggered.connect(lambda: self.open_help())
        self.actionData_Availability.triggered.connect(lambda: self.availability())
        self.actionOpen_picksnew.triggered.connect(lambda: self.open_solutions())
        self.actionRemove_picks.triggered.connect(lambda: self.remove_picks())
        self.actionNew_location.triggered.connect(lambda: self.start_location())
        self.actionRun_autoloc.triggered.connect(lambda: self.picker_all())
        self.actionFrom_Phase_Pick.triggered.connect(lambda: self.alaign_picks())
        self.actionUsing_MCCC.triggered.connect(lambda: self.alaign_mccc())
        self.actionDefaultPicks.triggered.connect(lambda: self.import_pick_from_file(default=True))
        self.actionPicksOther_file.triggered.connect(lambda: self.import_pick_from_file(default=False))
        self.actionNew_Project.triggered.connect(lambda: self.new_project())
        self.newProjectBtn.clicked.connect(lambda: self.new_project())
        self.actionLoad_Project.triggered.connect(lambda: self.load_project())
        self.actionPlot_Record_Section.triggered.connect(lambda: self.on_click_plot_record_section())
        self.actionCFs.triggered.connect(lambda: self.save_cf())
        self.actionConcatanate_Waveforms.triggered.connect(self.plot_stream_concat)
        self.runScriptBtn.clicked.connect(self.run_process)
        self.locateBtn.clicked.connect(self.open_locate)
        self.autoPickBtn.clicked.connect(self.open_auto_pick)

        self.pm = PickerManager()  # start PickerManager to save pick location to csv file.

        # Parameters settings

        self.parameters = ParametersSettings()

        # Uncertainity pick
        self.uncertainities = UncertainityInfo()

        # Project

        self.project_dialog = Project()

        # shortcuts
        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+D'), self)
        self.shortcut_open.activated.connect(self.run_process)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+U'), self)
        self.shortcut_open.activated.connect(self.open_solutions)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+L'), self)
        self.shortcut_open.activated.connect(self.open_parameters_settings)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+I'), self)
        self.shortcut_open.activated.connect(self.clean_chop_at_page)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+C'), self)
        self.shortcut_open.activated.connect(self.clean_events_detected)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+M'), self)
        self.shortcut_open.activated.connect(self.plot_stream_concat)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+P'), self)
        self.shortcut_open.activated.connect(self.comboBox_phases.showPopup)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('P'), self)
        self.shortcut_open.activated.connect(self.on_click_plot_record_section)

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

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('U'), self)
        self.shortcut_open.activated.connect(self.open_uncertainity_settings)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('I'), self)
        self.shortcut_open.activated.connect(self.multi_cursor_on)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('O'), self)
        self.shortcut_open.activated.connect(self.multi_cursor_off)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('R'), self)
        self.shortcut_open.activated.connect(self.reload_current_project)

    def cancelled_callback(self):
        self.cancelled = True

    def multi_cursor_on(self):
        self.canvas.activate_multi_cursor()

    def multi_cursor_off(self):
        self.canvas.deactivate_multi_cursor()

    def open_help(self):
        open_url(self.url)

    def open_parameters_settings(self):
        self.parameters.show()

    def open_uncertainity_settings(self):
        self.uncertainities.show()

    def run_process(self):

        from isp.scripts import run_process as rp

        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        hypo_lat = self.event_info.latitude
        hypo_lon = self.event_info.longitude
        hypo_depth = self.event_info.depth
        hypo_origin_time = self.event_info.event_time
        rp(self.st, chop=self.chop, starttime=start_time, endtime=end_time, hypo_lat=hypo_lat, hypo_lon=hypo_lon,
           hypo_depth_km=hypo_depth, hypo_origin_time=hypo_origin_time)

    def new_project(self):
        self.netForm.setText("")
        self.stationForm.setText("")
        self.channelForm.setText("")
        self.trimCB.setChecked(False)

        self.loaded_project = False
        self.setEnabled(False)
        self.project_dialog.exec()
        self.setEnabled(True)
        self.project = self.project_dialog.project

        self.get_now_files()
        # now we can access to #self.project_dialog.project

    def load_project(self):

        self.netForm.setText("")
        self.stationForm.setText("")
        self.channelForm.setText("")
        self.trimCB.setChecked(False)
        self.loaded_project = True

        selected = pw.QFileDialog.getOpenFileName(self, "Select Project", ROOT_DIR)

        md = MessageDialog(self)

        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            try:
                self.current_project_file = selected[0]
                self.project = MseedUtil.load_project(file=selected[0])
                project_name = os.path.basename(selected[0])
                info = MseedUtil.get_project_basic_info(self.project)
                md.set_info_message("Project {} loaded  ".format(project_name))
                if len(info) > 0 and len(self.project) > 0:
                    md.set_info_message("Project {} loaded  ".format(project_name),
                                        "Networks: " + ','.join(info["Networks"][0]) + "\n" +
                                        "Stations: " + ','.join(info["Stations"][0]) + "\n" +
                                        "Channels: " + ','.join(info["Channels"][0]) + "\n" + "\n" +

                                        "Networks Number: " + str(info["Networks"][1]) + "\n" +
                                        "Stations Number: " + str(info["Stations"][1]) + "\n" +
                                        "Channels Number: " + str(info["Channels"][1]) + "\n" +
                                        "Num Files: " + str(info["num_files"]) + "\n" +
                                        "Start Project: " + info["Start"] + "\n" + "End Project: " + info["End"])
                    self.get_now_files()
                else:
                    md.set_warning_message("Empty Project ", "Please provide a root path "
                                                           "with mseed files inside and check the wuery filters applied")
            except:
                md.set_error_message("Project couldn't be loaded ")

        else:
            print("No file selected")
            md.close()


    def reload_current_project(self):

        md = MessageDialog(self)

        self.get_now_files()

        info = MseedUtil.get_project_basic_info(self.project_filtered)

        if len(info) > 0:

            md.set_info_message("New Project reloaded",
                                "Networks: " + ','.join(info["Networks"][0]) + "\n" +
                                "Stations: " + ','.join(info["Stations"][0]) + "\n" +
                                "Channels: " + ','.join(info["Channels"][0]) + "\n" + "\n" +

                                "Networks Number: " + str(info["Networks"][1]) + "\n" +
                                "Stations Number: " + str(info["Stations"][1]) + "\n" +
                                "Channels Number: " + str(info["Channels"][1]) + "\n" +
                                "Num Files: " + str(info["num_files"]) + "\n" +
                                "Start Project: " + info["Start"] + "\n" + "End Project: " + info["End"])

        else:
            md.set_warning_message("Empty Filtered Project ", "Please provide a root path "
                                                              "with mseed files inside and check the query filters applied")

    def plot_particle_motion(self):

        if isinstance(self.st, Stream) and self.trimCB.isChecked():
            try:
                channels_fullfill = []
                for tr in self.st:
                    if tr.stats.channel[-1] == "Z":
                        z = tr
                        channels_fullfill.append("Z")
                    elif (tr.stats.channel[-1] == "1" or tr.stats.channel[-1] == "N" or tr.stats.channel[-1] == "Y"
                          or tr.stats.channel[-1] == "R"):
                        r = tr
                        channels_fullfill.append("1")
                    elif (tr.stats.channel[-1] == "2" or tr.stats.channel[-1] == "E" or tr.stats.channel[-1] == "X"
                          or tr.stats.channel[-1] == "T"):
                        t = tr
                        channels_fullfill.append("2")
                if len(channels_fullfill) ==3:
                    self._plot_polarization = PlotPolarization(z.data, r.data, t.data)
                    self._plot_polarization.show()
                else:
                    md = MessageDialog(self)
                    md.set_warning_message(
                        "You Need to select three components and trim it",
                        "Be sure you have process and plot Z, N, E or Z, 1 ,2 or Z, Y, X\n")
            except:
                md = MessageDialog(self)
                md.set_error_message(
                    "The action is nor allowed",
                    "Be sure you have process and plot all 3-components Z, N, E or Z, 1 ,2 "
                    "or Z, Y, X or Z, R, T\n")
        else:
            md = MessageDialog(self)
            md.set_warning_message(
                "You Need to select three components and trim it",
                "Be sure you have process and plot Z, N, E or Z, 1 ,2 or Z, Y, X\n")

    def open_catalog_viewer(self):

        # catalog viewer
        if isinstance(self.inventory, Inventory):

            self.catalog = SearchCatalogViewer(metadata=self.inventory)
        else:
            self.catalog = SearchCatalogViewer(metadata=None)
        self.catalog.show()

    @pyc.Slot()
    def _increase_progress(self):
        self.progressbar.setValue(self.progressbar.value() + 1)

    def alaign_picks(self):
        self.aligned_picks = True
        phase = self.comboBox_phases.currentText()
        self.aligned_checked = True
        self.pick_times = MseedUtil.get_NLL_phase_picks_phase(phase)
        self.plot_seismogram()

    def import_pick_from_file(self, default=True, reset=False):

        if default:
            selected = [os.path.join(PICKING_DIR, "output.txt")]
        else:
            selected = pw.QFileDialog.getOpenFileName(self, "Select picking file")

        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            self.pick_times_imported = MseedUtil.get_NLL_phase_picks(input_file=selected[0])

        if reset:
            self.pm = PickerManager()

        # save data
        if len(self.pick_times_imported) > 0:
            for key in self.pick_times_imported:
                pick = self.pick_times_imported[key]
                for j in range(len(pick)):
                    pick_value = pick[j][1]

                    self.pm.add_data(pick_value, pick[j][5], pick[j][7], key.split(".")[0], pick[j][0],
                                     First_Motion=pick[j][3], Component=pick[j][2])
                    self.pm.save()

        self.plot_seismogram()

    def __incorporate_picks(self):
        if len(self.pick_times_imported) > 0:
            stations_info = self.__stations_info_list()
            for count, station in enumerate(stations_info):
                id = station[1] + "." + station[3]
                if id in self.pick_times_imported:
                    pick = self.pick_times_imported[id]  # in UTCDatetime needs to be in samples
                    for j in range(len(pick)):
                        pick_value = pick[j][1].matplotlib_date
                        # build the label properly

                        if pick[j][3] == "U":
                            label = pick[j][0] + " +"
                        elif pick[j][3] == "D":
                            label = pick[j][0] + " -"
                        else:
                            label = pick[j][0] + " ?"

                        if [station[1], pick[j][1]] not in self.removed_picks:
                            line = self.canvas.draw_arrow(pick_value, count, arrow_label=label, amplitude=pick[j][7],
                                                          color="green", picker=True)
                            self.lines.append(line)
                            self.picked_at[str(line)] = PickerStructure(pick[j][1], id.split(".")[0], pick_value,
                                                                        pick[j][4],
                                                                        pick[j][7], "green", label,
                                                                        self.get_file_at_index(count))

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
        # self.plot_seismogram()

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
        # self.get_now_files()
        # self.set_pagination_files(self.files_path)

        # self.plot_seismogram()

    def set_pagination_files(self, files_path):
        self.files = files_path
        self.total_items = len(self.files)
        self.pagination.set_total_items(self.total_items)

    def get_now_files(self):

        self.project_filtered = copy.deepcopy(self.project)
        start = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        diff = end - start
        selection = [self.netForm.text(), self.stationForm.text(), self.channelForm.text()]

        try:
            self.project_filtered, self.files_path = MseedUtil.filter_project_keys(self.project_filtered, net=selection[0],
                                                                                   station=selection[1],
                                                                                   channel=selection[2])
        except:
            self.files_path = []

        if len(self.files_path) > 0 and self.trimCB.isChecked() and diff > 0:
            try:
                self.project_filtered, self.files_path = MseedUtil.filter_time(self.project_filtered, starttime=start,
                                                                               endtime=end)
            except:
                self.files_path = []

        if len(self.files_path) > 0:
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
            self.st, self.shift_times = bp.multichanel(self.st, resample=True)
            self.plot_seismogram()

    def sort_by_distance_traces(self, trace):

        st_stats = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, trace)
        if st_stats:

            dist, _, _ = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                          self.event_info.longitude)
            self.dist_all.append(dist / 1000)

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
        if st_stats is not None:

            dist, _, _ = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                          self.event_info.longitude)
            self.dist_all.append(dist / 1000)
            return dist
        else:
            self.dataless_not_found.add(file)
            print("No Metadata found for {} file.".format(file))
            self.dist_all.append(0)
            return 0.

    def sort_by_baz_advance(self, file):

        self.baz_all = []
        st_stats = self.__metadata_manager.extract_coordinates(self.inventory, file)

        if st_stats is not None:

            _, _, az_from_epi = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                                 self.event_info.longitude)
            self.baz_all.append(az_from_epi)
            return az_from_epi
        else:

            self.dataless_not_found.add(file)
            print("No Metadata found for {} file.".format(file))
            self.baz_all.append(0)
            return 0.

    def set_pagination_stream_concat(self, st):

        self.total_items = len(st)
        self.pagination.set_total_items(self.total_items)

    def plot_stream_concat(self):

        # info = MseedUtil.get_project_basic_info(self.project)
        params = self.settings_dialog.getParameters()
        sharey = params["amplitudeaxis"]
        self.concatanate = True

        ## Merge traces from files ##
        all_traces = []

        for index, file_path in enumerate(self.files_at_page):
            sd = SeismogramDataAdvanced(file_path)
            tr = sd.get_waveform_advanced([], self.inventory,
                                          filter_error_callback=self.filter_error_message,
                                          start_time=self.start_time, end_time=self.end_time, trace_number=index)
            all_traces.append(tr)

        st = Stream(all_traces)
        st.merge()

        if len(st) == 0 and len(self.st) > 0:
            md = MessageDialog(self)
            md.set_warning_message("No Concatenate action is required")

        else:
            if isinstance(st, Stream):

                self.canvas.clear()
                self.set_pagination_stream_concat(st)
                self.canvas.set_new_subplot(nrows=len(st), ncols=1, sharey=sharey)
                self.st = []
                self.all_traces = []
                for index, tr in enumerate(st):


                    sd = SeismogramDataAdvanced(file_path=None, stream=tr, realtime=True)
                    tr = sd.get_waveform_advanced(self.parameters_list, self.inventory,
                                                  filter_error_callback=self.filter_error_message,
                                                  start_time=self.start_time, end_time=self.end_time, trace_number=index)
                    if len(tr) > 0:

                        if self.aligned_checked:
                            try:
                                pick_reference = self.pick_times[tr.stats.station + "." + tr.stats.channel]
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

                        self.canvas.plot_date(t, s, index, color="black", fmt='-', linewidth=0.5)
                        info = "{}.{}.{}".format(tr.stats.network, tr.stats.station, tr.stats.channel)
                        self.canvas.set_plot_label(index, info)

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
                        if index != (self.pagination.items_per_page - 1):
                            try:
                                ax.tick_params(bottom=False)
                            except:
                                pass

                        try:
                            self.min_starttime[index] = min(t)
                            self.max_endtime[index] = max(t)
                        except:
                            print("Empty traces")

                        formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
                        ax.xaxis.set_major_formatter(formatter)

                        self.all_traces.append(tr)

                self.st = Stream(self.all_traces)



    def plot_seismogram(self):

        info = MseedUtil.get_project_basic_info(self.project_filtered)
        start_project = UTCDateTime(info['Start'])
        end_project = UTCDateTime(info['End'])
        num_current_seismograms = len(self.files_path)
        params = self.settings_dialog.getParameters()
        self.auto_refresh = params["auto_refresh"]
        self.auto_resample = params["auto_resample"]

        sharey = params["amplitudeaxis"]
        self.concatanate = False
        self.start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        self.end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)

        if len(info) > 0 and num_current_seismograms > 0:
            if self.trimCB.isChecked() and (self.end_time < start_project or self.start_time > end_project):
                md = MessageDialog(self)
                md.set_warning_message("You might want to procees and plot seismograms out of the project scope?,"
                                       " then click cancel",
                                       "Please cancel this operation if last too long")

            if self.trimCB.isChecked() and (self.start_time >= self.end_time):
                md = MessageDialog(self)
                md.set_warning_message("Your endtime is bigger than your starttime!!",
                                       "Please cancel this operation ")

            self.workers = ParallelWorkers(os.cpu_count())

            # Here we can disabled thing or make additional staff
            self.workers.job(self.__plot_seismogram)

            self.decimator = [None, False]
            if self.st:
                del self.st

            self.canvas.clear()

            # if self.trimCB.isChecked() and self.check_start_time != None and self.check_end_time != None:
            #     if self.check_start_time != convert_qdatetime_utcdatetime(self.dateTimeEdit_1) and \
            #             self.check_end_time != convert_qdatetime_utcdatetime(self.dateTimeEdit_2):
            #         self.get_now_files()

            if self.sortCB.isChecked():
                if self.comboBox_sort.currentText() == "Distance":
                    self.files_path.sort(key=self.sort_by_distance_advance)
                    # self.actionPlot_Record_Section.setEnabled(True)
                    self.message_dataless_not_found()

                elif self.comboBox_sort.currentText() == "Back Azimuth":
                    self.files_path.sort(key=self.sort_by_baz_advance)
                    self.message_dataless_not_found()

            if len(self.special_selection) > 0:
                self.set_pagination_files(self.special_selection)
                self.files_at_page = self.get_files_at_page()
            else:
                self.set_pagination_files(self.files_path)
                self.files_at_page = self.get_files_at_page()

            ##

            self.check_start_time = self.start_time
            self.check_end_time = self.end_time
            ##
            self.diff = self.end_time - self.start_time
            if len(self.canvas.axes) != len(self.files_at_page) or self.auto_refresh:
                self.canvas.set_new_subplot(nrows=len(self.files_at_page), ncols=1, sharey=sharey)
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

                ax = self.canvas.get_axe(len(self.files_at_page) - 1)
                # ax.callbacks.connect('xlim_changed', self.on_xlims_change)
                if self.trimCB.isChecked() and self.aligned_picks == False:

                    ax.set_xlim(self.start_time.matplotlib_date, self.end_time.matplotlib_date)
                else:

                    ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))

                self.canvas.set_xlabel(len(self.files_at_page) - 1, "Date")

                # include picked

                self.__incorporate_picks()

            except:
                pass
            self.special_selection = []
            self.aligned_checked = False
            self.aligned_picks = False
        else:
            md = MessageDialog(self)
            md.set_warning_message("Empty Project ", "Please provide a root path "
                                                     "with mseed files inside and check the query filters applied")

    def __plot_seismogram(self, tuple_files):

        index = tuple_files[0]
        file_path = tuple_files[1]

        sd = SeismogramDataAdvanced(file_path)

        if self.trimCB.isChecked() and self.diff >= 0 and self.auto_resample:

            self.decimator = sd.resample_check(start_time=self.start_time, end_time=self.end_time)

        elif self.trimCB.isChecked() == False and self.auto_resample == True:

            self.decimator = sd.resample_check()

        if self.decimator[1]:
            self.parameters_list.insert(0, ['resample_simple', self.decimator[0], True])

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
                    pick_reference = self.pick_times[tr.stats.station + "." + tr.stats.channel]
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

            self.canvas.plot_date(t, s, index, color="black", fmt='-', linewidth=0.5)

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
            if index != (self.pagination.items_per_page - 1):
                try:
                    ax.tick_params(bottom=False)
                except:
                    pass

            st_stats = ObspyUtil.get_stats(file_path)
            self.redraw_chop(tr, s, index)

            if self.decimator[1]:
                warning = "Decimated to " + str(self.decimator[0]) + "  Hz"
                self.canvas.set_warning_label(index, warning)

            if st_stats and self.sortCB.isChecked() == False:
                info = "{}.{}.{}".format(st_stats.Network, st_stats.Station, st_stats.Channel)
                self.canvas.set_plot_label(index, info)

            elif st_stats and self.sortCB.isChecked() and self.comboBox_sort.currentText() == "Distance":

                dist = self.sort_by_distance_advance(file_path)
                dist = "{:.1f}".format(dist / 1000.0)
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

                if self.angCB.currentText() == "to this angle":

                    st1 = self.st.copy()
                    st2 = st1.select(station=stations[k])
                    maxstart = np.max([tr.stats.starttime for tr in st2])
                    minend = np.min([tr.stats.endtime for tr in st2])
                    st2.trim(maxstart, minend)
                    bazim = self.rot_ang.value()
                    bazim = bazim + 180

                else:

                    st1 = self.st.copy()
                    st2 = st1.select(station=stations[k])
                    maxstart = np.max([tr.stats.starttime for tr in st2])
                    minend = np.min([tr.stats.endtime for tr in st2])
                    st2.trim(maxstart, minend)
                    tr = st2[0]
                    coordinates = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, tr)
                    [azim, bazim, inci] = ObspyUtil.coords2azbazinc(coordinates.Latitude, coordinates.Longitude,
                                                                    coordinates.Elevation, self.event_info.latitude,
                                                                    self.event_info.longitude,
                                                                    self.event_info.event_depth)
                    bazim = bazim + 180

                # I sum 180 degrees to be consistent with definition from station to point
                if bazim >= 360:
                    bazim = bazim-360


                # rename channels to ensure rotation
                st2 = ObspyUtil.rename_traces(st2)
                st2.rotate(method='NE->RT', back_azimuth=bazim)

                for tr in st2:
                    all_traces_rotated.append(tr)

            self.st = Stream(traces=all_traces_rotated)
            # plot
            # 8-feb-2025 change to work with possible concatenation of files

            files_at_page = self.get_files_at_page()
            for index, file_path in enumerate(self.st):
                tr = all_traces_rotated[index]
                t = tr.times("matplotlib")
                s = tr.data
                if tr.stats.channel[2] == "T" or tr.stats.channel[2] == "R":
                    st_stats = ObspyUtil.get_stats_from_trace(tr)
                    id_new = st_stats['net'] + "." + st_stats['station'] + "." + st_stats['location'] + "." \
                             + st_stats['channel']
                    # change chop_dictionary
                    if tr.stats.channel[2] == "T":
                        id_old = st_stats['net'] + "." + st_stats['station'] + "." + st_stats['location'] + "." \
                                 + st_stats['channel'][0:2] + "E"
                        try:
                            for key, value in self.chop.items():
                                if id_new in self.chop[key]:
                                    self.chop[key][id_new] = self.chop[key].pop(id_old)
                        except:
                            pass

                    if tr.stats.channel[2] == "R":
                        id_old = st_stats['net'] + "." + st_stats['station'] + "." + st_stats['location'] + "." \
                                 + st_stats['channel'][0:2] + "N"
                        try:
                            for key, value in self.chop.items():
                                if id_new in self.chop[key]:
                                    self.chop[key][id_new] = self.chop[key].pop(id_old)
                        except:
                            pass

                    self.canvas.plot_date(t, s, index, color="steelblue", fmt='-', linewidth=0.5)
                    info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
                    self.canvas.set_plot_label(index, info)

                    if self.concatanate:
                        pass
                    else:
                       self.redraw_pickers(files_at_page[index], index)

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
                last_index = index

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
        #files_at_page = self.get_files_at_page()
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        params = self.settings_dialog.getParameters()
        STA = params["STA"]
        LTA = params["LTA"]
        sharey = params["amplitudeaxis"]
        global_cf_min = float('inf')
        global_cf_max = float('-inf')
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(self.st), ncols=1, sharey=sharey)

        for index, file_path in enumerate(self.st):
            tr = self.st[index]

            if STA < LTA:
                cf = sta_lta(tr.data, tr.stats.sampling_rate, STA=STA, LTA=LTA)
            else:
                cf = sta_lta(tr.data, tr.stats.sampling_rate)

            st_stats = ObspyUtil.get_stats_from_trace(tr)
            # Normalize
            # cf =cf/max(cf)
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
            if sharey:
                # Clean ticks on ax2
                ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

            # Update global y-limits for ax2
            cf_min, cf_max = min(cf), max(cf)
            global_cf_min = min(global_cf_min, cf_min)
            global_cf_max = max(global_cf_max, cf_max)

            last_index = index
            tr_cf = tr.copy()
            tr_cf.data = cf
            tr_cf.times = t
            cfs.append(tr_cf)

        # After the loop, synchronize all ax2 y-limits
        if sharey:
            for index in range(len(self.st)):
                ax2 = self.canvas.get_axe(index).twinx()  # Get the corresponding ax2
                ax2.set_ylim(global_cf_min, global_cf_max)

                # Re-enable right-side ticks (if needed)
                ax2.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)

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
        params = self.settings_dialog.getParameters()
        cycles = params["Num Cycles"]
        fmin = params["Fmin"]
        fmax = params["Fmax"]
        sharey = params["amplitudeaxis"]
        global_cf_min = float('inf')
        global_cf_max = float('-inf')

        self.cf = []
        cfs = []
        # files_at_page = self.get_files_at_page()
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(self.st), ncols=1, sharey=sharey)

        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        diff = end_time - start_time


        for index, file_path in enumerate(self.st):
            tr = self.st[index]
            st_stats = ObspyUtil.get_stats_from_trace(tr)
            cw = ConvolveWaveletScipy(tr)
            if self.trimCB.isChecked() and diff >= 0:

                if fmin < fmax and cycles > 5:
                    tt = int(tr.stats.sampling_rate / fmin)
                    cw.setup_wavelet(start_time, end_time, wmin=cycles, wmax=cycles, tt=tt, fmin=fmin, fmax=fmax, nf=40,
                                     use_rfft=False, decimate=False)

                else:
                    cw.setup_wavelet(start_time, end_time, wmin=6, wmax=6, tt=10, fmin=0.2, fmax=10, nf=40,
                                     use_rfft=False,
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
            # f = np.logspace(np.log10(fmin), np.log10(fmax))
            # k = cycles / (2 * np.pi * f) #one standar deviation
            # delay = np.mean(k)

            tr.stats.starttime = start
            t = tr.times("matplotlib")

            self.canvas.plot_date(t, tr.data, index, color="black", fmt='-', linewidth=0.5)
            info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
            self.canvas.set_plot_label(index, info)
            self.redraw_pickers(file_path, index)
            self.redraw_chop(tr, tr.data, index)
            cf = cw.cf_lowpass()
            # Normalize
            # cf = cf / max(cf)
            t = t[0:len(cf)]
            # self.canvas.plot(t, cf, index, is_twinx=True, color="red",linewidth=0.5)
            # self.canvas.set_ylabel_twinx(index, "CWT (CF)")
            ax = self.canvas.get_axe(index)
            ax2 = ax.twinx()
            ax2.plot(t, cf, color="red", linewidth=0.5, alpha=0.5)
            if sharey:
                # Clean ticks on ax2
                ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
            # Update global y-limits for ax2
            cf_min, cf_max = min(cf), max(cf)
            global_cf_min = min(global_cf_min, cf_min)
            global_cf_max = max(global_cf_max, cf_max)
            #ax2.set_ylim(global_cf_min, global_cf_max)

            last_index = index
            tr_cf = tr.copy()
            tr_cf.data = cf
            tr_cf.times = t
            cfs.append(tr_cf)

        # After the loop, synchronize all ax2 y-limits
        if sharey:
            for index in range(len(self.st)):
                ax2 = self.canvas.get_axe(index).twinx()  # Get the corresponding ax2
                ax2.set_ylim(global_cf_min, global_cf_max)

                # Re-enable right-side ticks (if needed)
                ax2.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)

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

    def cwt_kurt(self):
        params = self.settings_dialog.getParameters()
        cycles = params["Num Cycles"]
        fmin = params["Fmin"]
        fmax = params["Fmax"]
        kurt_win = params["kurt_win"]
        sharey = params["amplitudeaxis"]
        global_cf_min = float('inf')
        global_cf_max = float('-inf')
        self.cf = []
        cfs = []
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(self.st), ncols=1, sharey=sharey)

        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)

        index = 0
        for index, file_path in enumerate(self.st):
            tr = self.st[index]
            t = tr.times("matplotlib")
            st_stats = ObspyUtil.get_stats_from_trace(tr)
            #YN1, CF1, cf, Tn, Nb, freqs = CFMB.compute_tf(tr, fmin=fmin, fmax=fmax, filter_npoles=4,
            #                    var_w=True, CF_type='kurtosis', CF_decay_win=4.0, hos_order=4, apply_taper=True)
            cf_kurt = CFKurtosis(self.st[index], 4, 4, fmin, fmax).run_kurtosis()[0]

            self.canvas.plot_date(t, tr.data, index, color="black", fmt='-', linewidth=0.5)
            info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
            self.canvas.set_plot_label(index, info)
            self.redraw_pickers(file_path, index)
            self.redraw_chop(tr, tr.data, index)

            ax = self.canvas.get_axe(index)
            ax2 = ax.twinx()
            ax2.plot(t, cf_kurt.data, color="red", linewidth=0.5, alpha=0.5)
            if sharey:
                # Clean ticks on ax2
                ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

            # Update global y-limits for ax2
            cf_min, cf_max = min(cf_kurt.data), max(cf_kurt.data)
            global_cf_min = min(global_cf_min, cf_min)
            global_cf_max = max(global_cf_max, cf_max)


            cfs.append(cf_kurt)

        # After the loop, synchronize all ax2 y-limits
        if sharey:
            for index in range(len(self.st)):
                ax2 = self.canvas.get_axe(index).twinx()  # Get the corresponding ax2
                ax2.set_ylim(global_cf_min, global_cf_max)

                # Re-enable right-side ticks (if needed)
                ax2.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)

        ax = self.canvas.get_axe(index)
        try:
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(index, "Date")
        except:
            pass
        self.cf = Stream(traces=cfs)

    def envelope(self):

        params = self.settings_dialog.getParameters()
        sharey = params["amplitudeaxis"]
        self.cf = []
        cfs = []
        #files_at_page = self.get_files_at_page()
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(self.st), ncols=1, sharey=sharey)
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        for index, file_path in enumerate(self.st):
            tr = self.st[index]
            t = tr.times("matplotlib")
            st_stats = ObspyUtil.get_stats_from_trace(tr)
            cf = envelope(tr.data, tr.stats.sampling_rate)
            self.canvas.plot_date(t, tr.data, index, color="black", fmt='-', linewidth=0.5)
            self.canvas.plot_date(t, cf, index, color="blue", clear_plot=False, fmt='-', linewidth=0.5, alpha=0.5)
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
        sharey = params["amplitudeaxis"]
        global_cf_min = float('inf')
        global_cf_max = float('-inf')

        self.cf = []
        cfs = []
        files_at_page = self.get_files_at_page()

        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1, sharey=sharey)
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        for index, file_path in enumerate(files_at_page):
            if self.cancelled:
                return

            tr = self.st[index]
            t = tr.times("matplotlib")

            delta = tr.stats.delta
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
            if sharey:
                # Clean ticks on ax2
                ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
            # Update global y-limits for ax2
            cf_min, cf_max = min(cf), max(cf)
            global_cf_min = min(global_cf_min, cf_min)
            global_cf_max = max(global_cf_max, cf_max)
            #ax2.set_ylim(global_cf_min, global_cf_max)

            last_index = index
            tr_cf = tr.copy()
            tr_cf.data = cf
            tr_cf.times = t_entropy
            cfs.append(tr_cf)
            self.value_entropy_init.emit(index + 1)

        # After the loop, synchronize all ax2 y-limits
        if sharey:
            for index in range(len(files_at_page)):
                ax2 = self.canvas.get_axe(index).twinx()  # Get the corresponding ax2
                ax2.set_ylim(global_cf_min, global_cf_max)

                # Re-enable right-side ticks (if needed)
                ax2.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)

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

    # def write_files_page(self):
    #
    #     root_path = os.path.dirname(os.path.abspath(__file__))
    #     if "darwin" == platform:
    #         dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
    #     else:
    #         dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
    #                                                        pw.QFileDialog.DontUseNativeDialog)
    #     if self.st:
    #         n = len(self.st)
    #         for j in range(n):
    #             try:
    #                 tr = self.st[j]
    #                 t1 = tr.stats.starttime
    #                 id = tr.id + "." + "D" + "." + str(t1.year) + "." + str(t1.julday)
    #                 print(tr.id, "Writing data processed")
    #                 path_output = os.path.join(dir_path, id)
    #                 tr.write(path_output, format="MSEED")
    #             except:
    #                 print("File cannot be written:", self.files_at_page[j])

    def write_files_page(self):

        errors = False
        root_path = os.path.dirname(os.path.abspath(__file__))

        # Get output directory from user
        if platform.system() == "Darwin":
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                           pw.QFileDialog.DontUseNativeDialog)

        # Check if user canceled directory selection
        if not dir_path:
            print("No directory selected, operation cancelled.")
            return

        if self.st:
            md = MessageDialog(self)
            for j, tr in enumerate(self.st):
                try:
                    t1 = tr.stats.starttime
                    base_name = f"{tr.id}.D.{t1.year}.{t1.julday}"
                    path_output = os.path.join(dir_path, base_name)

                    # Check if file exists and append a number if necessary
                    counter = 1
                    while os.path.exists(path_output):
                        path_output = os.path.join(dir_path, f"{base_name}_{counter}")
                        counter += 1

                    print(f"{tr.id} - Writing processed data to {path_output}")
                    tr.write(path_output, format="MSEED")

                except Exception as e:
                    errors = True
                    print(f"File cannot be written: {self.files_at_page[j]}, Error: {e}")

            if errors:
                md.set_info_message("Writting Complete with Errors, check output", dir_path)
            else:
                md.set_info_message("Writting Complete, check output", dir_path)

    def save_cf(self):

        errors = False
        root_path = os.path.dirname(os.path.abspath(__file__))

        # Get output directory from user
        if platform.system() == "Darwin":
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                           pw.QFileDialog.DontUseNativeDialog)

        # Check if user canceled directory selection
        if not dir_path:
            print("No directory selected, operation cancelled.")
            return

        if self.st:
            md = MessageDialog(self)
            for j, tr in enumerate(self.cf):
                try:
                    t1 = tr.stats.starttime
                    base_name = f"{tr.id}.M.{t1.year}.{t1.julday}"
                    path_output = os.path.join(dir_path, base_name)

                    # Check if file exists and append a number if necessary
                    counter = 1
                    while os.path.exists(path_output):
                        path_output = os.path.join(dir_path, f"{base_name}_{counter}")
                        counter += 1

                    print(f"{tr.id} - Writing processed data to {path_output}")
                    tr.write(path_output, format="MSEED")

                except Exception as e:
                    errors = True
                    print(f"File cannot be written: {self.files_at_page[j]}, Error: {e}")

            if errors:
                md.set_info_message("Writting Complete with Errors, check output", dir_path)
            else:
                md.set_info_message("Writting Complete, check output", dir_path)

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
        stack = wavenumber.stack(stream_stack, stack_type=params["stack type"])

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
        colors = ['black', 'indianred', 'chocolate', 'darkorange', 'olivedrab', 'lightseagreen',
                  'royalblue', 'darkorchid', 'magenta']
        ##
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)

        if len(self.st) > 0:
            i = 0
            for tr in self.st:
                if len(tr) > 0:
                    t = tr.times("matplotlib")
                    s = tr.data
                    if len(self.st) < 10:
                        self.canvas.plot_date(t, s, index, clear_plot=False, color=colors[i], fmt='-', alpha=0.5,
                                              linewidth=0.5, label=tr.id)

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

    def plot_map_stations(self):

        try:
            [lat, lon] = [self.event_info.latitude, self.event_info.longitude]
            # obsfiles = self.files_path

            map_dict = {}
            sd = []

            for tr in self.st:
                # st = SeismogramDataAdvanced(file)

                # name = st.stats.Network+"."+st.stats.Station
                name = tr.stats.network + "." + tr.stats.station
                sd.append(name)

                # st_coordinates = self.__metadata_manager.extract_coordinates(self.inventory, file)
                st_coordinates = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, tr)
                map_dict[name] = [st_coordinates.Latitude, st_coordinates.Longitude]

            self.map_stations = StationsMap(map_dict)
            self.map_stations.plot_stations_map(latitude=lat, longitude=lon)

        except:
            md = MessageDialog(self)
            md.set_error_message("couldn't plot stations map, please check your metadata and the trace headers")

    def redraw_pickers(self, file_name, axe_index):

        picked_at = {key: values for key, values in self.picked_at.items()}  # copy the dictionary.
        for key, value in picked_at.items():
            ps: PickerStructure = value
            if file_name == ps.FileName:
                # new_line = self.canvas.draw_arrow(ps.XPosition, axe_index, ps.Label,
                #                                  amplitude=ps.Amplitude, color=ps.Color, picker=True)
                new_line = self.canvas.draw_arrow(ps.XPosition, axe_index, ps.Label, amplitude=ps.Amplitude,
                                                  color=ps.Color, picker=True)
                # picked_at.pop(key)
                self.picked_at.pop(key)
                self.picked_at[str(new_line)] = ps

    def redraw_event_times(self, index):
        if len(self.events_times) > 0:
            for k in self.events_times:
                k = k.matplotlib_date
                self.canvas.draw_arrow(k, index, "Event Detected", color="blue", linestyles='--', picker=False)

    def redraw_chop(self, tr, s, ax_index):
        self.kind_wave = self.ChopCB.currentText()
        for key, value in self.chop.items():
            if tr.id in self.chop[key]:
                t = self.chop[key][tr.id][1]
                data = self.chop[key][tr.id][2]
                xmin_index = self.chop[key][tr.id][3]
                xmax_index = self.chop[key][tr.id][4]
                # data = s[xmin_index:xmax_index]
                self.chop[key][tr.id][2] = data
                self.canvas.plot_date(t, data, ax_index, clear_plot=False, color=self.color[key],
                                      fmt='-', linewidth=0.5)

    def on_click_matplotlib(self, event, canvas):
        if isinstance(canvas, MatplotlibCanvas):
            polarity, color = map_polarity_from_pressed_key(event.key)
            phase = self.comboBox_phases.currentText()
            # click_at_index = event.inaxes.rowNum
            click_at_index = self.ax_num
            x1, y1 = event.xdata, event.ydata
            # x2, y2 = event.x, event.y
            stats = ObspyUtil.get_stats(self.get_file_at_index(click_at_index))
            # Get amplitude from index
            # x_index = int(round(x1 * stats.Sampling_rate))  # index of x-axes time * sample_rate.
            # amplitude = canvas.get_ydata(click_at_index).item(x_index)  # get y-data from index.
            # amplitude = y1
            label = "{} {}".format(phase, polarity)
            tt = UTCDateTime(mdt.num2date(x1))
            diff = tt - self.st[self.ax_num].stats.starttime
            t = stats.StartTime + diff
            if self.decimator[0] is not None:
                idx_amplitude = int(self.decimator[0] * diff)
            else:
                idx_amplitude = int(stats.Sampling_rate * diff)
            amplitudes = self.st[self.ax_num].data
            amplitude = amplitudes[idx_amplitude]
            uncertainty = self.uncertainities.getUncertainity()

            line = canvas.draw_arrow(x1, click_at_index, label, amplitude=amplitude, color=color, picker=True)
            self.lines.append(line)
            self.picked_at[str(line)] = PickerStructure(tt, stats.Station, x1, uncertainty, amplitude, color, label,
                                                        self.get_file_at_index(click_at_index))
            # print(self.picked_at)
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
            self.remove_picker_structure(picker_structure.Time, picker_structure.Station)

    def remove_picker_structure(self, time: UTCDateTime, station: str):
        """
        Remove a PickerStructure from a dictionary based on the specified time and station.

        :param pickers: Dictionary containing PickerStructure objects as values.
        :param time: The UTCDateTime of the picker to be removed.
        :param station: The station name of the picker to be removed.
        :return: None
        """
        keys_to_remove = [
            key for key, picker in self.picked_at.items()
            if picker.Time == time and picker.Station == station
        ]

        if not keys_to_remove:
            raise ValueError(f"No PickerStructure found for time: {time} and station: {station}")

        for key in keys_to_remove:
            self.picked_at.pop(key)
            self.removed_picks.append([station, time])

    def on_click_plot_record_section(self, event_time: UTCDateTime, lat: float, long: float, depth: float):

        if self.sortCB.isChecked() and self.trimCB.isChecked():
            dist_all = []
            params = self.settings_dialog.getParameters()
            phases = params["prs_phases"]
            distance_in_km = True
            depth = self.event_info.event_depth
            otime = self.event_info.event_time

            self.canvas.clear()
            self.canvas.set_new_subplot(nrows=1, ncols=1)

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
                                dist, _, _ = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude,
                                                              self.event_info.latitude,
                                                              self.event_info.longitude)
                                dist_all.append(dist / 1000)

                        max_dist = kilometers2degrees(max(dist_all))
                        min_dist = kilometers2degrees(min(dist_all))
                        dist_all.sort()
                        dist_all.reverse()

                        if sum(dist_all) / len(dist_all) > 700:
                            distance_in_km = False
                            for index, dist in enumerate(dist_all):
                                dist_all[index] = kilometers2degrees(dist)
                            max_dist = max(dist_all)
                            min_dist = min(dist_all)
                        self.message_dataless_not_found()

                        arrivals = ObspyUtil.get_trip_times(source_depth=depth, min_dist=min_dist, max_dist=max_dist,
                                                            phases=phases)

                        all_arrivals = ObspyUtil.convert_travel_times(arrivals, otime,
                                                                      dist_km=distance_in_km)

                i = 0
                for tr in self.st:
                    if len(tr) > 0:
                        try:
                            t = tr.times("matplotlib")
                            if distance_in_km:
                                s = 5 * (tr.data / np.max(tr.data)) + dist_all[i]
                            else:
                                s = 0.5 * (tr.data / np.max(tr.data)) + dist_all[i]

                            self.canvas.plot_date(t, s, 0, clear_plot=False, color="black", fmt='-', alpha=0.5,
                                                  linewidth=0.5, label="")
                        except:
                            pass
                        i = i + 1

            for key in all_arrivals:
                self.canvas.plot_date(all_arrivals[key]["times"], all_arrivals[key]["distances"], 0,
                                      clear_plot=False, fmt='-', alpha=0.5, linewidth=1.0, label=str(key))

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

            # self.actionPlot_Record_Section.setDisabled(True)
        else:
            md = MessageDialog(self)
            md.set_warning_message("This action is only allowed if you check trim, sort and plot your waveforms "
                                   "by distance")

    def on_click_plot_arrivals(self, event_time: UTCDateTime, lat: float, long: float, depth: float):
        self.event_info.clear_arrivals()
        check_warning = False
        for index, file_path in enumerate(self.get_files_at_page()):
            # st_stats = self.dataless_manager.get_station_stats_by_mseed_file(file_path)
            st_stats = self.__metadata_manager.extract_coordinates(self.inventory, file_path)
            if st_stats is not None:
                self.event_info.plot_arrivals(index, st_stats)
            elif st_stats is None:
                check_warning = True

        if check_warning:
            md = MessageDialog(self)
            md.set_warning_message("Check your Metadata, some traces does't match with your metadata info")

    def get_arrivals_tf(self):

        try:
            st_stats = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, self.tr_tf)
            self.phases, self.travel_times = self.event_info.get_station_travel_times(st_stats)
            self.phases = list(self.phases)
            self.travel_times = list(self.travel_times)
            eventtime = self.event_info.event_time
            for index, time in enumerate(self.travel_times):
                self.travel_times[index] = (eventtime - self.tr_tf.stats.starttime) + self.travel_times[index]
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

            station = [st.stats.Network, st.stats.Station, st.stats.Location, st.stats.Channel, st.stats.StartTime,
                       st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]

            sd.append(station)

        self._stations_info = StationsInfo(sd)
        self._stations_info.show()

    def cross(self):

        if self.trimCB.isChecked():
            self.cf = []
            cfs = []
            max_values = []
            params = self.settings_dialog.getParameters()
            sharey = params["amplitudeaxis"]
            # files_at_page = self.get_files_at_page()
            self.canvas.clear()
            self.canvas.set_new_subplot(nrows=len(self.st), ncols=1, sharey=sharey)

            try:
                if len(self.st) > 0 and self.trimCB.isChecked():
                    num = self.crossSB.value()
                    template = self.st[num-1]
                    # if num <= len(files_at_page):
                    #     template = self.st[num - 1]
                    #
                    # else:
                    #     template = self.st[0]

                    for j, tr in enumerate(self.st):
                        sampling_rates = []
                        if self.crossCB.currentText() == "Auto":
                            template = tr
                            temp_stats = ObspyUtil.get_stats_from_trace(template)
                            st_stats = ObspyUtil.get_stats_from_trace(tr)
                            info = "Auto-Correlation {}.{}.{}".format(st_stats['net'], st_stats['station'],
                                                                      st_stats['channel'])
                        else:
                            st_stats = ObspyUtil.get_stats_from_trace(tr)
                            temp_stats = ObspyUtil.get_stats_from_trace(template)
                            info = "Cross-Correlation {}.{}.{} --> {}.{}.{}".format(st_stats['net'], st_stats['station'],
                                                                                    st_stats['channel'],
                                                                                    temp_stats['net'],
                                                                                    temp_stats['station'],
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

                        values = [np.max(cc), np.min(cc)]
                        values = np.abs(values)

                        if values[0] > values[1]:
                            maximo = np.where(cc == np.max(cc))
                        else:
                            maximo = np.where(cc == np.min(cc))

                        max_values.append(maximo)
                        self.canvas.plot(get_lags(cc) / max_sampling_rates, cc, j, clear_plot=True,
                                         linewidth=0.5, color="black")

                        max_line = ((maximo[0][0]) / max_sampling_rates) - 0.5 * (len(cc) / max_sampling_rates)
                        print("Zero lag at: "+info+": ", max_line)
                        self.canvas.draw_arrow(max_line, j, "max lag", color="red", linestyles='-', picker=False)
                        self.canvas.set_plot_label(j, info)
                        ax = self.canvas.get_axe(j)
                        ax.set_xlim(min(get_lags(cc) / max_sampling_rates), max(get_lags(cc) / max_sampling_rates))
                        # saving
                        cfs.append(Trace(cc, stats))

                    self.canvas.set_xlabel(j, "Time [s] from zero lag")

                    self.cf = Stream(cfs)
            except:
                md = MessageDialog(self)
                md.set_warning_message("Check correlation template and trim time is checked")
        else:
            md = MessageDialog(self)
            md.set_warning_message("Check correlation template and trim time is checked")

    # TODO implement your logic here for multiple select
    # def on_multiple_select(self, ax_index, xmin, xmax):
    #     pass

    def on_select(self, ax_index, xmin, xmax):
        # xmin xmax matplotlib_date
        self.kind_wave = self.ChopCB.currentText()
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
        t_start_utc = UTCDateTime(mdt.num2date(t[0]))
        t_end_utc = UTCDateTime(mdt.num2date(t[-1]))
        self.canvas.plot_date(t, s, ax_index, clear_plot=False, color=self.color[self.kind_wave],
                              fmt='-', linewidth=0.5)
        id = {id: [metadata, t, s, xmin_index, xmax_index, t_start_utc, t_end_utc]}
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
            t_start_utc = UTCDateTime(mdt.num2date(t[0]))
            t_end_utc = UTCDateTime(mdt.num2date(t[-1]))
            self.canvas.plot_date(t, s, ax_index, clear_plot=False, color=self.color[self.kind_wave], fmt='-',
                                  linewidth=0.5)
            id = {id: [metadata, t, s, xmin_index, xmax_index, t_start_utc, t_end_utc]}
            self.chop[self.kind_wave].update(id)

    def enter_axes(self, event):
        self.ax_num = self.canvas.figure.axes.index(event.inaxes)

    def find_chop_by_ax(self, ax):
        identified_chop = None
        id = self.st[ax].id
        # for key, value in self.chop[self.kind_wave].items():
        if id in self.chop[self.kind_wave]:
            identified_chop = self.chop[self.kind_wave][id]
        else:
            pass
        return identified_chop, id

    def clean_events_detected(self):
        if len(self.events_times) > 0:
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
                self.canvas.draw_arrow(x1, index, arrow_label="et", color="purple", linestyles='--', picker=False)

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
            [identified_chop, id] = self.find_chop_by_ax(self.ax_num)
            data = identified_chop[2]
            delta = 1 / identified_chop[0][6]
            [spec, freq, amplitude] = spectrumelement(data, delta, id)
            self.spectrum = PlotToolsManager(id)
            self.spectrum.plot_spectrum(freq, spec, amplitude)

        if event.key == 'm':
            self.canvas.draw_selection_TF(self.ax_num)
            self.tr_tf = self.st[self.ax_num]
            self.get_arrivals_tf()

        if event.key == 'h':
            self.canvas.draw_selection(self.ax_num)
            self.special_selection.append(self.files_at_page[self.ax_num])

        if event.key == 'j':

            self.canvas.draw_selection(self.ax_num, check=False)
            if self.files_at_page[self.ax_num] in self.special_selection:
                self.special_selection.pop(self.ax_num)

        if event.key == 'f':
            self.kind_wave = self.ChopCB.currentText()
            id = ""
            self.spectrum = PlotToolsManager(id)
            self.spectrum.plot_spectrum_all(self.chop[self.kind_wave].items())

        if event.key == 'z':
            # Compute Multitaper Spectrogram
            tr = self.st[self.ax_num]
            ax = self.canvas.get_axe(self.ax_num)
            ax2 = ax.twinx()

            start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
            end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
            [identified_chop, id] = self.find_chop_by_ax(self.ax_num)
            starttime = identified_chop[5]
            endtime = identified_chop[6]
            # if we want plot full trace#

            tr.trim(starttime=starttime, endtime=endtime)

            # tini = tr.times("matplotlib")[0]
            # tend = tr.times("matplotlib")[1]

            data = tr.data
            t = tr.times("matplotlib")
            npts = len(data)
            fs = tr.stats.sampling_rate
            delta = tr.stats.delta
            fn = fs / 2

            # win = int(params["Win"]*delta)

            f_min = 1 / (npts * delta)
            f_min = 0.1  # just for now
            f_max = fn

            ## using wavelet transform
            cw = ConvolveWaveletScipy(tr)
            cw.setup_wavelet(wmin=6, wmax=6, tt=int(fs / f_min), fmin=f_min, fmax=f_max, nf=80,
                             use_wavelet="Complex Morlet", m=30, decimate=False)
            z = cw.scalogram_in_dbs()
            z = np.clip(z, a_min=-80, a_max=0)
            x, y = np.meshgrid(t, np.linspace(f_min, f_max, z.shape[0]))
            # z = z[:, tini:tend]

            ## Using MTspectrogram still under check -->need to adapt the z to the chop
            # self.spectrogram = PlotToolsManager(id)
            # [x,y,z] = self.spectrogram.compute_spectrogram_plot(data, win, delta, f_min, f_max, t)
            ##

            ax2.contourf(x, y, z, levels=50, cmap=plt.get_cmap("jet"), alpha=0.2)
            ax2.set_ylim(f_min, fn)
            t = t[0:len(x)]
            ax2.set_xlim(t[0], t[-1])
            ax2.set_ylabel('Frequency [ Hz]')

            tt = tr.times("matplotlib")
            data = tr.data
            self.canvas.plot_date(tt, data, self.ax_num, clear_plot=False, color='black', fmt='-', linewidth=0.5)
            auto_start = min(tt)
            auto_end = max(tt)

            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(auto_start), mdt.num2date(auto_end))

            ax.set_ylim(min(data), max(data))
            formatter = mdt.DateFormatter('%Y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
        if event.key in self.numbers:
            self.number_press(event.key, event)

    def number_press(self, number, event):

        polarity, color = map_polarity_from_pressed_key(event.key)
        if number == "1":
            phase = "P"
            polarity = "?"
        elif number == "2":
            phase = "P"
            polarity = "+"
        elif number == "3":
            phase = "P"
            polarity = "-"
        elif number == "4":
            phase = "S"
            polarity = "?"
        elif number == "5":
            phase = "S"
            polarity = "+"
        elif number == "6":
            phase = "S"
            polarity = "-"
        # click_at_index = event.inaxes.rowNum
        click_at_index = self.ax_num
        x1, y1 = event.xdata, event.ydata
        # x2, y2 = event.x, event.y
        stats = ObspyUtil.get_stats(self.get_file_at_index(click_at_index))
        # Get amplitude from index
        # x_index = int(round(x1 * stats.Sampling_rate))  # index of x-axes time * sample_rate.
        # amplitude = canvas.get_ydata(click_at_index).item(x_index)  # get y-data from index.
        # amplitude = y1
        label = "{} {}".format(phase, polarity)
        tt = UTCDateTime(mdt.num2date(x1))
        diff = tt - self.st[self.ax_num].stats.starttime
        t = stats.StartTime + diff
        if self.decimator[0] is not None:
            idx_amplitude = int(self.decimator[0] * diff)
        else:
            idx_amplitude = int(stats.Sampling_rate * diff)
        amplitudes = self.st[self.ax_num].data
        amplitude = amplitudes[idx_amplitude]
        uncertainty = self.uncertainities.getUncertainity()

        line = self.canvas.draw_arrow(x1, click_at_index, label, amplitude=amplitude, color=color, picker=True)
        self.lines.append(line)
        self.picked_at[str(line)] = PickerStructure(tt, stats.Station, x1, uncertainty, amplitude, color, label,
                                                    self.get_file_at_index(click_at_index))
        # print(self.picked_at)
        # Add pick data to file.
        self.pm.add_data(tt, uncertainty, amplitude, stats.Station, phase, Component=stats.Channel,
                         First_Motion=polarity)
        self.pm.save()  # maybe we can move this to when you press locate.

    def availability(self):
        MseedUtil.data_availability_new(self.files_path)

    # def open_magnitudes_calculator(self):
    #
    #
    #     if isinstance(self.inventory, Inventory):
    #         choice, ok = pw.QInputDialog.getItem(
    #             self, "Open Magnitude Estimator ",
    #             "Please How to set Hypocenter parameters:",
    #             ["Load last location", "Load other location"], 0, False
    #         )
    #
    #         if ok:
    #
    #             # if choice == "Manually after open Magnitude module":
    #             #     option = "manually"
    #             if choice == "Load last location":
    #                 option = "last"
    #             elif choice == "Load other location":
    #                 option = "other"
    #
    #             self._magnitude_calc = MagnitudeCalc(option, self.inventory, self.project, self.chop)
    #             self._magnitude_calc.show()
    #
    #         else:
    #             # If the user cancels the choice dialog, do nothing
    #             return
    #     else:
    #         # Show an error message if required conditions are not met
    #         md = MessageDialog(self)
    #         md.set_error_message(
    #             "Please review the following requirements before proceeding:",
    #             "1. Ensure you have choped the waveforms. \n"
    #             "2. Ensure selected the correct hypocenter option\n"
    #             "3. Load a valid inventory metadata."
    #         )


    def open_solutions(self):

        output_path = os.path.join(ROOT_DIR, 'earthquakeAnalysis', 'location_output', 'obs', 'output.txt')

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
            md.set_error_message(f"Couldn't open pick file: {str(e)}")


    def remove_picks(self):
        md = MessageDialog(self)
        output_path = os.path.join(PICKING_DIR, 'output.txt')

        try:
            self.pm.clear()
            self.picked_at = {}
            command = "{} {}".format('rm', output_path)
            exc_cmd(command, cwd=ROOT_DIR)
            self.canvas.remove_arrows(self.lines)
            self.pm = PickerManager()
            md.set_info_message("Removed picks from file")
        except:

            md.set_error_message("Coundn't remove pick file", "Please check path: " + output_path)

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
                files = glob.glob(os.path.join(output_path, "*"))
                for f in files:
                    os.remove(f)
            except:
                pass

            output_path = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'first_polarity')
            try:

                os.remove(os.path.join(output_path, "test.inp"))
                os.remove(os.path.join(output_path, "mechanism.out"))
                os.remove(os.path.join(output_path, "focmec.lst"))

            except:
                pass

            md.set_info_message("Ready for new location")
        except:
            md = MessageDialog(self)
            md.set_error_message("Coundn't remove location, please review ", output_path)

    def open_array_analysis(self):
        # Confirm if the user wants to export to Array Analysis
        answer = pw.QMessageBox.question(
            self, "Export Seismograms to Array Analysis",
            "Do you want to export the current seismograms to Array Analysis?"
        )

        if answer == pw.QMessageBox.Yes:
            # Check required conditions: stream, inventory, and trim checkbox
            if len(self.st) > 0 and isinstance(self.inventory, Inventory) and self.trimCB.isChecked():
                # Prompt user to choose between FK process or BackProjection
                choice, ok = pw.QInputDialog.getItem(
                    self, "Select Array Analysis Process",
                    "Please choose the analysis process:",
                    ["FK process", "BackProjection"], 0, False
                )

                # If the user made a valid selection
                if ok:
                    starttime = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
                    endtime = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
                    self.controller().open_array_window()

                    # Call process based on user's selection
                    if choice == "FK process":
                        self.controller().array_analysis_frame.process_fk(self.st, self.inventory, starttime, endtime)
                    elif choice == "BackProjection":
                        # self.controller().array_analysis_frame.process_backprojection(self.st, self.inventory,
                        # start_time, end_time)
                        # TODO CONEXION WITH BACKPROJECTION
                        md = MessageDialog(self)
                        md.set_info_message("This conexion is still not available, please go to directly the "
                                            "Backproection Module")
                else:
                    # If the user cancels the choice dialog, do nothing
                    return
            else:
                # Show an error message if required conditions are not met
                md = MessageDialog(self)
                md.set_error_message(
                    "Please review the following requirements before proceeding:",
                    "1. Ensure seismograms are loaded.\n"
                    "2. Ensure seismograms are trimmed.\n"
                    "3. Load a valid inventory metadata."
                )

    def open_moment_tensor(self):
        # Check required conditions: stream, inventory, and trim checkbox
        if len(self.st) > 0 and isinstance(self.inventory, Inventory) and self.trimCB.isChecked():
            choice, ok = pw.QInputDialog.getItem(
                self, "Export Seismogram to Moment Tensor Inversion "
                      "Module if you have selected a seismogram and metadata this will be automatically sent it",
                "Please How to set Hypocenter parameters:",
                ["Manually after open MTI module", "Load last location", "Load other location"], 0, False
            )

            if ok:
                option = "manually"
                starttime = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
                endtime = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
                self.controller().open_momentTensor_window()

                if choice == "Manually after open MTI module":
                    option = "manually"
                elif choice == "Load last location":
                    option = "last"
                elif choice == "Load other location":
                    option = "other"

                self.controller().moment_tensor_frame.send_mti(self.st, self.inventory, starttime, endtime, option)
            else:
                # If the user cancels the choice dialog, do nothing
                return
        else:
            # Show an error message if required conditions are not met
            md = MessageDialog(self)
            md.set_error_message(
                "Please review the following requirements before proceeding:",
                "1. Ensure seismograms are loaded.\n"
                "2. Ensure seismograms are trimmed.\n"
                "3. Load a valid inventory metadata."
            )

    def time_frequency_analysis(self):
        self.controller().open_seismogram_window()
        answer = pw.QMessageBox.question(self, "Export Seismogram to TF Analysis",
                                         "if you have selected a seismogram this will be automatically processed in TF Analysis module")

        if pw.QMessageBox.Yes == answer:
            if len(self.tr_tf) > 0 and self.phases != None and self.travel_times != None:

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
        set_qdatetime(otime + 900, self.dateTimeEdit_2)
        self.event_info.set_time(otime)
        self.event_info.set_coordinates([lat, lon, depth])

    def set_event_download_values(self, event_catalog):
        print("Set event values", event_catalog)
        otime = event_catalog[0]
        lat = event_catalog[1]
        lon = event_catalog[2]
        depth = event_catalog[3]

        set_qdatetime(otime, self.dateTimeEdit_1)
        set_qdatetime(otime + 900, self.dateTimeEdit_2)
        self.event_info.set_time(otime)
        self.event_info.set_coordinates([lat, lon, depth])

    def open_auto_pick(self):
        if self.trimCB.isChecked():
            starttime = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
            endtime = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        else:
            starttime = None
            endtime = None
        self.__autopick = Autopick(self.project_filtered, self.metadata_path_bind.value, starttime=starttime,
                                   endtime=endtime)
        self.__autopick.signal.connect(self.slot)
        self.__autopick.signal2.connect(self.slot2)
        self.__autopick.show()

    @pyqtSlot(bool)
    def slot(self, reset):
        self.import_pick_from_file(default=True, reset=reset)

    @pyqtSlot()
    def slot2(self):
        self.events_times = self.__autopick.final_filtered_results
        self.plot_seismogram()

    def open_locate(self):
        # Add buttons
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Ready to Locate your event")
        msg_box.setText("Have you reviewed your picks and polarities?")
        yes_button = msg_box.addButton(pw.QMessageBox.Yes)
        no_button = msg_box.addButton(pw.QMessageBox.No)

        # Set the default button
        msg_box.setDefaultButton(yes_button)

        # Execute the message box and get the response
        response = msg_box.exec_()

        # response = QMessageBox.question(self, "Ready to Locate your event",
        #                                 "Have you reviewed your picks and polarities?")

        if response == QMessageBox.Yes:

            if isinstance(self.metadata_path_bind.value, str) and os.path.exists(self.metadata_path_bind.value):

                # create surf project
                sp = SurfProject()
                if len(self.project_filtered) == 0:
                    sp.project = self.project
                    sp.data_files = self.files_path
                else:
                    sp.project = self.project_filtered
                    sp.data_files = self.files_path

                self.__locate = Locate(self.metadata_path_bind.value, sp)

                self.__locate.show()
            else:

                md = MessageDialog(self)
                md.set_error_message(
                    "You have not selected any metadata to be export to locate",
                    "1. Click on Load metadata\n"
                    "2. Ensure that the metadata file is a valid Inventory\n")



        else:
            # If the user cancels the choice dialog, do nothing
            return
