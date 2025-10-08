import copy
import os.path
from PyQt5.QtCore import Qt
from obspy import Inventory
from surfquakecore.real.structures import RealConfig, GeographicFrame, GridSearch, TravelTimeGridSearch, ThresholdPicks
from isp import PICKING_DIR, POLARITY_NETWORK
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Gui import pyc, pw
from isp.Gui.Frames import BaseFrame, MessageDialog
from isp.Gui.Frames.uis_frames import UiAutopick
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject, convert_qdatetime_utcdatetime
from isp.LocCore.plot_tool_loc import plot_real_map
from isp.PolarCap.cnnFirstPolarity import Polarity
from isp.Utils import AsycTime, obspy_utils, MseedUtil
from isp.Utils.os_utils import OSutils
from surfquakecore.phasenet.phasenet_handler import PhasenetISP, PhasenetUtils
from surfquakecore.real.real_core import RealCore
from sys import platform
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.coincidence_trigger.structures import CoincidenceConfig, Kurtosis, STA_LTA, Cluster
from surfquakecore.coincidence_trigger.coincidence_trigger import CoincidenceTrigger
from datetime import datetime, timedelta

@add_save_load()
class Autopick(BaseFrame, UiAutopick):

    signal = pyc.pyqtSignal(bool)
    signal2 = pyc.pyqtSignal(bool)
    def __init__(self, project: SurfProject, metadata_path, starttime=None, endtime=None):

        super(Autopick, self).__init__()
        self.setupUi(self)

        """
        Picking & Associate Event Frame

        :param params required to initialize the class:
        project: surfquake_project
        metadata_path: str
        """

        self.project_filtered = None
        md = MessageDialog(self)
        if isinstance(project, SurfProject) and len(project.project) >0:
            self.project = project
            print(self.project)
            md.set_info_message("Project loaded, Ready to go!!!")

        else:
            md.set_info_message("Project NO loaded, First load a project in Earthquake Analysis")

        self.metadata_path = metadata_path
        self.starttime = starttime
        self.endtime = endtime
        self.inventory = None
        self.final_filtered_results = []

        ############# Project - Filter ###############
        # set project_path
        self.filterProjectBtn.clicked.connect(lambda: self._convert_project())

        ############# Phasent - Picking ##############
        self.picking_bind = BindPyqtObject(self.picking_LE, self.onChange_root_path)
        self.output_path_pickBtn.clicked.connect(lambda: self.on_click_select_directory(self.picking_bind))
        self.phasenetBtn.clicked.connect(self.run_phasenet)

        ############ REAL ###########################
        self.real_bind = BindPyqtObject(self.real_inputLE, self.onChange_root_path)
        self.real_picking_inputBtn.clicked.connect(lambda: self.on_click_select_directory(self.real_bind))
        self.real_output_bind = BindPyqtObject(self.output_realLE, self.onChange_root_path)
        self.output_realBtn.clicked.connect(lambda: self.on_click_select_directory(self.real_output_bind))
        self.realBtn.clicked.connect(self.run_real)
        self.plot_grid_stationsBtn.clicked.connect(self.plot_real_grid)

        #### Coincidence Trigger ####
        self.pick_file_bind = BindPyqtObject(self.trigger_inputLE)
        self.trigger_outpath_bind = BindPyqtObject(self.output_triggerLE)

        self.picking_triggerFileBtn.clicked.connect(lambda: self.on_click_select_file(self.pick_file_bind))
        self.output_triggerBtn.clicked.connect(lambda: self.on_click_select_directory(self.trigger_outpath_bind))
        self.realCoincidenceTrigger.clicked.connect(lambda: self.run_trigger())

        self.setDefaultPickPath.clicked.connect(self.setDefaultPick)

        self.snr_btn.toggled.connect(self._set_snr_method)
        self.kurtosis_btn.toggled.connect(self._set_kurt_method)
        self._set_snr_method()

        ### Polarities Determination ####
        self.runPolaritiesBtn.clicked.connect(lambda: self.run_polarities())

        # Dialog
        self.progress_dialog = pw.QProgressDialog(self)
        self.progress_dialog.setRange(0, 0)
        self.progress_dialog.setWindowTitle('Processing.....')
        self.progress_dialog.setLabelText('Please Wait')
        self.progress_dialog.setWindowIcon(self.windowIcon())
        self.progress_dialog.setWindowTitle(self.windowTitle())
        self.progress_dialog.close()

    def _convert_project(self):
        self.project.get_data_files()
        self.reload_current_project()

    def _set_kurt_method(self):
        self.staWinDB.setEnabled(False)
        self.ltaWinDB.setEnabled(False)
        self.kurtosisTimeWindowDB.setEnabled(True)

    def _set_snr_method(self):
        self.staWinDB.setEnabled(True)
        self.ltaWinDB.setEnabled(True)
        self.kurtosisTimeWindowDB.setEnabled(False)

    def _get_trigger_parameters(self):

        if self.snr_btn.isChecked():
            method = "SNR"
        else:
            method = "Kurtosis"

        config = CoincidenceConfig(
            kurtosis_configuration=Kurtosis(CF_decay_win=self.kurtosisTimeWindowDB.value()),
            sta_lta_configuration=STA_LTA(method="classic", sta_win=self.staWinDB.value(), lta_win=self.ltaWinDB.value()),
            cluster_configuration=Cluster(
                method_preferred=method,
                centroid_radio=self.clusterSizeDB.value(),
                coincidence=self.NumCoincidenceSB.value(),
                threshold_on=self.threshold_onDB.value(),
                threshold_off=self.threshold_offDB.value(),
                fmin=self.fminDB.value(),
                fmax=self.fmaxDB.value()
            )
        )

        return config

    def parse_datetime(self, dt_str: str) -> datetime:
        # try with microseconds, fall back if not present
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Date string not in expected format: {dt_str}")

    def run_trigger(self):

        if self.project is None and self.project_filtered:
            md = MessageDialog(self)
            md.set_error_message("Metadata run Picking, Please load a project first")
        else:
            self.send_trigger()
            self.progress_dialog.exec()
            md = MessageDialog(self)
            md.set_info_message("Coincidence Trigger Done, Review Files coincidence_*.txt ")

    @AsycTime.run_async()
    def send_trigger(self):

        if self.project_filtered:
            sp = self.project_filtered
        else:
            sp = self.project

        info = sp.get_project_basic_info()
        min_date = info["Start"]
        max_date = info["End"]
        span_seconds = 86400
        dt1 = self.parse_datetime(min_date)
        dt2 = self.parse_datetime(max_date)

        diff = abs(dt2 - dt1)
        if diff < timedelta(days=1):
            subprojects = [sp]

        else:
            print(f"[INFO] Splitting into subprojects every {span_seconds} seconds")
            subprojects = sp.split_by_time_spans(
                span_seconds=span_seconds,
                min_date=min_date,
                max_date=max_date,
                file_selection_mode="overlap_threshold",
                verbose=True)

        if self.pick_file_bind.value == "":
            picking_file = None
        else:
            picking_file = self.pick_file_bind.value

        config = self._get_trigger_parameters()

        ct = CoincidenceTrigger(subprojects, config, picking_file, self.trigger_outpath_bind.value,
                                plot=self.saveplotsCB.isChecked())
        self.final_filtered_results = ct.optimized_project_processing()

        if self.picking_bind.value != PICKING_DIR:
            destination = os.path.join(self.trigger_outpath_bind.value, "output.txt")
            origin_output = os.path.join(PICKING_DIR, "output.txt")
            OSutils.copy_and_rename_file(picking_file, PICKING_DIR, "output.txt")
            OSutils.create_symlink(origin_output, destination, overwrite=True)

        self.send_signal2(reset=True)
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)


    def on_click_select_file(self, bind: BindPyqtObject):
        file_path = pw.QFileDialog.getOpenFileName(self, 'Select File', bind.value)
        file_path = file_path[0]

        if file_path:
            bind.value = file_path

    def get_metadata(self):

        try:
            self.__metadata_manager = MetadataManager(self.metadata_path)
            self.inventory: Inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
        except:
            raise FileNotFoundError("The metadata is not valid")

    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        pass

    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path

    def _select_directory(self, bind: BindPyqtObject):

        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        return dir_path

    ## Picking ##

    def setDefaultPick(self):

        default_pick_file = os.path.join(PICKING_DIR, 'output.txt')

        self.pick_file_bind = BindPyqtObject(self.trigger_inputLE)
        output_real = os.path.join(PICKING_DIR, "associated_picks")

        self.picking_LE.setText(PICKING_DIR)
        self.trigger_inputLE.setText(default_pick_file)
        self.output_triggerLE.setText(output_real)
        self.real_inputLE.setText(PICKING_DIR)

        if os.path.isdir(output_real):
            self.output_realLE.setText(output_real)
        else:
            try:
                os.makedirs(output_real)
                self.output_realLE.setText(output_real)
            except Exception as error:
                print("An exception occurred:", error)


    def run_phasenet(self):

        if self.project is None and self.project_filtered is None:
            md = MessageDialog(self)
            md.set_error_message("Metadata run Picking, Please load a project first")
        else:
            if self.project_filtered:
                sp = self.project_filtered
            else:
                sp = self.project

            info = sp.get_project_basic_info()
            print("Networks: ", info["Networks"])
            print("Stations: ", info["Stations"])
            print("Channel: ", info["Channels"])
            print("Nume of Files: ", info["num_files"])
            print("Start: ", info["Start"])
            print("end: ", info["End"])

            self.send_phasenet()
            self.progress_dialog.exec()
            md = MessageDialog(self)
            md.set_info_message("Picking Done")
            self.send_signal()

    @AsycTime.run_async()
    def send_phasenet(self):
        print("Starting Picking")

        origin_output = os.path.join(PICKING_DIR, "output.txt")
        destination = os.path.join(self.picking_bind.value,"output.txt")

        if self.project_filtered:
            project = self.project_filterd.project
        else:
            project = self.project.project

        phISP = PhasenetISP(project, amplitude=True, min_p_prob=self.p_wave_picking_thresholdDB.value(),
                            min_s_prob=self.s_wave_picking_thresholdDB.value())

        picks = phISP.phasenet()
        picks_ = PhasenetUtils.split_picks(picks)

        PhasenetUtils.write_nlloc_format(picks_, self.picking_bind.value, starttime=self.starttime,
                                         endtime=self.endtime)
        PhasenetUtils.convert2real(picks_, self.picking_bind.value)
        PhasenetUtils.save_original_picks(picks_, self.picking_bind.value)

        file_picks = os.path.join(self.picking_bind.value, "nll_picks.txt")
        OSutils.copy_and_rename_file(file_picks, PICKING_DIR, "output.txt")
        if origin_output != destination:
            OSutils.create_symlink(origin_output, destination, overwrite=True)
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)

    def send_signal(self, reset=False):

        # Connect end of picking with Earthquake Analysis

        self.signal.emit(reset)

    def send_signal2(self, reset=False):

        # Connect end of picking with Earthquake Analysis

        self.signal2.emit(reset)


    ## End Picking ##

    def run_real(self):

        """ REAL """
        if self.inventory is None:
            self.get_metadata()

        self.send_real()
        self.progress_dialog.exec()
        md = MessageDialog(self)
        md.set_info_message("Association Done")


    @AsycTime.run_async()
    def send_real(self):

        real_config = RealConfig(
            geographic_frame=GeographicFrame(
                lat_ref_max=self.lat_refMaxSB.value(),
                lon_ref_max=self.lon_refMaxSB.value(),
                lat_ref_min=self.lat_refMinSB.value(),
                lon_ref_min=self.lon_refMin.value(),
                depth=self.depthSB.value()
            ),
            grid_search_parameters=GridSearch(
                horizontal_search_range=self.gridSearchParamHorizontalRangeBtn.value(),
                depth_search_range=self.DepthSearchParamHorizontalRangeBtn.value(),
                event_time_window=self.EventTimeWindow.value(),
                horizontal_search_grid_size=self.HorizontalGridSizeBtn.value(),
                depth_search_grid_size=self.DepthGridSizeBtn.value()),
            travel_time_grid_search=TravelTimeGridSearch(
                horizontal_range=self.TTHorizontalRangeBtn.value(),
                depth_range=self.TTDepthRangeBtn.value(),
                depth_grid_resolution_size=self.TTDepthGridSizeBtn.value(),
                horizontal_grid_resolution_size=self.TTHorizontalGridSizeBtn.value()),
            threshold_picks=ThresholdPicks(
                min_num_p_wave_picks=self.ThresholdPwaveSB.value(),
                min_num_s_wave_picks=self.ThresholdSwaveSB.value(),
                num_stations_recorded=self.number_stations_picksSB.value())
        )
        real_path_work = os.path.join(self.real_output_bind.value, "work_dir")

        if os.path.isdir(real_path_work):
            pass
        else:
            try:
                os.mkdir(real_path_work)
            except Exception as error:
                print("An exception occurred:", error)

        rc = RealCore(self.metadata_path, real_config, self.real_bind.value, real_path_work,
                      self.real_output_bind.value)
        rc.run_real()
        print("End of Events AssociationProcess, please see for results: ", self.real_output_bind.value)
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)

    def plot_real_grid(self):
        if self.inventory is None:
            self.get_metadata()
        print("Work in progress")
        lon_min = self.lon_refMin.value()
        lon_max = self.lon_refMaxSB.value()
        lat_max = self.lat_refMaxSB.value()
        lat_min = self.lat_refMinSB.value()
        x = [lon_min, lon_max, lon_max, lon_min, lon_min]
        y = [lat_max, lat_max, lat_min, lat_min, lat_max]
        area = x + y
        network = obspy_utils.ObspyUtil.stationsCoodsFromMeta(self.inventory)
        #network = obspy_utils.ObspyUtil.stationsCoodsFromMeta(self.inventory)
        plot_real_map(network, area=area)

    def run_polarities(self):

        """ Polarities """

        self.send_polarities()
        self.progress_dialog.exec()
        md = MessageDialog(self)
        md.set_info_message("Polarities determination Done")
        self.send_signal(reset=True)


    @AsycTime.run_async()
    def send_polarities(self):

        origin_pick_file = os.path.join(PICKING_DIR, "output.txt")
        target_file = os.path.join(self.picking_bind.value, "output.txt")


        pol = Polarity(project=self.project, model_path=POLARITY_NETWORK, arrivals_path=self.picking_bind.value,
                       threshold=self.polaritiesProbDB.value(),
                       output_path=origin_pick_file)

        pol.optimized_project_processing_pol()

        # copy_and_rename_file(src_path, dest_dir, new_name)
        OSutils.copy_and_rename_file(origin_pick_file, self.picking_bind.value, "nll_picks.txt")
        if self.picking_bind.value != PICKING_DIR:
            OSutils.create_symlink(origin_pick_file, target_file)
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)


    def reload_current_project(self):

        md = MessageDialog(self)

        self.get_now_files()

        if self.project_filtered:
            info = self.project_filtered.get_project_basic_info()

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


    def get_now_files(self):

        self.project_filtered = copy.deepcopy(self.project)
        start = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        diff = end - start
        selection = [self.netForm.text(), self.stationForm.text(), self.channelForm.text()]

        try:
            self.project_filtered.filter_project_keys(net=selection[0], station=selection[1], channel=selection[2])
        except:
            self.project_filtered = None

        if self.project_filtered and self.trimCB.isChecked() and diff > 0:
            try:
                self.project_filtered.filter_project_time(starttime=start, edntime=end)
            except:
                self.project_filtered = None
