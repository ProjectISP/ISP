import os.path
from PyQt5.QtCore import Qt
from obspy import Inventory
from surfquakecore.real.structures import RealConfig, GeographicFrame, GridSearch, TravelTimeGridSearch, ThresholdPicks
from isp import PICKING_DIR, POLARITY_NETWORK
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Gui import pyc, pw
from isp.Gui.Frames import BaseFrame, MessageDialog
from isp.Gui.Frames.uis_frames import UiAutopick
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.LocCore.plot_tool_loc import plot_real_map
from isp.PolarCap.cnnFirstPolarity import Polarity
from isp.Utils import AsycTime, obspy_utils, MseedUtil
from isp.Utils.os_utils import OSutils
from surfquakecore.phasenet.phasenet_handler import PhasenetISP, PhasenetUtils
from surfquakecore.real.real_core import RealCore
from sys import platform
from surfquakecore.project.surf_project import SurfProject
from isp.associate_events.coincidence_trigger import CoincidenceTrigger


@add_save_load()
class Autopick(BaseFrame, UiAutopick):

    signal = pyc.pyqtSignal()
    signal2 = pyc.pyqtSignal()
    def __init__(self, project, metadata_path, starttime=None, endtime=None):

        super(Autopick, self).__init__()
        self.setupUi(self)

        """
        Picking & Associate Event Frame

        :param params required to initialize the class:
        project: surfquake_project
        metadata_path: str
        """

        self.sp = project
        self.metadata_path = metadata_path
        self.starttime = starttime
        self.endtime = endtime
        self.inventory = None
        self.final_filtered_results = []
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


    def _get_trigger_parameters(self):
        self.trigger_parameters = {}

        if self.decimateCB.isChecked():
            decimate_sampling_rate = self.new_sampling_rateSB.value()
        else:
            decimate_sampling_rate = False

        if self.methodTrigger.currentText() == "Classic STA/LTA":
            method = "classicstalta"
            self.trigger_parameters["sta"] = self.staWinDB.value()
            self.trigger_parameters["lta"] = self.ltaWinDB.value()

        else:
            method = "kurtosis"

        self.trigger_parameters = {"method": method, "fmin": self.fminDB.value(), "fmax": self.fmaxDB.value(),
                                   "the_on": self.threshold_onDB.value(), "the_off": self.threshold_offDB.value(),
                                   "time_window": self.kurtosisTimeWindowDB.value(),
                                   "coincidence": self.NumCoincidenceSB.value(),
                                   "centroid_radio": self.clusterSizeDB.value(),
                                   "decimate_sampling_rate": decimate_sampling_rate}


    def run_trigger(self):

        if self.sp is None:
            md = MessageDialog(self)
            md.set_error_message("Metadata run Picking, Please load a project first")
        else:
            self.send_trigger()
            self.progress_dialog.exec()
            md = MessageDialog(self)
            md.set_info_message("Coincidence Trigger Done")


    @AsycTime.run_async()
    def send_trigger(self):

        sp = SurfProject()
        sp.project = self.sp

        self._get_trigger_parameters()
        if self.pick_file_bind.value == "":
            pick_file = None
        if self.trigger_outpath_bind.value == "":
            out_path = None
        else:
            out_path = os.path.join(self.trigger_outpath_bind.value, "triggered_picks.txt")

        if len(self.trigger_parameters)>0:
            ct = CoincidenceTrigger(project=sp, parameters=self.trigger_parameters)
            final_filtered_results, details = ct.optimized_project_processing(input_file=self.pick_file_bind.value,
                                            output_file=out_path)

            self.final_filtered_results = final_filtered_results
            self.send_signal2()
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

        if self.sp is None:
            md = MessageDialog(self)
            md.set_error_message("Metadata run Picking, Please load a project first")
        else:
            info = MseedUtil.get_project_basic_info(self.sp)
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

        phISP = PhasenetISP(self.sp, amplitude=True, min_p_prob=self.p_wave_picking_thresholdDB.value(),
                            min_s_prob=self.s_wave_picking_thresholdDB.value())

        picks = phISP.phasenet()
        picks_ = PhasenetUtils.split_picks(picks)

        PhasenetUtils.write_nlloc_format(picks_, self.picking_bind.value, starttime=self.starttime,
                                         endtime=self.endtime)
        PhasenetUtils.convert2real(picks_, self.picking_bind.value)
        PhasenetUtils.save_original_picks(picks_, self.picking_bind.value)

        file_picks = os.path.join(self.picking_bind.value, "nll_picks.txt")
        OSutils.copy_and_rename_file(file_picks, PICKING_DIR, "output.txt")
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)

    def send_signal(self):

        # Connect end of picking with Earthquake Analysis

        self.signal.emit()

    def send_signal2(self):

        # Connect end of picking with Earthquake Analysis

        self.signal2.emit()


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
        self.send_signal()


    @AsycTime.run_async()
    def send_polarities(self):

        arrivals_path = os.path.join(self.picking_bind.value, "nll_picks.txt")
        output_path = os.path.join(self.picking_bind.value, "nll_picks_polarities.txt")

        sp = SurfProject()
        sp.project = self.sp
        pol = Polarity(project=sp, model_path=POLARITY_NETWORK, arrivals_path=arrivals_path,
                       threshold=self.polaritiesProbDB.value(),
                       output_path=output_path)

        pol.optimized_project_processing_pol()
        # copy_and_rename_file(src_path, dest_dir, new_name)
        #OSutils.copy_and_rename_file(output_path, self.picking_bind.value, "nll_picks_polarities.txt")
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)