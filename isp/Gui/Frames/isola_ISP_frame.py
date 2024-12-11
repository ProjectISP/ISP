#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
isola_ISP_frame



:param : 
:type : 
:return: 
:rtype: 
"""

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QWidget, QVBoxLayout
from obspy.core.event import Origin
from isp import ROOT_DIR, ALL_LOCATIONS
from isp.Exceptions import InvalidFile
from isp.Gui import pw, qt
from isp.Gui.Frames import BaseFrame, MessageDialog, UiMomentTensor, MatplotlibFrame
from isp.Gui.Frames.crustal_model_parameters_frame import CrustalModelParametersFrame
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, add_save_load, convert_qdatetime_utcdatetime, set_qdatetime, \
    convert_qdatetime_datetime
from isp.Utils import MseedUtil, ObspyUtil
from isp.Utils.subprocess_utils import open_html_file, open_url
from isp.mti.mti_utilities import MTIManager
from isp.mti.class_isola_new import *
from obspy import Stream, UTCDateTime, Inventory
import platform

from surfquakecore.moment_tensor.mti_parse import read_isola_result, WriteMTI
from surfquakecore.moment_tensor.sq_isola_tools.sq_bayesian_isola import BayesianIsolaCore
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig, StationConfig, InversionParameters

@add_save_load()
class MTIFrame(BaseFrame, UiMomentTensor):

    def __init__(self):
        super(MTIFrame, self).__init__()
        self.setupUi(self)
        self.inventory = {}
        self._stations_info = {}
        self.stream = None
        self.stations_check = False
        self.url = 'https://projectisp.github.io/ISP_tutorial.github.io/mti/'
        self.scroll_area_widget.setWidgetResizable(True)

        # Binding
        self.earth_path_bind = BindPyqtObject(self.earth_model_path)
        self.output_path_bind = BindPyqtObject(self.MTI_output_path)

        self.earthModelMTIBtn.clicked.connect(lambda: self.on_click_select_file(self.earth_path_bind))
        self.outputDirectoryMTIBtn.clicked.connect(lambda: self.on_click_select_directory(self.output_path_bind))

        # actions
        self.stationSelectBtn.clicked.connect(lambda: self.stationsInfo())
        self.runInversionMTIBtn.clicked.connect(lambda: self.run_inversion())

    def get_inversion_parameters(self):

        parameters = {'output_directory': self.MTI_output_path.text(),
                      'earth_model': self.earth_model_path.text(),
                      'location_unc': self.HorizontalLocUncertainityMTIDB.value(),
                      'time_unc': self.timeUncertainityMTIDB.value(), 'depth_unc': self.depthUncertainityMTIDB.value(),
                      'deviatoric': self.deviatoricCB.isChecked(), 'covariance': self.covarianceCB.isChecked(),
                      'plot_save': self.savePlotsCB.isChecked(), 'rupture_velocity': self.ruptureVelMTIDB.value(),
                      'source_type': self.sourceTypeCB.currentText(), 'min_dist': self.minDistMTIDB.value(),
                      'max_dist': self.maxDistMTIDB.value(), 'fmin': self.freq_minMTI.value(),
                      'fmax': self.freq_maxMTI.value(), 'rms_thresh': self.rms_threshMTI.value(),
                      'max_num_stationsMTI0': self.max_num_stationsMTI.value(),
                      'source_duration': self.sourceTypeLenthgMTIDB.value(),
                      'max_number_stations': self.max_num_stationsMTI.value()}

        return parameters

    def run_inversion(self):


        parameters = self.get_inversion_parameters()

        mti_config = MomentTensorInversionConfig(
            origin_date=convert_qdatetime_datetime(self.origin_time),
            latitude=self.latDB.value(),
            longitude=self.lonDB.value(),
            depth_km=self.depthDB.value(),
            magnitude=self.magnitudeDB.value(),
            stations=[StationConfig(name=".", channels=["."])],
            inversion_parameters=InversionParameters(
                earth_model_file=parameters['earth_model'],
                location_unc=parameters['location_unc'],
                time_unc=parameters['time_unc'],
                depth_unc=parameters['depth_unc'],
                source_duration=parameters['source_duration'],
                rupture_velocity=parameters['rupture_velocity'],
                min_dist=parameters['min_dist'],
                max_dist=parameters['max_dist'],
                source_type=parameters['source_type'],
                deviatoric=parameters['deviatoric'],
                covariance=parameters['covariance'],
                max_number_stations=parameters['max_number_stations']

            ),
        )


        bic = BayesianIsolaCore(project=self.stream, inventory_file="", output_directory=self.output_path_bind.value)

        # # Run Inversion
        bic.run_inversion(mti_config=path_to_configfiles, map_stations=None)

    def on_click_select_file(self, bind: BindPyqtObject):
        file_path = pw.QFileDialog.getOpenFileName(self, 'Select Directory', bind.value)
        file_path = file_path[0]

        if file_path:
            bind.value = file_path

    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path

    def on_click_select_hyp_file(self):
        file_path = pw.QFileDialog.getOpenFileName(self, 'Select Directory', ALL_LOCATIONS)
        file_path = file_path[0]
        if isinstance(file_path, str):
            return file_path
        else:
            md = MessageDialog(self)
            md.set_info_message("No selected any file, please set hypocenter parameters manually")
            return None


    def load_event(self):
        hyp_file = self.on_click_select_hyp_file()
        if isinstance(hyp_file, str):
            origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file, modified=True)
            hyp_values = MTIManager.get_hyp_values(origin[0])
            self.__set_hyp(hyp_values)
            md = MessageDialog(self)
            md.set_info_message("Loaded information and set hypocenter parameters, please click stations info")

    def stationsInfo(self):

        self.stations_check = True
        md = MessageDialog(self)
        md.set_info_message("Select your Station/Channel", "Sorted by distance to the epicenter")
        parameters = self.get_inversion_parameters()
        lat = float(parameters['latitude'])
        lon = float(parameters['longitude'])

        if self.st:
            min_dist = self.min_distCB.value()
            max_dist = self.max_distCB.value()
            mt = MTIManager(self.st, self.inventory, lat, lon, min_dist, max_dist)
            [self.stream, self.deltas, self.stations_isola_path] = mt.get_stations_index()

        all_stations = []

        for stream in self.stream:
            for tr in stream:
                station = [tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel, tr.stats.starttime,
                           tr.stats.endtime, tr.stats.sampling_rate, tr.stats.npts]

                all_stations.append(station)

        self._stations_info = StationsInfo(all_stations, check=True)
        self._stations_info.show()

    def send_mti(self, stream: Stream, inventory: Inventory, starttime: UTCDateTime, endtime: UTCDateTime,
                 option: str):

        self.st = stream
        print(self.st)

        self.inventory = inventory
        self.starttime = starttime
        self.endttime = endtime
        self.stream_frame = MatplotlibFrame(self.st, type='normal')
        self.stream_frame.show()

        # next set the hypocenter parameters
        if option == "manually":
            md = MessageDialog(self)
            md.set_info_message("Loaded information, please set hypocenter parameters by yourself, "
                                "then click stations info")

        elif option == "last":
            hyp_file = os.path.join(ROOT_DIR, "earthquakeAnalisysis", "location_output", "loc", "last.hyp")
            origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file, modified=True)
            hyp_values = MTIManager.get_hyp_values(origin[0])
            self.__set_hyp(hyp_values)
            md = MessageDialog(self)
            md.set_info_message("Loaded information and set hypocenter parameters, please click stations info")

        elif option == "other":
            hyp_file = self.on_click_select_hyp_file()
            if isinstance(hyp_file, str):
                origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file, modified=True)
                hyp_values = MTIManager.get_hyp_values(origin[0])
                self.__set_hyp(hyp_values)
                md = MessageDialog(self)
                md.set_info_message("Loaded information and set hypocenter parameters, please click stations info")



class ImageLabel(QLabel):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path  # Store the image file path

    def mouseDoubleClickEvent(self, event):
        if event.button() == qt.LeftButton:
            # Open the image with the system's default image viewer
            try:
                if platform.system() == 'Darwin':

                    subprocess.run(['open', self.image_path])
                else:
                    subprocess.run(['xdg-open', self.image_path])

            except Exception as e:
                print(f"Error opening image: {e}")