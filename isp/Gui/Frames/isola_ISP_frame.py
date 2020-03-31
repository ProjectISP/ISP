from obspy import Stream
from isp.DataProcessing import SeismogramDataAdvanced
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import InvalidFile
from isp.Gui import pw
from isp.Gui.Frames import BaseFrame, MatplotlibCanvas, MessageDialog, UiMomentTensor, MatplotlibFrame
from isp.Gui.Frames.crustal_model_parameters_frame import CrustalModelParametersFrame
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, add_save_load, convert_qdatetime_utcdatetime
from isp.Utils import MseedUtil
import os
#import matplotlib.pyplot as plt
#import numpy as np

@add_save_load()
class MTIFrame(BaseFrame, UiMomentTensor):

    def __init__(self):
        super(MTIFrame, self).__init__()
        self.setupUi(self)

        super(MTIFrame, self).__init__()
        self.setupUi(self)
        self.__stations_dir = None
        self.__metadata_manager = None
        self.inventory = {}
        self._stations_info = {}

        # Binding
        self.root_path_bind = BindPyqtObject(self.rootPathForm)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.earth_path_bind =  BindPyqtObject(self.earth_modelPathForm)

        # Binds
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.datalessBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        self.earthmodelBtn.clicked.connect(lambda: self.on_click_select_directory(self.earth_path_bind))

        # Action Buttons
        self.actionSettings.triggered.connect(lambda: self.open_parameters_settings())
        self.plotBtn.clicked.connect(self.plot_seismograms)
        self.actionSettings.triggered.connect(lambda: self.open_parameters_settings())
        self.actionWrite.triggered.connect(self.write)
        self.actionEarth_Model.triggered.connect(lambda: self.open_earth_model())
        self.stationsBtn.clicked.connect(self.stationsInfo)
        self.earthmodelBtn.clicked.connect(self.read_earth_model)
        # Parameters settings
        self.parameters = ParametersSettings()
        self.earth_model = CrustalModelParametersFrame()

    def open_parameters_settings(self):
        self.parameters.show()

    def open_earth_model(self):
        self.earth_model.show()

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def message_dataless_not_found(self):
        if len(self.dataless_not_found) > 1:
            md = MessageDialog(self)
            md.set_info_message("Metadata not found.")
        else:
            for file in self.dataless_not_found:
                md = MessageDialog(self)
                md.set_info_message("Metadata for {} not found.".format(file))

    def validate_file(self):
        if not MseedUtil.is_valid_mseed(self.file_selector.file_path):
            msg = "The file {} is not a valid mseed. Please, choose a valid format". \
                format(self.file_selector.file_name)
            raise InvalidFile(msg)

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

    def read_earth_model(self):
        model = self.earth_model.getParametersWithFormat()
        print(model)

    def plot_seismograms(self):

        starttime = convert_qdatetime_utcdatetime(self.starttime_date)
        endtime = convert_qdatetime_utcdatetime(self.endtime_date)
        diff = endtime - starttime
        file_path = self.root_path_bind.value
        obsfiles = []

        for dirpath, _, filenames in os.walk(file_path):
            for f in filenames:
                if f != ".DS_Store":
                    obsfiles.append(os.path.abspath(os.path.join(dirpath, f)))
        obsfiles.sort()
        parameters = self.parameters.getParameters()
        all_traces = []

        for file in obsfiles:
            sd = SeismogramDataAdvanced(file)
            if self.trimCB.isChecked() and diff >= 0:
                tr = sd.get_waveform_advanced(parameters, self.inventory,
                                              filter_error_callback=self.filter_error_message,
                                              start_time=starttime, end_time=endtime)
            else:
                tr = sd.get_waveform_advanced(parameters, self.inventory,
                                              filter_error_callback=self.filter_error_message)

            all_traces.append(tr)

        self.st = Stream(traces=all_traces)
        self.stream_frame = MatplotlibFrame(self.st, type='normal')
        self.stream_frame.show()

    def stationsInfo(self):

        file_path = self.root_path_bind.value
        obsfiles = []

        for dirpath, _, filenames in os.walk(file_path):
            for f in filenames:
                if f != ".DS_Store":
                    obsfiles.append(os.path.abspath(os.path.join(dirpath, f)))
        obsfiles.sort()
        sd = []

        for file in obsfiles:

            st = SeismogramDataAdvanced(file)

            station = [st.stats.Network,st.stats.Station,st.stats.Location,st.stats.Channel,st.stats.StartTime,
                       st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]

            sd.append(station)

        self._stations_info = StationsInfo(sd)
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