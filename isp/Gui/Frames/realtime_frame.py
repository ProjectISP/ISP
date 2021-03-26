
from isp.DataProcessing import DatalessManager, SeismogramDataAdvanced
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import parse_excepts
from isp.Gui import pw, pqg
from isp.Gui.Frames import BaseFrame, UiRealTimeFrame, Pagination, MessageDialog, EventInfoBox, \
    MatplotlibCanvas
from isp.Gui.Frames.help_frame import HelpDoc
from isp.Gui.Frames.earth_model_viewer import EarthModelViewer
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.settings_dialog import SettingsDialog
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from isp.Utils import AsycTime


class RealTimeFrame(BaseFrame, UiRealTimeFrame):

    def __init__(self):
        super(RealTimeFrame, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(pqg.QIcon(':\\icons\\map-icon.png'))

        self.settings_dialog = SettingsDialog(self)
        self.inventory = {}
        self.files = []
        self.events_times = []
        self.total_items = 0
        self.items_per_page = 1
        self.__dataless_manager = None
        self.__metadata_manager = None
        self.st = None
        self.dataless_not_found = set()  # a set of mseed files that the dataless couldn't find.

        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_dataless_path)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)

        # Bind buttons
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        self.stations_infoBtn.clicked.connect(self.stationsInfo)
        self.__metadata_manager = MetadataManager(self.dataless_path_bind.value)
        self.actionSet_Parameters.triggered.connect(lambda: self.open_parameters_settings())
        self.actionArray_Anlysis.triggered.connect(self.open_array_analysis)
        self.actionMoment_Tensor_Inversion.triggered.connect(self.open_moment_tensor)
        self.actionTime_Frequency_Analysis.triggered.connect(self.time_frequency_analysis)
        self.actionReceiver_Functions.triggered.connect(self.open_receiver_functions)
        self.actionOpen_Settings.triggered.connect(lambda: self.settings_dialog.show())
        self.actionOpen_Help.triggered.connect(lambda: self.open_help())

        # Parameters settings
        self.parameters = ParametersSettings()
        # Earth Model Viewer
        self.earthmodel = EarthModelViewer()
        # help Documentation
        self.help = HelpDoc()

        # shortcuts

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

    def onChange_dataless_path(self, value):
        self.__dataless_manager = DatalessManager(value)
        self.earthquake_location_frame.set_dataless_dir(value)

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    @AsycTime.run_async()
    def onChange_metadata_path(self, value):
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
        except:
            raise FileNotFoundError("The metadata is not valid")

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)

        if dir_path:
            bind.value = dir_path

    def stationsInfo(self):

        files_path = self.get_files(self.root_path_bind.value)
        if self.sortCB.isChecked():
            if self.comboBox_sort.currentText() == "Distance":
                files_path.sort(key=self.sort_by_distance_advance)
                self.message_dataless_not_found()

            elif self.comboBox_sort.currentText() == "Back Azimuth":
                files_path.sort(key=self.sort_by_baz_advance)
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

    # TODO: this should be generic to be invoked from other windows
    def open_array_analysis(self):
        self.controller().open_array_window()

    def open_moment_tensor(self):
        self.controller().open_momentTensor_window()

    def time_frequency_analysis(self):
        self.controller().open_seismogram_window()

    def open_receiver_functions(self):
        self.controller().open_receiverFunctions()

    def controller(self):
        from isp.Gui.controllers import Controller
        return Controller()
