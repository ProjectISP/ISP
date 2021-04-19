import os

from obspy import UTCDateTime
from isp.DataProcessing import DatalessManager, SeismogramDataAdvanced
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import parse_excepts
from isp.Gui import pw, pqg, pyc
from isp.Gui.Frames import BaseFrame, UiRealTimeFrame, MessageDialog, MatplotlibCanvas
from isp.Gui.Frames.help_frame import HelpDoc
from isp.Gui.Frames.earth_model_viewer import EarthModelViewer
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.settings_dialog import SettingsDialog
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from isp.Utils import AsycTime
from obspy.clients.seedlink.easyseedlink import create_client
import matplotlib.dates as mdt
from datetime import datetime

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
        self.client = None
        self.stations_available = []
        self.data_dict = {}
        self.dataless_not_found = set()  # a set of mseed files that the dataless couldn't find.

        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_dataless_path)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=self.numTracesCB.value(), constrained_layout=False)
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
        self.RetrieveBtn.clicked.connect(self.retrieve_data)
        self.stopBtn.clicked.connect(self.stop)
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


    def handle_data(self, tr):

        s = [tr.stats.network, tr.stats.station, tr.stats.channel]
        key = ".".join(s)
        now = datetime.now()
        start_time = UTCDateTime(now) - self.timewindowSB.value()*60

        if key in self.data_dict.keys():
            # Update
            #if tr.stats.starttime-start_time<0:
                #Update and write after deadline
            #self.data_dict[key] = tr
            #if self.saveDataCB.isChecked():
            #        self.write_trace(tr)


            self.data_dict[key] = self.data_dict[key] + tr


        else:
            # insert New Key
            self.data_dict[key]= tr
            #self.data_dict[key].trim(starttime=tr.stats.starttime, endtime=tr.stats.endtime)

        self.plot_seismogram()

    @AsycTime.run_async()
    def retrieve_data(self, e):

        self.client = create_client(self.serverAddressForm.text(), on_data=self.handle_data)

        self.client.select_stream(self.netForm.text(), self.stationForm.text(), self.channelForm.text())

        self.client.run()



    def plot_seismogram(self):

        now = datetime.now()
        start_time = UTCDateTime(now)-self.timewindowSB.value()
        end_time = UTCDateTime(now)
        self.canvas.set_new_subplot(nrows=self.numTracesCB.value(), ncols=1)
        self.canvas.set_xlabel(self.numTracesCB.value()-1, "Date")
        ax = self.canvas.get_axe(self.numTracesCB.value()-1)
        ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
        index = 0
        for key, tr in self.data_dict.items():

            t = tr.times("matplotlib")
            s = tr.data
            info = "{}.{}.{}".format(tr.stats.network, tr.stats.station, tr.stats.channel)
            self.canvas.plot_date(t, s, index, color="black",  fmt = '-', linewidth=0.5)
            self.canvas.set_plot_label(index, info)
            ax = self.canvas.get_axe(index)


            #if index == self.numTracesCB.value()-1:
            #   ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            index = index + 1

        formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S')
        ax.xaxis.set_major_formatter(formatter)



        #pyc.QCoreApplication.instance().processEvents()


    def stop(self):
        self.client.close()

    def write_trace(self, tr):

        # TODO: this should append miniseed not to overwrite cat file1 file2 > file3
        t1 = tr.stats.starttime
        id = tr.id + "." + "D" + "." + str(t1.year) + "." + str(t1.julday)
        print(tr.id, "Writing data processed")
        path_output = os.path.join(self.outputPathLE.text(), id)
        tr.write(path_output, format="MSEED")

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
