import os
import tempfile
from obspy import UTCDateTime
from isp.DataProcessing import DatalessManager, SeismogramDataAdvanced
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Gui import pw, pqg, pyc
from isp.Gui.Frames import BaseFrame, UiRealTimeFrame, MessageDialog, MatplotlibCanvas
from isp.Gui.Frames.map_realtime_frame import MapRealTime
from isp.Gui.Frames.earth_model_viewer import EarthModelViewer
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.settings_dialog import SettingsDialog
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from isp.Utils import AsycTime
from obspy.clients.seedlink.easyseedlink import create_client
import matplotlib.dates as mdt
import datetime
import numpy as np
from sys import platform

from isp.Utils.subprocess_utils import open_url


class RealTimeFrame(BaseFrame, UiRealTimeFrame):

    def __init__(self):
        super(RealTimeFrame, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(pqg.QIcon(':\\icons\\map-icon.png'))
        self.widget_map = None
        self.settings_dialog = SettingsDialog(self)
        self.inventory = {}
        self.files = []
        self.events_times = []
        self.total_items = 0
        self.items_per_page = 1
        self.__dataless_manager = None
        self.__metadata_manager = None
        self.url = "https://projectisp.github.io/ISP_tutorial.github.io/nrt/"
        self.st = None
        self.client = None
        self.stations_available = []
        self.data_dict = {}
        self.dataless_not_found = set()  # a set of mseed files that the dataless couldn't find.
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=self.numTracesCB.value(), constrained_layout=False)
        self.canvas.figure.tight_layout()
        self.timer_outdated = pyc.QTimer()
        self.timer_outdated.setInterval(1000)  # 1 second
        self.timer_outdated.timeout.connect(self.outdated_stations)

        # Binding
        self.root_path_bind = BindPyqtObject(self.rootPathForm)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)

        # Bind

        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))

        #self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))

        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_metadata_file(self.metadata_path_bind))

        self.actionSet_Parameters.triggered.connect(lambda: self.open_parameters_settings())
        self.mapBtn.clicked.connect(self.show_map)
        # self.__metadata_manager = MetadataManager(self.dataless_path_bind.value)
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


        # shortcuts
    def open_help(self):
        open_url(self.url)

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
                md.set_info_message("Loaded Metadata Successfully.")

    def open_parameters_settings(self):
        self.parameters.show()

    def onChange_metadata_path(self, value):

        md = MessageDialog(self)
        try:

            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
            md.set_info_message("Loaded Metadata, please check your terminal for further details")

        except:

            md.set_error_message("Something went wrong. Please check your metada file is a correct one")

    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)

        if dir_path:
            bind.value = dir_path

    def on_click_select_metadata_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]

    def handle_data(self, tr):
        # If tr is not valid, discard it
        if not tr:
            return

        s = [tr.stats.network, tr.stats.station, tr.stats.channel]
        key = ".".join(s)
        outdated_tr = None

        if key in self.data_dict.keys():
            tr.data = np.float64(tr.data)
            current_tr = self.data_dict[key]
            # If received trace is in the same day as current, append received to current
            if current_tr.stats.endtime.julday == tr.stats.endtime.julday:
                self.data_dict[key] = self.data_dict[key] + tr

            else:
                # TODO: maybe the trace should have all the info and the save file only saves current day
                # TODO: so that way data is not erased on day change
                # If received trace has a part within current day, add it to current trace and save as outdated and
                # trim the trace to contain only next day's data
                if current_tr.stats.endtime.julday == tr.stats.starttime.julday:
                    day_start = datetime.datetime.combine(tr.stats.endtime.date, datetime.time())
                    last_day = day_start - datetime.timedelta(microseconds=1)
                    outdated_tr = current_tr + tr.slice(tr.stats.starttime, UTCDateTime(last_day), False)
                    tr.trim(UTCDateTime(day_start), tr.stats.endtime)
                # Set new day's trace as current
                self.data_dict[key] = tr

        else:
            # insert New Key
            tr.data = np.float64(tr.data)
            self.data_dict[key] = tr

        self.plot_seismogram()

        if self.widget_map is not None:
            station_list = self.get_station_info(tr)
            self.widget_map.plot_set_stations(station_list)

        if self.saveDataCB.isChecked():
            if outdated_tr is not None:
                self.write_trace(outdated_tr)
            self.write_trace(tr)

    def outdated_stations(self):
        # this method is run every t seconds to check and plot which stations do not send data
        outdated_stations = []
        outdated_traces = []
        # Outdated stations' traces will be erased from current list
        for key in list(self.data_dict.keys()):
            if self.data_dict[key].stats.endtime + 60 < UTCDateTime.now():
                outdated_traces.append(self.data_dict[key])
                outdated_stations.append(self.get_station_info(outdated_traces[-1]))
                del self.data_dict[key]

        if outdated_stations:
            self.widget_map.plot_unset_stations(outdated_stations)

        if self.saveDataCB.isChecked():
            for tr in outdated_traces:
                self.write_trace(tr)

    def seedlink_error(self, tr):

        print("seedlink_error")

    def terminate_data(self, tr):

        print("terminate_data")

    @AsycTime.run_async()
    def retrieve_data(self, e):
        # server address example: geofon.gfz-potsdam.de
        # DATA = NET 'WM', STATION 'SFS', CHANNEL 'BHZ'
        self.client = create_client(self.serverAddressForm.text(), on_data=self.handle_data,
                                    on_seedlink_error=self.seedlink_error, on_terminate=self.terminate_data)

        pyc.QMetaObject.invokeMethod(self.timer_outdated, 'start')

        for net in self.netForm.text().split(","):
            for sta in self.stationForm.text().split(","):
                for chn in self.channelForm.text().split(","):
                    self.client.select_stream(net, sta, chn)

        # self.client.on_data()
        self.client.run()
        # self.client.get_info(level="ALL")

    def plot_seismogram(self):
        # TODO: y axis should be independent for each subplot

        now = UTCDateTime.now()
        start_time = now - self.timewindowSB.value() * 60
        end_time = now + 30
        self.canvas.set_new_subplot(nrows=self.numTracesCB.value(), ncols=1, update=False)
        self.canvas.set_xlabel(self.numTracesCB.value() - 1, "Date", update=False)
        index = 0
        parameters = self.parameters.getParameters()
        for key, tr in self.data_dict.items():
            # sd = SeismogramDataAdvanced(file_path=None, stream=Stream(traces=tr), realtime=True)
            sd = SeismogramDataAdvanced(file_path=None, stream=tr, realtime=True)
            tr = sd.get_waveform_advanced(parameters, self.inventory)

            t = tr.times("matplotlib")
            s = tr.data
            info = "{}.{}.{}".format(tr.stats.network, tr.stats.station, tr.stats.channel)
            self.canvas.plot_date(t, s, index, clear_plot=False, update=False, color="black", fmt='-', linewidth=0.5)
            self.canvas.set_plot_label(index, info)
            ax = self.canvas.get_axe(index)

            if index == self.numTracesCB.value() - 1:
                ax.set_xlim(mdt.num2date(start_time.matplotlib_date), mdt.num2date(end_time.matplotlib_date))

            index = index + 1

            formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
        self.canvas.draw()
        # pyc.QCoreApplication.instance().processEvents()

    def stop(self):
        # TODO: not working. Maybe subclassing is necessary
        self.client.close()
        pyc.QMetaObject.invokeMethod(self.timer_outdated, 'stop')

    def write_trace(self, tr):
        t1 = tr.stats.starttime
        file_name = tr.id + "." + "D" + "." + str(t1.year) + "." + str(t1.julday)
        path_output = os.path.join(self.rootPathForm.text(), file_name)
        if os.path.exists(path_output):
            temp = tempfile.mkstemp()
            tr.write(temp[1], format="MSEED")
            with open(path_output, 'ab') as current:
                with open(temp[1], 'rb') as temp_bin:
                    current.write(temp_bin.read())
            os.remove(temp[1])
        else:
            tr.write(path_output, format="MSEED")

    def show_map(self):
        if self.inventory:
            self.widget_map = MapRealTime(self.inventory)
            try:
                self.widget_map.show()
            except:
                pass

    def get_station_info(self, tr):
        coordinates = {}
        net_ids = []
        sta_ids = []
        latitude = []
        longitude = []
        net_ids.append(tr.stats.network)
        sta_ids.append(tr.stats.station)
        coords = self.inventory.get_coordinates(tr.id)
        latitude.append(coords['latitude'])
        longitude.append(coords['longitude'])
        net_content = [net_ids, sta_ids, latitude, longitude]
        coordinates[tr.stats.network] = net_content

        return coordinates

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
