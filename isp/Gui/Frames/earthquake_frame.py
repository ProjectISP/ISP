import matplotlib.dates as mdt
from obspy import UTCDateTime, Stream
from obspy.core.event import Origin
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.trigger import coincidence_trigger
from isp.DataProcessing import DatalessManager, SeismogramDataAdvanced, ConvolveWaveletScipy, ConvolveWavelet
from isp.DataProcessing.NeuralNetwork import CNNPicker
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.DataProcessing.plot_tools_manager import PlotToolsManager
from isp.Exceptions import parse_excepts
from isp.Gui import pw, pqg, pyc, qt
from isp.Gui.Frames import BaseFrame, UiEarthquakeAnalysisFrame, Pagination, MessageDialog, EventInfoBox, \
    MatplotlibCanvas 
from isp.Gui.Frames.earthquake_frame_tabs import Earthquake3CFrame, EarthquakeLocationFrame
from isp.Gui.Frames.open_magnitudes_calc import MagnitudeCalc
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.settings_dialog import SettingsDialog
from isp.Gui.Utils import map_polarity_from_pressed_key
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, convert_qdatetime_utcdatetime
from isp.Structures.structures import PickerStructure
from isp.Utils import MseedUtil, ObspyUtil, AsycTime
from isp.earthquakeAnalisysis import PickerManager
import numpy as np
import os
from isp import ROOT_DIR
import matplotlib.pyplot as plt
from isp.earthquakeAnalisysis.stations_map import StationsMap
from isp.seismogramInspector.signal_processing_advanced import spectrumelement, sta_lta, envelope


class PickerWorker(pyc.QThread):
    def __init__(self, progressbar, st, cnn, canvas):
        super(PickerWorker, self).__init__()
        self.progressbar = progressbar
        self.st = st
        self.cnn = cnn
        self.canvas = canvas

    def run(self):
        self._run_picker()

    def _run_picker(self):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        if self.st:
            stations = ObspyUtil.get_stations_from_stream(self.st)
            N = len(stations)
            pyc.QMetaObject.invokeMethod(self.progressbar, 'setValue', qt.AutoConnection, pyc.Q_ARG(int, 0))
            pyc.QMetaObject.invokeMethod(self.progressbar, 'setMaximum', qt.AutoConnection, pyc.Q_ARG(int, N))
            # TODO OPEN MP FOR MULTIPLE STATIONS
            index=0
            for station in stations:
                st2 = self.st.select(station=station)
                try:
                    maxstart = np.max([tr.stats.starttime for tr in st2])
                    minend = np.min([tr.stats.endtime for tr in st2])
                    st2.trim(maxstart, minend)
                    self.cnn.setup_stream(st2)  # set stream to use in prediction.
                    self.cnn.predict()
                    arrivals = self.cnn.get_arrivals()
                    for k , times in arrivals.items():
                        for t in times:
                            if k == "p":
                                self.canvas.draw_arrow(t.matplotlib_date, index + 2,
                                                       "P", color="blue", linestyles='--', picker=False)
                            if k == "s":
                                self.canvas.draw_arrow(t.matplotlib_date, index + 0,
                                                       "S", color="purple", linestyles='--', picker=False)
                                self.canvas.draw_arrow(t.matplotlib_date, index + 1,
                                                       "S", color="purple", linestyles='--', picker=False)

                except ValueError as e:
                    md = MessageDialog(self)
                    md.set_info_message("Prediction failed for station {}\n{}".format(station,e))

                index = index + 3
                pyc.QMetaObject.invokeMethod(self.progressbar, 'setValue', qt.AutoConnection, pyc.Q_ARG(int, index/3))

class EarthquakeAnalysisFrame(BaseFrame, UiEarthquakeAnalysisFrame):

    def __init__(self):
        super(EarthquakeAnalysisFrame, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Earthquake Location')
        self.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))

        try:
            self.cnn = CNNPicker()
        except:
            print("Neural Network cannot be loaded")

        self.progressbar = pw.QProgressDialog(self)
        self.progressbar.setWindowTitle('Neural Network Running')
        self.progressbar.setLabelText(" Computing Auto-Picking ")
        self.progressbar.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.progressbar.close()
        self.settings_dialog = SettingsDialog(self)
        self.inventory = {}
        self.files = []
        self.events_times = []
        self.total_items = 0
        self.items_per_page = 1
        # dict to keep track of picks-> dict(key: PickerStructure) as key we use the drawn line.
        self.picked_at = {}
        self.__dataless_manager = None
        self.__metadata_manager = None
        self.st = None
        self.chop = {'Body waves':{}, 'Surf Waves':{}, 'Coda':{}, 'Noise':{}}
        self.color={'Body waves':'orangered','Surf Waves':'blue','Coda':'grey','Noise':'green'}
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
        self.canvas.register_on_select(self.on_select, rectprops = dict(alpha=0.2, facecolor='red'))
        self.canvas.mpl_connect('key_press_event', self.key_pressed)
        self.canvas.mpl_connect('axes_enter_event', self.enter_axes)

        self.event_info = EventInfoBox(self.eventInfoWidget, self.canvas)
        self.event_info.register_plot_arrivals_click(self.on_click_plot_arrivals)

        self.earthquake_3c_frame = Earthquake3CFrame(self.parentWidget3C)
        self.earthquake_location_frame = EarthquakeLocationFrame(self.parentWidgetLocation)

        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_dataless_path)

        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)

        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        self.updateBtn.clicked.connect(self.plot_seismogram)
        self.stations_infoBtn.clicked.connect(self.stationsInfo)
        self.rotateBtn.clicked.connect(self.rotate)
        self.mapBtn.clicked.connect(self.plot_map_stations)
        self.__metadata_manager = MetadataManager(self.dataless_path_bind.value)
        self.actionSet_Parameters.triggered.connect(lambda: self.open_parameters_settings())
        self.actionWrite_Current_Page.triggered.connect(self.write_files_page)
        self.actionArray_Anlysis.triggered.connect(self.open_array_analysis)
        self.actionMoment_Tensor_Inversion.triggered.connect(self.open_moment_tensor)
        self.actionTime_Frequency_Analysis.triggered.connect(self.time_frequency_analysis)
        self.actionOpen_Magnitudes_Calculator.triggered.connect(self.open_magnitudes_calculator)
        self.actionSTA_LTA.triggered.connect(self.run_STA_LTA)
        self.actionCWT_CF.triggered.connect(self.cwt_cf)
        self.actionEnvelope.triggered.connect(self.envelope)
        self.actionReceiver_Functions.triggered.connect(self.open_receiver_functions)
        self.actionRun_picker.triggered.connect(self.picker_thread)
        self.actionRun_Event_Detector.triggered.connect(self.detect_events)
        self.actionOpen_Settings.triggered.connect(lambda : self.settings_dialog.show())
        self.pm = PickerManager()  # start PickerManager to save pick location to csv file.

        # Parameters settings

        self.parameters = ParametersSettings()

        # shortcuts
        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+L'), self)
        self.shortcut_open.activated.connect(self.open_parameters_settings)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+O'),self)
        self.shortcut_open.activated.connect(self.clean_chop_at_page)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+M'), self)
        self.shortcut_open.activated.connect(self.open_magnitudes_calculator)

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+P'),self)
        self.shortcut_open.activated.connect(self.comboBox_phases.showPopup)

    def open_parameters_settings(self):
        self.parameters.show()

    #@property
    #def event_info(self) -> EventInfoBox:
    #    return self.__event_info

    def picker_thread(self):
        w = PickerWorker(self.progressbar, self.st, self.cnn, self.canvas)
        self.progressbar.reset()
        w.start()
        self.progressbar.exec()
        # TODO METHOD TO READ ARRIVALS AND WRITE TO XML

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
        self.plot_seismogram()

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        files_path = self.get_files(value)
        self.set_pagination_files(files_path)

        # self.plot_seismogram()

    def set_pagination_files(self, files_path):
        self.files = files_path
        self.total_items = len(self.files)
        self.pagination.set_total_items(self.total_items)

    def get_files(self, dir_path):

        files_path = MseedUtil.get_mseed_files(dir_path)

        if self.selectCB.isChecked():
            selection = [self.netForm.text(), self.stationForm.text(), self.channelForm.text()]
            files_path = MseedUtil.get_selected_files(files_path, selection)

        return files_path


    def onChange_dataless_path(self, value):
        self.__dataless_manager = DatalessManager(value)
        self.earthquake_location_frame.set_dataless_dir(value)

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def onChange_metadata_path(self, value):
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
        except:
            raise FileNotFoundError("The metada is not valid")


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


    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)

        if dir_path:
            bind.value = dir_path

    def sort_by_distance_advance(self, file):

         st_stats = self.__metadata_manager.extract_coordinates(self.inventory, file)
         if st_stats:

             dist, _, _ = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                           self.event_info.longitude)
             # print("File, dist: ", file, dist)
             return dist
         else:
             self.dataless_not_found.add(file)
             print("No Metadata found for {} file.".format(file))
             return 0.

    def sort_by_baz_advance(self, file):

         st_stats = self.__metadata_manager.extract_coordinates(self.inventory, file)

         if st_stats:

             _, _, az_from_epi = gps2dist_azimuth(st_stats.Latitude, st_stats.Longitude, self.event_info.latitude,
                                          self.event_info.longitude)
             return az_from_epi
         else:

             self.dataless_not_found.add(file)
             print("No Metadata found for {} file.".format(file))
             return 0.


    def plot_seismogram(self):
        if self.st:
            del self.st

        self.canvas.clear()
        ##
        self.nums_clicks = 0
        all_traces = []
        files_path = self.get_files(self.root_path_bind.value)
        if self.sortCB.isChecked():
            if self.comboBox_sort.currentText() == "Distance":
                files_path.sort(key=self.sort_by_distance_advance)
                self.message_dataless_not_found()

        #
            elif self.comboBox_sort.currentText() == "Back Azimuth":
                files_path.sort(key=self.sort_by_baz_advance)
                self.message_dataless_not_found()

        self.set_pagination_files(files_path)
        files_at_page = self.get_files_at_page()
        ##
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        diff = end_time - start_time
        if len(self.canvas.axes) != len(files_at_page):
            self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        last_index = 0
        min_starttime = []
        max_endtime = []
        parameters = self.parameters.getParameters()
        for index, file_path in enumerate(files_at_page):

            sd = SeismogramDataAdvanced(file_path)

            if self.trimCB.isChecked() and diff >= 0:
                tr = sd.get_waveform_advanced(parameters, self.inventory,
                                              filter_error_callback=self.filter_error_message,
                                              start_time=start_time, end_time=end_time)
            else:

                tr = sd.get_waveform_advanced(parameters, self.inventory,
                                                   filter_error_callback=self.filter_error_message)
            if len(tr) > 0:
                t = tr.times("matplotlib")
                s = tr.data
                self.canvas.plot_date(t, s, index, color="black", fmt = '-', linewidth=0.5)
                self.redraw_pickers(file_path, index)
                #redraw_chop = 1 redraw chopped data, 2 update in case data chopped is midified
                self.redraw_chop(tr, s, index)
                self.redraw_event_times(index)

                last_index = index

                st_stats = ObspyUtil.get_stats(file_path)

                if st_stats and self.sortCB.isChecked() == False:
                    info = "{}.{}.{}".format(st_stats.Network, st_stats.Station, st_stats.Channel)
                    self.canvas.set_plot_label(index, info)

                elif st_stats and self.sortCB.isChecked() and self.comboBox_sort.currentText() == "Distance":

                    dist = self.sort_by_distance_advance(file_path)
                    dist = "{:.1f}".format(dist/1000)
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
                    min_starttime.append(min(t))
                    max_endtime.append(max(t))
                except:
                    print("Empty traces")

            all_traces.append(tr)

        self.st = Stream(traces=all_traces)
        try:
            if min_starttime and max_endtime is not None:
                auto_start = min(min_starttime)
                auto_end = max(max_endtime)
                self.auto_start = auto_start
                self.auto_end = auto_end

            ax = self.canvas.get_axe(last_index)
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(auto_start), mdt.num2date(auto_end))
            formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(last_index, "Date")
        except:
            pass

    # Rotate to GAC,only first version #
    def rotate(self):
        if self.st:
            self.canvas.clear()
            all_traces_rotated = []
            stations = ObspyUtil.get_stations_from_stream(self.st)
            start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
            end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
            for k in range(len(stations)):
                st1 = self.st.copy()
                #print("Computing", stations[k])
                st2 = st1.select(station=stations[k])
                try:
                    maxstart = np.max([tr.stats.starttime for tr in st2])
                    minend = np.min([tr.stats.endtime for tr in st2])
                    st2.trim(maxstart, minend)
                    tr = st2[0]
                    coordinates = self.__metadata_manager.extrac_coordinates_from_trace(self.inventory, tr)
                    [azim, bazim, inci] = ObspyUtil.coords2azbazinc(coordinates.Latitude,coordinates.Longitude,
                    coordinates.Elevation,self.event_info.latitude, self.event_info.longitude, self.event_info.event_depth)

                    st2.rotate(method='NE->RT', back_azimuth=bazim)
                except:
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
                formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
                ax.xaxis.set_major_formatter(formatter)
                self.canvas.set_xlabel(last_index, "Date")
            except:
                pass

    def run_STA_LTA(self):
        files_at_page = self.get_files_at_page()
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        params = self.settings_dialog.getParameters()
        print(params)
        for index, file_path in enumerate(files_at_page):

            tr = self.st[index]
            cf = sta_lta(tr.data, tr.stats.sampling_rate)
            st_stats = ObspyUtil.get_stats_from_trace(tr)
            # Normalize
            cf =cf/max(cf)
            # forward to be centered the peak
            start = tr.stats.starttime-0.5
            tr.stats.starttime = start
            t = tr.times("matplotlib")
            self.canvas.plot_date(t, tr.data, index, color="black", fmt='-', linewidth=0.5)
            info = "{}.{}.{}".format(st_stats['net'], st_stats['station'], st_stats['channel'])
            self.canvas.set_plot_label(index, info)
            self.redraw_pickers(file_path, index)
            self.redraw_chop(tr, tr.data, index)

            t = t[0:len(cf)]
            ax = self.canvas.get_axe(index)
            ax2 = ax.twinx()
            ax2.plot(t, cf,color="grey", linewidth=0.5, alpha = 0.5)
            ax2.set_ylim(-1.05, 1.05)
            last_index = index
        ax = self.canvas.get_axe(last_index)
        try:
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(last_index, "Date")
        except:
            pass

    def cwt_cf(self):
        files_at_page = self.get_files_at_page()
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        diff = end_time - start_time
        for index, file_path in enumerate(files_at_page):
            tr = self.st[index]
            cw = ConvolveWaveletScipy(tr)
            if self.trimCB.isChecked() and diff >= 0:
                cw.setup_wavelet(start_time, end_time, wmin=6, wmax=6, tt=10, fmin=0.2, fmax=10, nf=40, use_rfft=False,
                                 decimate=False)
            else:
                cw.setup_wavelet(wmin=6, wmax=6, tt=10, fmin=0.2, fmax=10, nf=40, use_rfft=False,
                                 decimate=False)

            # delay = cw.get_time_delay()
            f = np.logspace(np.log10(0.2), np.log10(10))
            k =6 / (2 * np.pi * f) #one standar deviation
            delay = np.mean(k)
            start=tr.stats.starttime+delay
            tr.stats.starttime=start
            t = tr.times("matplotlib")
            cf = cw.cf_lowpass()
            # Normalize
            cf = cf / max(cf)
            t=t[0:len(cf)]
            #self.canvas.plot(t, cf, index, is_twinx=True, color="red",linewidth=0.5)
            #self.canvas.set_ylabel_twinx(index, "CWT (CF)")
            ax = self.canvas.get_axe(index)
            ax2 = ax.twinx()
            ax2.plot(t, cf, color="red", linewidth=0.5)
            ax2.set_ylim(-1.05, 1.05)
            last_index = index
        ax = self.canvas.get_axe(last_index)
        try:
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(last_index, "Date")
        except:
            pass

    def envelope(self):
        files_at_page = self.get_files_at_page()
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        for index, file_path in enumerate(files_at_page):
            tr = self.st[index]
            t = tr.times("matplotlib")
            cf = envelope(tr.data,tr.stats.sampling_rate)
            self.canvas.plot(t, cf, index, clear_plot=False, color="blue", linewidth=0.5, alpha = 0.5)
            last_index = index

        ax = self.canvas.get_axe(last_index)
        try:
            if self.trimCB.isChecked():
                ax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
            else:
                ax.set_xlim(mdt.num2date(self.auto_start), mdt.num2date(self.auto_end))
            formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(last_index, "Date")
        except:
            pass


    # @AsycTime.run_async()
    def detect_events(self):

        all_traces = []

        parameters = self.parameters.getParameters()
        starttime = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        endtime = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        diff = endtime - starttime
        obsfiles = MseedUtil.get_mseed_files(self.root_path_bind.value)
        obsfiles.sort()

        for file in obsfiles:
            sd = SeismogramDataAdvanced(file)
            if self.trimCB.isChecked() and diff >= 0:
                tr = sd.get_waveform_advanced(parameters, self.inventory,
                                              filter_error_callback=self.filter_error_message,
                                              start_time=starttime, end_time=endtime)
                cw = ConvolveWaveletScipy(tr)
                cw.setup_wavelet(starttime, endtime, wmin=6, wmax=6, tt=10, fmin=0.2, fmax=10, nf=40, use_rfft=False,
                                 decimate=False)
            else:
                tr = sd.get_waveform_advanced(parameters, self.inventory,
                                              filter_error_callback=self.filter_error_message)
                cw = ConvolveWaveletScipy(tr)
                cw.setup_wavelet(wmin=6, wmax=6, tt=10, fmin=0.2, fmax=10, nf=40, use_rfft=False,
                                 decimate=False)

            cf = cw.cf_lowpass()
            # Normalize
            cf = cf / max(cf)
            tr.data = cf
            all_traces.append(tr)

        self.st = Stream(traces=all_traces)
        trigger =coincidence_trigger(trigger_type=None, thr_on = 0.8, thr_off = 0.4,
                                  thr_coincidence_sum = 5, stream=self.st, details=True)


        for k in range(len(trigger)):
            detection = trigger[k]
            for key in detection:

                if key == 'time':
                    time = detection[key]
                    self.events_times.append(time)

        md = MessageDialog(self)
        md.set_info_message("Events Detection done")
        self.plot_seismogram()


    def write_files_page(self):
        import os
        root_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        if self.st:
            n = len(self.st)
            for j in range(n):
                tr = self.st[j]
                t1 = tr.stats.starttime
                id = tr.id+"."+"D"+"."+str(t1.year)+"."+str(t1.julday)
                print(tr.id, "Writing data processed")
                path_output = os.path.join(dir_path, id)
                tr.write(path_output, format="MSEED")


    def plot_map_stations(self):

        obsfiles = MseedUtil.get_mseed_files(self.root_path_bind.value)
        obsfiles.sort()

        #files_at_page = self.get_files_at_page()
        map_dict={}
        sd = []

        for file in obsfiles:
            st = SeismogramDataAdvanced(file)

            name = st.stats.Network+"."+st.stats.Station

            sd.append(name)

            st_coordinates = self.__metadata_manager.extract_coordinates(self.inventory, file)

            map_dict[name] = [st_coordinates.Latitude, st_coordinates.Longitude]

        self.map_stations = StationsMap(map_dict)
        self.map_stations.plot_stations_map()

    def redraw_pickers(self, file_name, axe_index):

        picked_at = {key: values for key, values in self.picked_at.items()}  # copy the dictionary.
        for key, value in picked_at.items():
            ps: PickerStructure = value
            if file_name == ps.FileName:
                new_line = self.canvas.draw_arrow(ps.XPosition, axe_index, ps.Label,
                                                  amplitude=ps.Amplitude, color=ps.Color, picker=True)
                self.picked_at.pop(key)
                self.picked_at[str(new_line)] = ps

    def redraw_event_times(self, index):
        if len(self.events_times)>0:
            for k in self.events_times:
                k = k.matplotlib_date
                self.canvas.draw_arrow(k ,index, "Event Detected", color="blue",linestyles='--', picker=False)

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
            click_at_index = event.inaxes.rowNum
            x1, y1 = event.xdata, event.ydata
            x2, y2 = event.x, event.y
            stats = ObspyUtil.get_stats(self.get_file_at_index(click_at_index))
            # Get amplitude from index
            #x_index = int(round(x1 * stats.Sampling_rate))  # index of x-axes time * sample_rate.
            #amplitude = canvas.get_ydata(click_at_index).item(x_index)  # get y-data from index.
            amplitude = y1
            label = "{} {}".format(phase, polarity)
            line = canvas.draw_arrow(x1, click_at_index, label, amplitude=amplitude, color=color, picker=True)
            tt = UTCDateTime(mdt.num2date(x1))
            diff = tt - stats.StartTime
            t = stats.StartTime + diff
            self.picked_at[str(line)] = PickerStructure(t, stats.Station, x1, amplitude, color, label,
                                                        self.get_file_at_index(click_at_index))
            # Add pick data to file.
            self.pm.add_data(t, amplitude, stats.Station, phase, Component = stats.Channel,  First_Motion=polarity)
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

            station = [st.stats.Network,st.stats.Station,st.stats.Location,st.stats.Channel,st.stats.StartTime,
                       st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]

            sd.append(station)

        self._stations_info = StationsInfo(sd)
        self._stations_info.show()

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

        if event.key == 's':
            self.kind_wave = self.ChopCB.currentText()
            id = ""
            self.spectrum = PlotToolsManager(id)
            self.spectrum.plot_spectrum_all(self.chop[self.kind_wave].items())


        if event.key == 'z':
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
            win = int(3*fs)
            tbp = 3
            ntapers = 3
            f_min = 0
            f_max = fn

            self.spectrogram = PlotToolsManager(id)
            [x,y,z] = self.spectrogram.compute_spectrogram_plot(data, win, delta, tbp, ntapers, f_min, f_max, t)
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
            axs = []
            for j in range(self.items_per_page):
                axs.append(self.canvas.get_axe(j))

            if self.nums_clicks > 0:
                pass
            else:

                self.cbar = fig.colorbar(cs, ax=axs[j], extend='both', orientation='horizontal', pad=0.2)
                self.cbar.ax.set_ylabel("Power [dB]")
                #self.cbar.ax.set_ylim(-100,0)

            tr=self.st[self.ax_num]
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
            formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
            ax.xaxis.set_major_formatter(formatter)
            self.nums_clicks = self.nums_clicks+1

    def open_magnitudes_calculator(self):
        hyp_file = os.path.join(ROOT_DIR, "earthquakeAnalisysis", "location_output", "loc", "last.hyp")
        origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file)
        if isinstance(origin, Origin):
            self._magnitude_calc = MagnitudeCalc(origin, self.inventory, self.chop)
            self._magnitude_calc.show()

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