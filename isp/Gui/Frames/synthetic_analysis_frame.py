import json
import os
from concurrent.futures import ThreadPoolExecutor
from obspy.geodetics import gps2dist_azimuth
from obspy import Stream, UTCDateTime
from obspy.taup import TauPyModel
from isp.DataProcessing import SeismogramDataAdvanced
from isp.Gui import pw, pqg, pyc, qt
from isp.Gui.Frames import UiSyntheticsAnalisysFrame, MatplotlibCanvas, CartopyCanvas, FocCanvas
from isp.Gui.Frames.qt_components import ParentWidget, MessageDialog
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.synthetics_generator_dialog import SyntheticsGeneratorDialog
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from sys import platform
from isp.Utils import MseedUtil
import pandas as pd
import numpy as np
import matplotlib.dates as mdt
import matplotlib.colors as colors


@add_save_load()
class SyntheticsAnalisysFrame(pw.QMainWindow, UiSyntheticsAnalisysFrame):

    def __init__(self, parent: pw.QWidget = None):

        super(SyntheticsAnalisysFrame, self).__init__(parent)

        self.setupUi(self)
        ParentWidget.set_parent(parent, self)
        self.setWindowTitle('Synthetics Analysis Frame')
        self.setWindowIcon(pqg.QIcon(':\icons\pen-icon.png'))

        self.progressbar = pw.QProgressDialog(self)
        self.progressbar.setWindowTitle('Synthetics Analysis Frame')
        self.progressbar.setLabelText(" Computing ")
        self.progressbar.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.progressbar.close()

        # Initialize parametrs for plot rotation

        self.parameters = None
        self.stream = None
        self.inventory = None
        self.model = TauPyModel(model="ak135")
        self._generator = SyntheticsGeneratorDialog(self)
        self._generator.show()

        # Process Part
        self.focmec_canvas = FocCanvas(self.widget_fp)
        self.canvas = MatplotlibCanvas(self.plotMatWidget_3C)
        self.canvas.set_new_subplot(1, ncols=1)

        # Map
        self.cartopy_canvas = CartopyCanvas(self.map_widget)

        # binds

        self.generation_params_bind = BindPyqtObject(self.paramsPathLineEdit, self.onChange_root_path)
        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.stations_coords_path_bind = BindPyqtObject(self.stationscoordsPathForm, self.onChange_root_path)

        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.infoBtn.clicked.connect(lambda: self.on_click_select_file(self.generation_params_bind))
        self.stationsCoordsBtn.clicked.connect(lambda: self.on_click_select_file(self.stations_coords_path_bind))

        self.plotBtn.clicked.connect(self.plot)

        self.readFilesBtn.clicked.connect(lambda: self.get_now_files())
        self.stationsBtn.clicked.connect(self.stationsInfo)
        self.PABtn.clicked.connect(self.plotArrivals)
        self.mapBtn.clicked.connect(self.map_coords)
        ###

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def on_click_select_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]

    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path

    def onChange_root_path(self, value):

        """
        Fired every time the root_path is changed
        :param value: The path of the new directory.
        :return:
        """
        pass

    def plot(self):

        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=1, ncols=1)
        #self.__map_coords()

        parameters = []
        min_starttime = []
        max_endtime = []

        self.generationParams()
        if self.sortCB.isChecked():
            stations_df = self.sort_traces()

        for index, tr in enumerate(self.stream):

            sd = SeismogramDataAdvanced(file_path=None, stream=tr, realtime=True)
            tr = sd.get_waveform_advanced(parameters, self.inventory,
                                          filter_error_callback=self.filter_error_message, trace_number=0)

            distance = stations_df.loc[(stations_df['Network'] == tr.stats.network) &
                                       (stations_df['Station'] == tr.stats.station), 'Distance'].values[0] * 1e-3

            if len(tr) > 0:
                t = tr.times("matplotlib")
                tr.detrend(type="simple")
                s = (tr.data / np.max(tr.data))
                if distance:
                    s = s * self.sizeSB.value() + distance
                else:
                    s = s + index

                label_trace = tr.stats.network+"."+tr.stats.station+"."+tr.stats.channel
                self.canvas.plot_date(t, s, 0, clear_plot=False, fmt='-', alpha=0.5,
                                      linewidth=0.5, label=label_trace)

                try:
                    min_starttime.append(min(t))
                    max_endtime.append(max(t))
                except:
                    print("Empty traces")

        try:
            if min_starttime and max_endtime is not None:
                auto_start = min(min_starttime)
                auto_end = max(max_endtime)
                self.auto_start = auto_start
                self.auto_end = auto_end

            ax = self.canvas.get_axe(0)
            ax.set_xlim(mdt.num2date(auto_start), mdt.num2date(auto_end))
            formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(0, "Date")
            self.canvas.set_ylabel(0, "Distance [km]")
            ax.legend()
        except:
            pass


    def plotArrivals(self):

        travel_times_df = self.get_phases_and_arrivals()
        if isinstance(travel_times_df, pd.DataFrame):

            # Applying specific filter
            phases_to_filter = self.phasesLE.text().split(",")
            pp_phase_df = travel_times_df.query("Phase in @phases_to_filter")
            all_plot_phases = []
            # Loop through each unique Network and Station combination
            for (network, station), group_df in pp_phase_df.groupby(['Network', 'Station']):
                print(f"Network: {network}, Station: {station}")

                # Extract distance (assuming distance is the same for all phases at this station)
                distance = group_df['Distance_km'].iloc[0] * 1e-3

                print(f"Distance (km): {distance}")

                # Loop over each row in the group to get Phase and Time
                phases_plot = []  # to select the first phase of one kind
                unique_phases = self._get_unique_phases(group_df)
                for _, row in group_df.iterrows():
                    phase = row['Phase']
                    if phase not in phases_plot:
                        # if choosen_color not in selected_colors:

                        arrival_time = UTCDateTime(self.params['origintime']) + float(row['Time_s'])

                        print(f"  Phase: {phase}, Arrival Time (s): {arrival_time}")
                        if phase not in all_plot_phases:
                            self.canvas.plot_date(arrival_time, distance, 0, color=unique_phases[phase],
                                                  clear_plot=False, fmt='.', markeredgecolor='black', markeredgewidth=1,
                                                  linewidth=0.5, label=phase)
                        else:
                            self.canvas.plot_date(arrival_time, distance, 0, color=unique_phases[phase],
                                                  clear_plot=False, fmt='.', markeredgecolor='black', markeredgewidth=1,
                                                  linewidth=0.5, label="")
                    all_plot_phases.append(phase)
                    phases_plot.append(phase)

                print("------------------")
            ax = self.canvas.get_axe(0)
            ax.legend()

    def _get_unique_phases(self, group_df: pd.DataFrame) -> dict:

        named_colors = list(colors.CSS4_COLORS.keys())
        unique_phases = {}
        j = 0
        for _, row in group_df.iterrows():
            if row['Phase'] not in unique_phases.keys():
                unique_phases[row['Phase']] = named_colors[j]
                j = j + 1

        return unique_phases

    def get_files(self, dir_path):

        selection = {}
        if self.netLE.text() != "":
            selection["network"] = self.netLE.text()
        else:
            selection["network"] = "*"

        if self.stationLE.text() != "":
            selection["station"] = self.stationLE.text()
        else:
            selection["station"] = "*"

        if self.channelLE.text() != "":
            selection["channel"] = self.channelLE.text()
        else:
            selection["channel"] = "*"
        stream = MseedUtil.get_stream(dir_path, selection=selection)
        pyc.QMetaObject.invokeMethod(self.progressbar, 'accept', qt.AutoConnection)
        return stream

    def get_now_files(self):

        md = MessageDialog(self)
        md.hide()
        try:

            self.progressbar.reset()
            self.progressbar.setLabelText(" Reading Files ")
            self.progressbar.setRange(0, 0)
            with ThreadPoolExecutor(1) as executor:
                f = executor.submit(lambda: self.get_files(self.root_path_bind.value))
                self.progressbar.exec()
                self.stream = f.result()
                f.cancel()

            md.set_info_message("Readed data files Successfully")

        except:

            md.set_error_message("Something went wrong. Please check your data files are correct mseed files")

        md.show()

    def stationsInfo(self):

        print("StationsInfo")

    def generationParams(self):
        with open(self.generation_params_bind.value, 'rb') as f:
            self.params = json.load(f)
            depth_est = self.params['sourcedepthinmeters']
            depth_est = (float(depth_est)) / 1000
            # self.paramsTextEdit.setPlainText(str(params))
            self.paramsTextEdit.setPlainText("Earth Model: {model}".format(model=self.params['model']))
            self.paramsTextEdit.appendPlainText("Event Coords: {lat} {lon} {depth}".format(
                lat=self.params['sourcelatitude'], lon=self.params['sourcelongitude'],
                depth=str(depth_est)))
            self.paramsTextEdit.appendPlainText("Time: {time}".format(time=self.params['origintime']))
            self.paramsTextEdit.appendPlainText("Units: {units}".format(units=self.params['units']))

            if 'sourcedoublecouple' in self.params:
                self.paramsTextEdit.appendPlainText("Source: {source}".format(source=self.params['sourcedoublecouple']))
            if 'sourcemomenttensor' in self.params:
                self.paramsTextEdit.appendPlainText("Source: {source}".format(source=self.params['sourcemomenttensor']))


    def _get_stations(self):

        self._stations = []
        for treace in self.stream:
            station = treace.stats.station
            if station not in self._stations:
                self._stations.append(station)


    def sort_traces(self):

        stations_df = pd.read_csv(self.stations_coords_path_bind.value, delimiter=';')
        self._get_stations()

        # Filter travel_times_df to include only rows where 'Station' is in the available_stations list
        stations_df = stations_df[stations_df['Station'].isin(self._stations)]

        with open(self.generation_params_bind.value, 'rb') as f:
            self.params = json.load(f)

        if self.sortbyCB.currentText() == "Distance":
            stations_df['Distance'] = stations_df.apply(self._calculate_distance, axis=1)
            stations_df = stations_df.sort_values(by='Distance').reset_index(drop=True)
            self.stream = self._sort_stream_by(stations_df)


        elif self.sortbyCB.currentText() == "Azimuth":
            stations_df['Azimuth'] = stations_df.apply(self._calculate_azim, axis=1)
            stations_df = stations_df.sort_values(by='Azimuth').reset_index(drop=True)
            self.stream = self._sort_stream_by(stations_df)

        return stations_df

    def _calculate_distance(self, row):

        dist, bazim, azim = gps2dist_azimuth(row['Latitude'], row['Longitude'], self.params['sourcelatitude'],
                                             self.params['sourcelongitude'])
        return dist

    def _calculate_azim(self, row):

        dist, bazim, azim = gps2dist_azimuth(row['Latitude'], row['Longitude'], self.params['sourcelatitude'],
                                             self.params['sourcelongitude'])
        return azim

    def _sort_stream_by(self, stations_df):
        sorted_traces = []
        for _, station in stations_df.iterrows():
            # Find matching trace in the stream by Network and Station codes
            for trace in self.stream:
                if (trace.stats.network == station['Network'] and
                        trace.stats.station == station['Station']):
                    sorted_traces.append(trace)
                    #break  # Move to the next station after finding a match
        return Stream(traces=sorted_traces)

    def get_phases_and_arrivals(self):

        travel_times = []
        stations_df = pd.read_csv(self.stations_coords_path_bind.value, delimiter=';')
        # Filter travel_times_df to include only rows where 'Station' is in the available_stations list
        stations_df = stations_df[stations_df['Station'].isin(self._stations)]

        stations_df['Distance'] = stations_df.apply(self._calculate_distance, axis=1)
        stations_df = stations_df.sort_values(by='Distance').reset_index(drop=True)

        with open(self.generation_params_bind.value, 'rb') as f:
            self.params = json.load(f)
            depth_est = self.params['sourcedepthinmeters']
            depth_est = (float(depth_est)) / 1000

        for _, station in stations_df.iterrows():
            distance_deg = (station['Distance'] * 1e-3) / 111.19  # Approximate conversion from km to degrees
            arrivals = self.model.get_travel_times(distance_in_degree=distance_deg,
                                                   source_depth_in_km=depth_est)  # Depth can be set as needed

            for arrival in arrivals:
                travel_times.append({
                    "Station": station['Station'],
                    "Network": station['Network'],
                    "Distance_km": station['Distance'],
                    "Phase": arrival.name,
                    "Time_s": arrival.time
                })

        # Convert the list of dictionaries into a DataFrame
        travel_times_df = pd.DataFrame(travel_times)
        # Remove duplicate rows
        #travel_times_df_drop = travel_times_df.drop_duplicates(keep=False)

        return travel_times_df

    def stationsInfo(self):

        sd = []
        for index, tr in enumerate(self.stream):
            station = [tr.stats.network, tr.stats.station, "", tr.stats.channel, tr.stats.starttime,
                       tr.stats.endtime, tr.stats.sampling_rate, tr.stats.npts]

            sd.append(station)

        self._stations_info = StationsInfo(sd)
        self._stations_info.show()

    def map_coords(self):

        map_dict = {}
        sd = []

        with open(self.generation_params_bind.value, 'rb') as f:
            params = json.load(f)

        n = len(params["bulk"])
        for j in range(n):

            for key in params["bulk"][j]:
                if key == "latitude":
                   lat = params["bulk"][j][key]

                elif key == "longitude":
                    lon = params["bulk"][j][key]

                elif key == "stationcode":
                    sta = params["bulk"][j][key]

                    sd.append(sta)
                    map_dict[sta] = [lon, lat]


        self.cartopy_canvas.plot_map(params['sourcelongitude'], params['sourcelatitude'],
                                     0, 0, 0, 0, resolution='low',
                                     stations=map_dict)

        if 'sourcedoublecouple' in params:
            self.focmec_canvas.drawSynthFocMec(0, first_polarity=params['sourcedoublecouple'], mti = [])
        if 'sourcemomenttensor' in params:
            self.focmec_canvas.drawSynthFocMec(0, first_polarity= [], mti=params['sourcemomenttensor'])

        self.tabWidget.setCurrentIndex(1)