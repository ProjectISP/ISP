from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from isp.DataProcessing import SeismogramData, DatalessManager
from isp.Gui import pw
from isp.Gui.Frames import BaseFrame, UiEarthquakeAnalysisFrame, Pagination, MessageDialog, FilterBox, EventInfoBox, \
    MatplotlibCanvas
from isp.Gui.Frames.earthquake_frame_tabs import Earthquake3CFrame, EarthquakeLocationFrame
from isp.Gui.Utils import map_polarity_from_pressed_key
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, convert_qdatetime_utcdatetime
from isp.Structures.structures import PickerStructure
from isp.Utils import MseedUtil, ObspyUtil, AsycTime
from isp.earthquakeAnalisysis import PickerManager


class EarthquakeAnalysisFrame(BaseFrame, UiEarthquakeAnalysisFrame):

    def __init__(self):
        super(EarthquakeAnalysisFrame, self).__init__()
        self.setupUi(self)

        self.files = []
        self.total_items = 0
        self.items_per_page = 1
        # dict to keep track of picks-> dict(key: PickerStructure) as key we use the drawn line.
        self.picked_at = {}
        self.__dataless_manager = None
        self.dataless_not_found = set()  # a set of mseed files that the dataless couldn't find.

        self.filter = FilterBox(self.filterWidget)  # add filter box component.

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

        self.event_info = EventInfoBox(self.eventInfoWidget, self.canvas)
        self.event_info.register_plot_arrivals_click(self.on_click_plot_arrivals)

        self.earthquake_3c_frame = Earthquake3CFrame(self.parentWidget3C)
        self.earthquake_location_frame = EarthquakeLocationFrame(self.parentWidgetLocation)

        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_dataless_path)

        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        self.sortBtn.clicked.connect(self.on_click_sort)
        self.updateBtn.clicked.connect(self.plot_seismogram)
        self.pm = PickerManager()  # start PickerManager to save pick location to csv file.

    @property
    def dataless_manager(self):
        if not self.__dataless_manager:
            self.__dataless_manager = DatalessManager(self.dataless_path_bind.value)
        return self.__dataless_manager

    def message_dataless_not_found(self):
        if len(self.dataless_not_found) > 1:
            md = MessageDialog(self)
            md.set_info_message("Dataless not found.")
        else:
            for file in self.dataless_not_found:
                md = MessageDialog(self)
                md.set_info_message("Dataless for {} not found.".format(file))

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
        self.files = MseedUtil.get_mseed_files(value)
        self.total_items = len(self.files)
        self.pagination.set_total_items(self.total_items)
        self.plot_seismogram()

    def onChange_dataless_path(self, value):
        self.__dataless_manager = DatalessManager(value)
        self.earthquake_location_frame.set_dataless_dir(value)

    def sort_by_distance(self, file):
        st_stats = self.dataless_manager.get_station_stats_by_mseed_file(file)
        if st_stats:
            dist, _, _ = gps2dist_azimuth(st_stats.Lat, st_stats.Lon, 0., 0.)
            # print("File, dist: ", file, dist)
            return dist
        else:
            self.dataless_not_found.add(file)
            print("No dataless found for {} file.".format(file))
            return 0.

    def on_click_sort(self):
        self.files.sort(key=self.sort_by_distance)
        self.message_dataless_not_found()
        self.plot_seismogram()

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)

        if dir_path:
            bind.value = dir_path


    def plot_seismogram(self):
        self.canvas.clear()
        files_at_page = self.get_files_at_page()
        start_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        end_time = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)
        if len(self.canvas.axes) != len(files_at_page):
            self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        last_index = 0
        time_min = []
        time_max = []
        for index, file_path in enumerate(files_at_page):
            sd = SeismogramData(file_path)
            t,t_sec, s = sd.get_waveform(filter_error_callback=self.filter_error_message,
                                   filter_value=self.filter.filter_value,
                                   f_min=self.filter.min_freq, f_max=self.filter.max_freq, start_time=start_time,
                                   end_time=end_time)

            self.canvas.plot(t, s, index, color="black", linewidth=0.5)
            self.redraw_pickers(file_path, index)
            last_index = index

            st_stats = ObspyUtil.get_stats(file_path)
            if st_stats:
                info = "{}-{}-{}".format(st_stats.Network, st_stats.Station, st_stats.Channel)
                self.canvas.set_plot_label(index, info)

            time_min.append(min(t))
            time_max.append(max(t))

        # set x-label at the last axes.
        ax = self.canvas.get_axe(last_index)
        ax.set_xlim(min(time_min), max(time_max))
        self.canvas.set_xlabel(last_index, "Date")
        self.canvas.figure.autofmt_xdate()

    def redraw_pickers(self, file_name, axe_index):

        picked_at = {key: values for key, values in self.picked_at.items()}  # copy the dictionary.
        for key, value in picked_at.items():
            ps: PickerStructure = value
            if file_name == ps.FileName:
                new_line = self.canvas.draw_arrow(ps.XPosition, axe_index, ps.Label,
                                                  amplitude=ps.Amplitude, color=ps.Color, picker=True)
                self.picked_at.pop(key)
                self.picked_at[str(new_line)] = ps

    def on_click_matplotlib(self, event, canvas):
        if isinstance(canvas, MatplotlibCanvas):
            polarity, color = map_polarity_from_pressed_key(event.key)
            phase = self.comboBox_phases.currentText()
            click_at_index = event.inaxes.rowNum
            x1, y1 = event.xdata, event.ydata
            stats = ObspyUtil.get_stats(self.get_file_at_index(click_at_index))
            # Get amplitude from index
            x_index = int(round(x1 * stats.Sampling_rate))  # index of x-axes time * sample_rate.
            amplitude = canvas.get_ydata(click_at_index).item(x_index)  # get y-data from index.
            label = "{} {}".format(phase, polarity)
            line = canvas.draw_arrow(x1, click_at_index, label, amplitude=amplitude, color=color, picker=True)

            t = stats.StartTime + x1
            self.picked_at[str(line)] = PickerStructure(t, stats.Station, x1, amplitude, color, label,
                                                        self.get_file_at_index(click_at_index))
            # Add pick data to file.
            self.pm.add_data(t, amplitude, stats.Station, phase, First_Motion=polarity)
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
            st_stats = self.dataless_manager.get_station_stats_by_mseed_file(file_path)
            stats = ObspyUtil.get_stats(file_path)
            # TODO remove stats.StartTime and use the picked one from UI.
            self.event_info.plot_arrivals(index, stats.StartTime, st_stats)
