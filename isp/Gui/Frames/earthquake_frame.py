from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from isp.DataProcessing import SeismogramData, DatalessManager
from isp.Gui import pw
from isp.Gui.Frames import BaseFrame, UiEarthquakeAnalysisFrame, Pagination, MessageDialog, FilterBox, EventInfoBox, \
    MatplotlibCanvas, CartopyCanvas, FilesView
from isp.Gui.Utils import map_polarity_from_pressed_key
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, convert_qdatetime_utcdatetime
from isp.Structures.structures import PickerStructure
from isp.Utils import MseedUtil, ObspyUtil
from isp.earthquakeAnalisysis import PickerManager, NllManager, PolarizationAnalyis


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
        self.filter_3ca = FilterBox(self.filter3CAWidget)  # add filter box component.

        self.pagination = Pagination(self.pagination_widget, self.total_items, self.items_per_page)
        self.pagination.set_total_items(0)
        self.pagination.bind_onPage_changed(self.onChange_page)
        self.pagination.bind_onItemPerPageChange_callback(self.onChange_items_per_page)

        self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=self.items_per_page)
        self.canvas.set_xlabel(0, "Time (s)")
        self.canvas.on_double_click(self.on_click_matplotlib)
        self.canvas.on_pick(self.on_pick)

        self.event_info = EventInfoBox(self.eventInfoWidget, self.canvas)
        self.event_info.register_plot_arrivals_click(self.on_click_plot_arrivals)

        # 3C_Component

        self.canvas_3C = MatplotlibCanvas(self.plotMatWidget_3C)
        self.canvas_3C.set_new_subplot(3, ncols=1)
        self.canvas_pol = MatplotlibCanvas(self.Widget_polarization)

        # Map
        self.cartopy_canvas = CartopyCanvas(self.widget_map)

        # Testing map
        self.cartopy_canvas = CartopyCanvas(self.widget_map)

        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_dataless_path)
        # New Binding for 3C
        self.root_path_bind_3C = BindPyqtObject(self.rootPathForm_3C, self.onChange_root_path_3C)
        # Add file selector to the widget
        self.file_selector = FilesView(self.root_path_bind_3C.value, parent=self.fileSelectorWidget,
                                       on_change_file_callback=lambda file_path: self.onChange_file(file_path))

        self.grid_latitude_bind = BindPyqtObject(self.gridlatSB)
        self.grid_longitude_bind = BindPyqtObject(self.gridlonSB)
        self.grid_depth_bind = BindPyqtObject(self.griddepthSB)
        self.grid_xnode_bind = BindPyqtObject(self.xnodeSB)
        self.grid_ynode_bind = BindPyqtObject(self.ynodeSB)
        self.grid_znode_bind = BindPyqtObject(self.znodeSB)
        self.grid_dxsize_bind = BindPyqtObject(self.dxsizeSB)
        self.grid_dysize_bind = BindPyqtObject(self.dysizeSB)
        self.grid_dzsize_bind = BindPyqtObject(self.dzsizeSB)
        self.degreeSB_bind = BindPyqtObject(self.degreeSB)
        # Bind buttons
        self.selectDirBtn_3C.clicked.connect(self.on_click_select_directory_3C)
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        self.sortBtn.clicked.connect(self.on_click_sort)
        self.genvelBtn.clicked.connect(self.on_click_run_vel_to_grid)
        self.grdtimeBtn.clicked.connect(self.on_click_run_grid_to_time)
        self.runlocBtn.clicked.connect(self.on_click_run_loc)
        self.plotmapBtn.clicked.connect(self.on_click_plot_map)
        self.selectVerticalBtn.clicked.connect(self.on_click_set_vertical_component)
        self.selectNorthBtn.clicked.connect(self.on_click_set_north_component)
        self.selectEastBtn.clicked.connect(self.on_click_set_east_component)
        self.rotateplotBtn.clicked.connect(lambda: self.on_click_3C_components(self.canvas_3C))
        self.polarizationBtn.clicked.connect(self.on_click_polarization)
        # self.degreeSB.valueChanged.connect(self.on_click_3C_components)
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

    # Function added for 3C Components
    def onChange_root_path_3C(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        self.file_selector.set_new_rootPath(value)

    # Function added for 3C Components
    def onChange_file(self, file_path):
        # Called every time user select a different file
        pass

    # Function added for 3C Components
    def on_click_select_directory_3C(self):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind_3C.value)

        if dir_path:
            self.root_path_bind_3C.value = dir_path

    def onChange_dataless_path(self, value):
        self.__dataless_manager = DatalessManager(value)

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
        if len(self.canvas.axes) != len(files_at_page):
            self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        last_index = 0
        for index, file_path in enumerate(files_at_page):
            sd = SeismogramData(file_path)
            t, s = sd.get_waveform(filter_error_callback=self.filter_error_message,
                                   filter_value=self.filter.filter_value,
                                   f_min=self.filter.min_freq, f_max=self.filter.max_freq)

            self.canvas.plot(t, s, index, color="black", linewidth=0.5)
            self.redraw_pickers(file_path, index)
            last_index = index

        # set x-label at the last axes.
        self.canvas.set_xlabel(last_index, "Time (s)")

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

    def on_click_run_vel_to_grid(self):
        nll_manager = NllManager(self.pm.output_path, self.dataless_path_bind.value)
        nll_manager.vel_to_grid(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                                self.grid_depth_bind.value, self.grid_xnode_bind.value,
                                self.grid_ynode_bind.value, self.grid_znode_bind.value,
                                self.grid_dxsize_bind.value, self.grid_dysize_bind.value,
                                self.grid_dzsize_bind.value, self.comboBox_gridtype.currentText(),
                                self.comboBox_wavetype.currentText())

    def on_click_run_grid_to_time(self):
        nll_manager = NllManager(self.pm.output_path, self.dataless_path_bind.value)
        nll_manager.grid_to_time(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                                 self.grid_depth_bind.value, self.comboBox_grid.currentText(),
                                 self.comboBox_angles.currentText(), self.comboBox_ttwave.currentText())

    def on_click_run_loc(self):
        nll_manager = NllManager(self.pm.output_path, self.dataless_path_bind.value)
        nll_manager.run_nlloc(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                              self.grid_depth_bind.value)

    def on_click_plot_map(self):
        print("Plotting Map")
        nll_manager = NllManager(self.pm.output_path, self.dataless_path_bind.value)
        lat, lon = nll_manager.get_NLL_info()
        scatter_x, scatter_y, scatter_z = nll_manager.get_NLL_scatter(lat, lon)
        self.cartopy_canvas.plot_map(lon, lat, scatter_x, scatter_y, scatter_z, 0)

    # 3C COMPONENT METHODS####
    # RETRIEVING WAVEFORMS

    def on_click_set_vertical_component(self):
        self.root_path_Form_Vertical.setText(self.file_selector.file_path)

    def on_click_set_north_component(self):
        self.root_path_Form_North.setText(self.file_selector.file_path)

    def on_click_set_east_component(self):
        self.root_path_Form_East.setText(self.file_selector.file_path)

    def on_click_3C_components(self, canvas):
        time1 = convert_qdatetime_utcdatetime(self.dateTimeEdit_4)
        time2 = convert_qdatetime_utcdatetime(self.dateTimeEdit_5)
        angle = self.degreeSB.value()
        sd = PolarizationAnalyis(self.root_path_Form_Vertical.text(), self.root_path_Form_North.text(),
                    self.root_path_Form_East.text())

        time, z, r, t, st = sd.rotate(time1, time2, method="NE->RT", angle=angle,
                                   filter_error_callback=self.filter_error_message,
                                   filter_value=self.filter_3ca.filter_value,
                                   f_min=self.filter_3ca.min_freq, f_max=self.filter_3ca.max_freq)
        rotated_seismograms = [z, r, t]
        for index, data in enumerate(rotated_seismograms):
            canvas.plot(time, data, index, color="black", linewidth=0.5)
        canvas.set_xlabel(2, "Time (s)")

    def on_click_polarization(self):
        time1 = convert_qdatetime_utcdatetime(self.dateTimeEdit_4)
        time2 = convert_qdatetime_utcdatetime(self.dateTimeEdit_5)
        sd = PolarizationAnalyis(self.root_path_Form_Vertical.text(), self.root_path_Form_North.text(),
                 self.root_path_Form_East.text())

        var = sd.polarize(time1, time2, self.doubleSpinBox_winlen.value(), self.spinBox_winoverlap.value(),
                             self.filter_3ca.min_freq, self.filter_3ca.max_freq,
                             method=self.comboBox_methodpolarization.currentText())

        artist = self.canvas_pol.plot(var['time'], var[self.comboBox_yaxis.currentText()], 0, clear_plot=True,linewidth=0.5)
        self.canvas_pol.set_ylabel(0, "Time [s]")
        self.canvas_pol.set_ylabel(0, self.comboBox_yaxis.currentText())
        self.canvas_pol.set_yaxis_color(self.canvas_pol.get_axe(0), artist.get_color(), is_left=True)
        self.canvas_pol.plot(var['time'], var[self.comboBox_polarity.currentText()], 0, is_twinx=True, color="red",linewidth=0.5)
        self.canvas_pol.set_ylabel_twinx(0, self.comboBox_polarity.currentText())



