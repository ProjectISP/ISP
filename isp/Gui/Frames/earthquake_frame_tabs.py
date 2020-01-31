import os

from obspy.core.event import Origin

from isp.Exceptions import InvalidFile, parse_excepts
from isp.Gui import pw, pqg
from isp.Gui.Frames import UiEarthquake3CFrame, MatplotlibCanvas, UiEarthquakeLocationFrame, CartopyCanvas
from isp.Gui.Frames.qt_components import ParentWidget, FilterBox, FilesView, MessageDialog
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject, convert_qdatetime_utcdatetime
from isp.earthquakeAnalisysis import PolarizationAnalyis, NllManager, PickerManager


@add_save_load()
class Earthquake3CFrame(pw.QFrame, UiEarthquake3CFrame):

    def __init__(self, parent: pw.QWidget):

        super(Earthquake3CFrame, self).__init__(parent)

        self.setupUi(self)
        ParentWidget.set_parent(parent, self)

        self.filter_3ca = FilterBox(self.toolQFrame, 1)  # add filter box component.

        # 3C_Component
        self.canvas = MatplotlibCanvas(self.plotMatWidget_3C)
        self.canvas.set_new_subplot(3, ncols=1)
        self.canvas_pol = MatplotlibCanvas(self.Widget_polarization)

        # binds
        self.root_path_bind_3C = BindPyqtObject(self.rootPathForm_3C, self.onChange_root_path_3C)
        self.degreeSB_bind = BindPyqtObject(self.degreeSB)
        self.vertical_form_bind = BindPyqtObject(self.verticalQLineEdit)
        self.north_form_bind = BindPyqtObject(self.northQLineEdit)
        self.east_form_bind = BindPyqtObject(self.eastQLineEdit)

        # accept drops
        self.vertical_form_bind.accept_dragFile(drop_event_callback=self.drop_event)
        self.north_form_bind.accept_dragFile(drop_event_callback=self.drop_event)
        self.east_form_bind.accept_dragFile(drop_event_callback=self.drop_event)

        # Add file selector to the widget
        self.file_selector = FilesView(self.root_path_bind_3C.value, parent=self.fileSelectorWidget)
        self.file_selector.setDragEnabled(True)

        self.selectDirBtn_3C.clicked.connect(self.on_click_select_directory_3C)
        self.rotateplotBtn.clicked.connect(lambda: self.on_click_rotate(self.canvas))
        self.polarizationBtn.clicked.connect(self.on_click_polarization)

    def info_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    @staticmethod
    def drop_event(event: pqg.QDropEvent, bind_object: BindPyqtObject):
        data = event.mimeData()
        url = data.urls()[0]
        bind_object.value = url.fileName()

    @property
    def north_component_file(self):
        return os.path.join(self.root_path_bind_3C.value, self.north_form_bind.value)

    @property
    def vertical_component_file(self):
        return os.path.join(self.root_path_bind_3C.value, self.vertical_form_bind.value)

    @property
    def east_component_file(self):
        return os.path.join(self.root_path_bind_3C.value, self.east_form_bind.value)

    def onChange_root_path_3C(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        self.file_selector.set_new_rootPath(value)

    # Function added for 3C Components
    def on_click_select_directory_3C(self):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind_3C.value)

        if dir_path:
            self.root_path_bind_3C.value = dir_path

    def on_click_rotate(self, canvas):
        time1 = convert_qdatetime_utcdatetime(self.dateTimeEdit_4)
        time2 = convert_qdatetime_utcdatetime(self.dateTimeEdit_5)
        angle = self.degreeSB.value()
        try:
            sd = PolarizationAnalyis(self.vertical_component_file, self.north_component_file,
                                     self.east_component_file)

            time, z, r, t, st = sd.rotate(time1, time2, method="NE->RT", angle=angle,
                                          filter_error_callback=self.info_message,
                                          filter_value=self.filter_3ca.filter_value,
                                          f_min=self.filter_3ca.min_freq, f_max=self.filter_3ca.max_freq)
            rotated_seismograms = [z, r, t]
            for index, data in enumerate(rotated_seismograms):
                canvas.plot(time, data, index, color="black", linewidth=0.5)
            canvas.set_xlabel(2, "Time (s)")

        except InvalidFile:
            self.info_message("Invalid mseed files. Please, make sure to select all the three components (Z, N, E) "
                              "for rotate.")
        except ValueError as error:
            self.info_message(str(error))

    def on_click_polarization(self):
        time1 = convert_qdatetime_utcdatetime(self.dateTimeEdit_4)
        time2 = convert_qdatetime_utcdatetime(self.dateTimeEdit_5)
        sd = PolarizationAnalyis(self.vertical_component_file, self.north_component_file,
                                 self.east_component_file)
        try:
            var = sd.polarize(time1, time2, self.doubleSpinBox_winlen.value(), self.spinBox_winoverlap.value(),
                              self.filter_3ca.min_freq, self.filter_3ca.max_freq,
                              method=self.comboBox_methodpolarization.currentText())

            artist = self.canvas_pol.plot(var['time'], var[self.comboBox_yaxis.currentText()], 0, clear_plot=True,
                                          linewidth=0.5)
            self.canvas_pol.set_xlabel(0, "Time [s]")
            self.canvas_pol.set_ylabel(0, self.comboBox_yaxis.currentText())
            self.canvas_pol.set_yaxis_color(self.canvas_pol.get_axe(0), artist.get_color(), is_left=True)
            self.canvas_pol.plot(var['time'], var[self.comboBox_polarity.currentText()], 0, is_twinx=True, color="red",
                                 linewidth=0.5)
            self.canvas_pol.set_ylabel_twinx(0, self.comboBox_polarity.currentText())
        except InvalidFile:
            self.info_message("Invalid mseed files. Please, make sure to select all the three components (Z, N, E) "
                              "for polarization.")
        except ValueError as error:
            self.info_message(str(error))


@add_save_load()
class EarthquakeLocationFrame(pw.QFrame, UiEarthquakeLocationFrame):

    def __init__(self, parent: pw.QWidget):
        super(EarthquakeLocationFrame, self).__init__(parent)

        self.setupUi(self)
        ParentWidget.set_parent(parent, self)
        self.__pick_output_path = PickerManager.get_default_output_path()
        self.__dataless_dir = None
        self.__nll_manager = None

        # Map
        self.cartopy_canvas = CartopyCanvas(self.widget_map)
        # Canvas for Earthquake Location Results
        self.residuals_canvas = MatplotlibCanvas(self.plotMatWidget_residuals)

        self.grid_latitude_bind = BindPyqtObject(self.gridlatSB)
        self.grid_longitude_bind = BindPyqtObject(self.gridlonSB)
        self.grid_depth_bind = BindPyqtObject(self.griddepthSB)
        self.grid_xnode_bind = BindPyqtObject(self.xnodeSB)
        self.grid_ynode_bind = BindPyqtObject(self.ynodeSB)
        self.grid_znode_bind = BindPyqtObject(self.znodeSB)
        self.grid_dxsize_bind = BindPyqtObject(self.dxsizeSB)
        self.grid_dysize_bind = BindPyqtObject(self.dysizeSB)
        self.grid_dzsize_bind = BindPyqtObject(self.dzsizeSB)

        self.genvelBtn.clicked.connect(lambda: self.on_click_run_vel_to_grid())
        self.grdtimeBtn.clicked.connect(lambda: self.on_click_run_grid_to_time())
        self.runlocBtn.clicked.connect(lambda: self.on_click_run_loc())
        self.plotmapBtn.clicked.connect(self.on_click_plot_map)

    @property
    def nll_manager(self):
        if not self.__nll_manager:
            self.__nll_manager = NllManager(self.__pick_output_path, self.__dataless_dir)
        return self.__nll_manager

    def set_dataless_dir(self, dir_path):
        self.__dataless_dir = dir_path
        self.nll_manager.set_dataless_dir(dir_path)

    def set_pick_output_path(self, file_path):
        self.__pick_output_path = file_path
        self.nll_manager.set_observation_file(file_path)

    def info_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def subprocess_feedback(self, msg: str):
        md = MessageDialog(self)
        if msg:
            if "Error code" in msg:
                md.set_error_message("Click in show details detail for more info.", msg)
            else:
                md.set_warning_message("Click in show details for more info.", msg)
        else:
            md.set_info_message("Completed Successfully")

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_run_vel_to_grid(self):
        self.nll_manager.vel_to_grid(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                                     self.grid_depth_bind.value, self.grid_xnode_bind.value,
                                     self.grid_ynode_bind.value, self.grid_znode_bind.value,
                                     self.grid_dxsize_bind.value, self.grid_dysize_bind.value,
                                     self.grid_dzsize_bind.value, self.comboBox_gridtype.currentText(),
                                     self.comboBox_wavetype.currentText())

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_run_grid_to_time(self):
        self.nll_manager.grid_to_time(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                                      self.grid_depth_bind.value, self.comboBox_grid.currentText(),
                                      self.comboBox_angles.currentText(), self.comboBox_ttwave.currentText())

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_run_loc(self):
        self.nll_manager.run_nlloc(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                                   self.grid_depth_bind.value)

    def on_click_plot_map(self):
        try:
            origin = self.nll_manager.get_NLL_info()
            scatter_x, scatter_y, scatter_z, pdf = self.nll_manager.get_NLL_scatter()
            lat = origin.latitude
            lon = origin.longitude
            self.cartopy_canvas.plot_map(lon, lat, scatter_x, scatter_y, scatter_z, 0)
            # Writing Location information
            self.add_earthquake_info(origin)
            xp, yp, xs, ys = self.nll_manager.ger_NLL_residuals()
            self.plot_residuals(xp, yp, xs, ys)
        except (FileNotFoundError, AttributeError) as error:
            self.info_message(str(error))

    def plot_residuals(self, xp, yp, xs, ys):

        artist = self.residuals_canvas.plot(xp, yp, axes_index=0, linewidth=0.5)
        self.canvas_pol.set_xlabel(0, "Station Name")
        self.canvas_pol.set_ylabel(0, "P wave Residuals")
        self.canvas_pol.set_yaxis_color(self.residuals_canvas.get_axe(0), artist.get_color(), is_left=True)
        self.canvas_pol.plot(xs, ys, 0, is_twinx=True, color="red", linewidth=0.5)
        self.canvas_pol.set_ylabel_twinx(0, "S wave Residuals")

    def add_earthquake_info(self, origin: Origin):

        self.EarthquakeInfoText.setPlainText("  Origin time and RMS:     {origin_time}     {standard_error:.3f}".
                                             format(origin_time=origin.time,
                                                    standard_error=origin.quality.standard_error))
        self.EarthquakeInfoText.appendPlainText("  Hypocenter Geographic Coordinates:     "
                                                "Latitude {lat:.3f} "
                                                "Longitude {long:.3f}     Depth {depth:.3f}     "
                                                "Uncertainty {unc:.3f}".
                                                format(lat=origin.latitude, long=origin.longitude,
                                                       depth=origin.depth / 1000,
                                                       unc=origin.depth_errors['uncertainty'] / 10000))
        self.EarthquakeInfoText.appendPlainText("  Horizontal Ellipse:     Max Horizontal Err {:.3f}     "
                                                "Min Horizontal Err {:.3f}     "
                                                "Azimuth {:.3f}"
                                                .format(origin.origin_uncertainty.max_horizontal_uncertainty / 1000,
                                                        origin.origin_uncertainty.min_horizontal_uncertainty / 1000,
                                                        origin.origin_uncertainty.azimuth_max_horizontal_uncertainty))

        self.EarthquakeInfoText.appendPlainText("  Quality Parameters:     Number of Phases {:.3f}     "
                                                "Azimuthal GAP {:.3f}     Minimum Distance {:.3f}     "
                                                "Maximum Distance {:.3f}"
                                                .format(origin.quality.used_phase_count,
                                                        origin.quality.azimuthal_gap,
                                                        origin.quality.minimum_distance,
                                                        origin.quality.maximum_distance))
