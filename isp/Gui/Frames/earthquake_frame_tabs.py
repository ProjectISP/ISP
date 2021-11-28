import os
import matplotlib.dates as mdt
from obspy import Stream
from obspy.core.event import Origin
from isp import ROOT_DIR
from isp.DataProcessing import SeismogramDataAdvanced
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import InvalidFile, parse_excepts
from isp.Gui import pw, pqg
from isp.Gui.Frames import UiEarthquake3CFrame, MatplotlibCanvas, UiEarthquakeLocationFrame, CartopyCanvas, FocCanvas
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.plot_polarization import PlotPolarization
from isp.Gui.Frames.qt_components import ParentWidget, FilterBox, FilesView, MessageDialog
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject, convert_qdatetime_utcdatetime
from isp.earthquakeAnalisysis import NllManager, PolarizationAnalyis, PickerManager, FirstPolarity, PDFmanger
import numpy as np
from sys import platform

@add_save_load()
class Earthquake3CFrame(pw.QFrame, UiEarthquake3CFrame):

    def __init__(self, parent: pw.QWidget):

        super(Earthquake3CFrame, self).__init__(parent)

        self.setupUi(self)
        ParentWidget.set_parent(parent, self)
        #Initialize parametrs for plot rotation
        self._z = {}
        self._r = {}
        self._t = {}
        self._st = {}
        self.inventory = {}
        #self.filter_3ca = FilterBox(self.toolQFrame, 1)  # add filter box component.
        self.parameters = ParametersSettings()

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
        self.rot_macroBtn.clicked.connect(lambda: self.open_parameters_settings())
        self.polarizationBtn.clicked.connect(self.on_click_polarization)
        ###
        self.plotpolBtn.clicked.connect(self.plot_particle_motion)
        self.stationsBtn.clicked.connect(self.stationsInfo)
        self.save_rotatedBtn.clicked.connect(self.save_rotated)
        ###

    def open_parameters_settings(self):
        self.parameters.show()

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

        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind_3C.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind_3C.value,
                                                           pw.QFileDialog.DontUseNativeDialog)

        if dir_path:
            self.root_path_bind_3C.value = dir_path


    def set_times(self, st):

        maxstart = np.max([tr.stats.starttime for tr in st])
        minend = np.min([tr.stats.endtime for tr in st])

        return minend, maxstart

    def on_click_rotate(self, canvas):

        time1 = convert_qdatetime_utcdatetime(self.dateTimeEdit_4)
        time2 = convert_qdatetime_utcdatetime(self.dateTimeEdit_5)

        #self.st = Stream(traces=[self.vertical_component_file, self.north_component_file, self.east_component_file])
        #minend, maxstart = self.set_times(self.st)

        #if time1 < minend:
        #    time1=minend
        #if time2 > maxstart:
        #    time2 = maxstart
        #try:

        angle = self.degreeSB.value()
        incidence_angle= self.incidenceSB.value()
        method = self.methodCB.currentText()
        parameters = self.parameters.getParameters()

        try:
            sd = PolarizationAnalyis(self.vertical_component_file, self.north_component_file, self.east_component_file)
            time, z, r, t, st = sd.rotate(self.inventory, time1, time2, angle, incidence_angle, method = method, parameters = parameters,
                                          trim = True)
            self._z = z
            self._r = r
            self._t = t
            self._st = st
            rotated_seismograms = [z, r, t]
            for index, data in enumerate(rotated_seismograms):
                self.canvas.plot(time, data, index, color="black", linewidth=0.5)
                info = "{}.{}.{}".format(self._st[index].stats.network, self._st[index].stats.station,
                                         self._st[index].stats.channel)
                ax = self.canvas.get_axe(0)
                ax.set_xlim(time1.matplotlib_date, time2.matplotlib_date)
                formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
                ax.xaxis.set_major_formatter(formatter)
                self.canvas.set_plot_label(index, info)

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
            var = sd.polarize(time1, time2, self.doubleSpinBox_winlen.value(),
                              self.freq_minDB.value(), self.freq_maxDB.value())

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

    def plot_particle_motion(self):
        self._plot_polarization = PlotPolarization(self._z, self._r, self._t)
        self._plot_polarization.show()

    def stationsInfo(self):
        files = []
        try:
            if self.vertical_component_file and self.north_component_file and self.east_component_file:
                files = [self.vertical_component_file, self.north_component_file, self.east_component_file]
        except:
            pass

        sd = []
        if len(files)==3:
            for file in files:
                try:
                    st = SeismogramDataAdvanced(file)

                    station = [st.stats.Network,st.stats.Station,st.stats.Location,st.stats.Channel,st.stats.StartTime,
                           st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]

                    sd.append(station)
                except:
                    pass

            self._stations_info = StationsInfo(sd)
            self._stations_info.show()

    def save_rotated(self):
        import os
        root_path = os.path.dirname(os.path.abspath(__file__))

        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if self._st:
            n = len(self._st)
            for j in range(n):
                tr = self._st[j]
                t1 = tr.stats.starttime
                id = tr.id+"."+"D"+"."+str(t1.year)+"."+str(t1.julday)
                print(tr.id, "Writing data processed")
                path_output = os.path.join(dir_path, id)
                tr.write(path_output, format="MSEED")


@add_save_load()
class EarthquakeLocationFrame(pw.QFrame, UiEarthquakeLocationFrame):

    def __init__(self, parent: pw.QWidget):
        super(EarthquakeLocationFrame, self).__init__(parent)

        self.setupUi(self)
        ParentWidget.set_parent(parent, self)
        self.__pick_output_path = PickerManager.get_default_output_path()
        self.__dataless_dir = None
        self.__nll_manager = None
        self.__first_polarity = None

        # Map
        self.cartopy_canvas = CartopyCanvas(self.widget_map)
        # Canvas for Earthquake Location Results
        self.residuals_canvas = MatplotlibCanvas(self.plotMatWidget_residuals)
        #self.residuals_canvas.figure.subplots_adjust(left = 0.03, bottom = 0.36, right=0.97, top=0.95, wspace=0.2,
        #                                         hspace=0.0)

        # Canvas for FOCMEC  Results
        self.focmec_canvas = FocCanvas(self.widget_focmec)
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
        self.plotmapBtn.clicked.connect(lambda: self.on_click_plot_map())
        self.stationsBtn.clicked.connect(lambda: self.on_click_select_metadata_file())
        self.firstpolarityBtn.clicked.connect(self.first_polarity)
        self.plotpdfBtn.clicked.connect(self.plot_pdf)

    @property
    def nll_manager(self):
        if not self.__nll_manager:
            self.__nll_manager = NllManager(self.__pick_output_path, self.__dataless_dir)
        return self.__nll_manager

    @property
    def firstpolarity_manager(self):
        if not self.__first_polarity:
            self.__first_polarity = FirstPolarity()
        return self.__first_polarity

    def on_click_select_metadata_file(self):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata/stations coordinates file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            self.stationsPath.setText(selected[0])
            self.set_dataless_dir(self.stationsPath.text())

    def set_dataless_dir(self, dir_path):
        self.__dataless_dir = dir_path
        self.nll_manager.set_dataless_dir(dir_path)

    def set_pick_output_path(self, file_path):
        self.__pick_output_path = file_path
        self.nll_manager.set_observation_file(file_path)

    def info_message(self, msg, detailed_message=None):
        md = MessageDialog(self)
        md.set_info_message(msg, detailed_message)

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
                md.set_info_message("Completed Successfully.")

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_run_vel_to_grid(self):
        self.nll_manager.vel_to_grid(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                                     self.grid_depth_bind.value, self.grid_xnode_bind.value,
                                     self.grid_ynode_bind.value, self.grid_znode_bind.value,
                                     self.grid_dxsize_bind.value, self.grid_dysize_bind.value,
                                     self.grid_dzsize_bind.value, self.comboBox_gridtype.currentText(),
                                     self.comboBox_wavetype.currentText(),self.modelCB.currentText())

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_run_grid_to_time(self):


        if self.distanceSB.value()>0:
            limit = self.distanceSB.value()
        else:
            limit = np.sqrt((self.grid_xnode_bind.value * self.grid_dxsize_bind.value) ** 2 +
                            (self.grid_xnode_bind.value * self.grid_dxsize_bind.value) ** 2)

        self.nll_manager.grid_to_time(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                                      self.grid_depth_bind.value, self.comboBox_grid.currentText(),
                                      self.comboBox_angles.currentText(), self.comboBox_ttwave.currentText(),limit)

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg, set_default_complete=False))
    def on_click_run_loc(self):
        transform = self.transCB.currentText()
        std_out = self.nll_manager.run_nlloc(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                                             self.grid_depth_bind.value,transform)
        self.info_message("Location complete. Check details.", std_out)

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_plot_map(self):
        origin = self.nll_manager.get_NLL_info()
        scatter_x, scatter_y, scatter_z, pdf = self.nll_manager.get_NLL_scatter()
        lat = origin.latitude
        lon = origin.longitude
        stations = self.nll_manager.stations_match()

        self.cartopy_canvas.plot_map(lon, lat, scatter_x, scatter_y, scatter_z, 0,
                                     resolution= 'high', stations = stations)
        # Writing Location information
        self.add_earthquake_info(origin)
        xp, yp, xs, ys = self.nll_manager.ger_NLL_residuals()
        self.plot_residuals(xp, yp, xs, ys)



    def plot_pdf(self):
        scatter_x, scatter_y, scatter_z, pdf = self.nll_manager.get_NLL_scatter()
        self.pdf = PDFmanger(scatter_x, scatter_y, scatter_z, pdf)
        self.pdf.plot_scatter()

    def plot_residuals(self, xp, yp, xs, ys):

        artist = self.residuals_canvas.plot(xp, yp, axes_index=0, linewidth=0.5)
        self.residuals_canvas.set_xlabel(0, "Station")
        self.residuals_canvas.set_ylabel(0, "P wave Res")
        self.residuals_canvas.set_yaxis_color(self.residuals_canvas.get_axe(0), artist.get_color(), is_left=True)
        self.residuals_canvas.plot(xs, ys, 0, is_twinx=True, color="red", linewidth=0.5)
        self.residuals_canvas.set_ylabel_twinx(0, "S wave Res")
        self.residuals_canvas.plot(xp, yp, axes_index=0, linewidth=0.5)

    def first_polarity(self):
        import pandas as pd
        path_output = os.path.join(ROOT_DIR, "earthquakeAnalisysis", "location_output", "loc", "first_polarity.fp")
        self.firstpolarity_manager.create_input()
        self.firstpolarity_manager.run_focmec()
        Station, Az, Dip, Motion = self.firstpolarity_manager.get_dataframe()
        cat,Plane_A=self.firstpolarity_manager.extract_focmec_info()
        #print(cat[0].focal_mechanisms[0])
        strike_A = Plane_A.strike
        dip_A = Plane_A.dip
        rake_A = Plane_A.rake
        misfit_first_polarity = cat[0].focal_mechanisms[0].misfit
        azimuthal_gap = cat[0].focal_mechanisms[0].azimuthal_gap
        number_of_polarities = cat[0].focal_mechanisms[0].station_polarity_count

        first_polarity_results = {"First_Polarity":["Strike", "Dip", "Rake","misfit_first_polarity","azimuthal_gap",
         "number_of_polarities"],"results":[strike_A,dip_A,rake_A,misfit_first_polarity,azimuthal_gap,
                                            number_of_polarities]}
        df = pd.DataFrame(first_polarity_results, columns=["First_Polarity","results"])
        df.to_csv(path_output, sep=' ', index=False)
        self.focmec_canvas.drawFocMec(strike_A, dip_A, rake_A, Station, Az, Dip, Motion, 0)
        self.add_first_polarity_info(first_polarity_results)


    def add_first_polarity_info(self, first_polarity_results):
        self.FirstPolarityInfoText.setPlainText("First Polarity Results")
        self.FirstPolarityInfoText.appendPlainText("Strike: {Strike:.3f}".format(Strike=first_polarity_results["results"][0]))
        self.FirstPolarityInfoText.appendPlainText("Dip: {Dip:.3f}".format(Dip=first_polarity_results["results"][1]))
        self.FirstPolarityInfoText.appendPlainText("Rake: {Rake:.3f}".format(Rake=first_polarity_results["results"][2]))
        self.FirstPolarityInfoText.appendPlainText("Misfit: {Misfit:.3f}".format(Misfit=first_polarity_results["results"][3]))
        self.FirstPolarityInfoText.appendPlainText("GAP: {GAP:.3f}".format(GAP=first_polarity_results["results"][4]))
        self.FirstPolarityInfoText.appendPlainText("Number of polarities: {NP:.3f}".format(NP=first_polarity_results["results"][5]))


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
                                                       unc=origin.depth_errors['uncertainty']))
        self.EarthquakeInfoText.appendPlainText("  Horizontal Ellipse:     Max Horizontal Err {:.3f}     "
                                                "Min Horizontal Err {:.3f}     "
                                                "Azimuth {:.3f}"
                                                .format(origin.origin_uncertainty.max_horizontal_uncertainty,
                                                        origin.origin_uncertainty.min_horizontal_uncertainty,
                                                        origin.origin_uncertainty.azimuth_max_horizontal_uncertainty))

        self.EarthquakeInfoText.appendPlainText("  Quality Parameters:     Number of Phases {:.3f}     "
                                                "Azimuthal GAP {:.3f}     Minimum Distance {:.3f}     "
                                                "Maximum Distance {:.3f}"
                                                .format(origin.quality.used_phase_count,
                                                        origin.quality.azimuthal_gap,
                                                        origin.quality.minimum_distance,
                                                        origin.quality.maximum_distance))
