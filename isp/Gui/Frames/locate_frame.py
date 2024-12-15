#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: run_nll.py
# Program: surfQuake & ISP
# Date: December 2024
# Purpose: Manage Event Locator
# Author: Roberto Cabieces & Thiago C. Junqueira
# Email: rcabdia@roa.es
# --------------------------------------------------------------------

"""
locate_frame
"""

import os
import shutil
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import  QCheckBox
from obspy import Inventory
from obspy.core.event import Origin
from surfquakecore.magnitudes.source_tools import ReadSource
from surfquakecore.project.surf_project import SurfProject
from isp import PICKING_DIR, LOCATION_OUTPUT_PATH, LOC_STRUCTURE, source_config
from surfquakecore.magnitudes.run_magnitudes import Automag
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import parse_excepts
from isp.Gui import pw, pyc
from isp.Gui.Frames import BaseFrame, MessageDialog, CartopyCanvas, FocCanvas
from isp.Gui.Frames.uis_frames import UiLocFlow
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.LocCore.pdf_plot import plot_scatter
from isp.LocCore.plot_tool_loc import StationUtils
from isp.Utils import ObspyUtil, AsycTime
from isp.earthquakeAnalysis import NllManager, FirstPolarity
from isp.earthquakeAnalysis.run_nll import Nllcatalog
from isp.earthquakeAnalysis.structures import TravelTimesConfiguration, LocationParameters, NLLConfig, \
    GridConfiguration
from sys import platform

@add_save_load()
class Locate(BaseFrame, UiLocFlow):
    def __init__(self, inv_path: str, project: SurfProject):
        super(Locate, self).__init__()
        self.setupUi(self)

        """
        Locate Event Frame

        :param params required to initialize the class:

        """
        self.config_automag = {}
        self.sp = project
        self.datalessPathForm.setText(inv_path)
        self.inventory = None
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm)
        self.setMetaBtn.clicked.connect(lambda: self.on_click_select_file(self.metadata_path_bind))

        # Binds
        self.loc_work_bind = BindPyqtObject(self.loc_workLE, self.onChange_root_pathLoc)

        # NonLinLoc
        self.grid_latitude_bind = BindPyqtObject(self.gridlatSB)
        self.grid_longitude_bind = BindPyqtObject(self.gridlonSB)
        self.grid_depth_bind = BindPyqtObject(self.griddepthSB)
        self.grid_xnode_bind = BindPyqtObject(self.xnodeSB)
        self.grid_ynode_bind = BindPyqtObject(self.ynodeSB)
        self.grid_znode_bind = BindPyqtObject(self.znodeSB)
        self.grid_dxsize_bind = BindPyqtObject(self.dxsizeSB)
        self.grid_dysize_bind = BindPyqtObject(self.dysizeSB)
        self.grid_dzsize_bind = BindPyqtObject(self.dzsizeSB)

        self.loc_work_dirBtn.clicked.connect(lambda: self.on_click_select_directory(self.loc_work_bind))

        self.model_path_bind = BindPyqtObject(self.modelLE, self.onChange_root_path)
        self.modelPathBtn.clicked.connect(lambda: self.on_click_select_directory(self.model_path_bind))
        self.picks_path_bind = BindPyqtObject(self.picksLE, self.onChange_root_path)
        self.picksBtn.clicked.connect(lambda: self.on_click_select_file(self.picks_path_bind))
        self.genvelBtn.clicked.connect(lambda: self.on_click_run_vel_to_grid())
        self.grdtimeBtn.clicked.connect(lambda: self.on_click_run_grid_to_time())
        self.runlocBtn.clicked.connect(lambda: self.on_click_run_loc())
        self.pltMap.clicked.connect(lambda: self.on_click_plot_map())
        self.runFocMecBtn.clicked.connect(lambda: self.first_polarity())
        self.pltFocMecBtn.clicked.connect(lambda: self.pltFocMec())
        self.saveLocBtn.clicked.connect(lambda: self.saveLoc())
        self.saveMecBtn.clicked.connect(lambda: self.saveMec())
        self.setDefaultPathBtn.clicked.connect(lambda: self.setDefault())
        self.loadMetaBtn.clicked.connect(lambda: self.reloadMetadata())
        self.pltFocMecLocBtn.clicked.connect(lambda: self.pltFocMec(set_page=1))

        # Magnitude
        self.source_locs_bind = BindPyqtObject(self.source_locsLE)
        self.setLocFolderBtn.clicked.connect(lambda: self.on_click_select_directory(self.source_locs_bind))

        self.source_out_bind = BindPyqtObject(self.source_outLE)
        self.setSourceOutBtn.clicked.connect(lambda: self.on_click_select_directory(self.source_out_bind))
        self.mag_runBtn.clicked.connect(lambda: self.run_automag())
        self.printSourceResultsBtn.clicked.connect(lambda: self.print_source_results())


        # Map
        self.cartopy_canvas = CartopyCanvas(self.widget_map, constrained_layout=True)
        # FocMec
        self.focmec_canvas = FocCanvas(self.widget_focmec)

        self.resultsShow.stateChanged.connect(lambda: self.show_results())
        self.selectAllLocCB.stateChanged.connect(lambda: self.check_all_checkboxes_Loc())
        self.selectAllFocCB.stateChanged.connect(lambda: self.check_all_checkboxes_Mec())

        # Dialog
        self.progress_dialog = pw.QProgressDialog(self)
        self.progress_dialog.setRange(0, 0)
        self.progress_dialog.setWindowTitle('Processing.....')
        self.progress_dialog.setLabelText('Please Wait')
        self.progress_dialog.setWindowIcon(self.windowIcon())
        self.progress_dialog.setWindowTitle(self.windowTitle())
        self.progress_dialog.close()

    def check_all_checkboxes_Loc(self):
        # Iterate through the rows and check all checkboxes
        if not self.selectAllLocCB.checkState():
            for row in range(self.locFilesQTW.rowCount()):
                cell_widget = self.locFilesQTW.cellWidget(row, 1)
                if isinstance(cell_widget, QCheckBox):
                    cell_widget.setChecked(False)
        else:
            for row in range(self.locFilesQTW.rowCount()):
                cell_widget = self.locFilesQTW.cellWidget(row, 1)
                if isinstance(cell_widget, QCheckBox):
                    cell_widget.setChecked(True)

    def check_all_checkboxes_Mec(self):
        # Iterate through the rows and check all checkboxes
        if not self.selectAllFocCB.checkState():
            for row in range(self.focmecTW.rowCount()):
                cell_widget = self.focmecTW.cellWidget(row, 1)
                if isinstance(cell_widget, QCheckBox):
                    cell_widget.setChecked(False)
        else:
            for row in range(self.focmecTW.rowCount()):
                cell_widget = self.focmecTW.cellWidget(row, 1)
                if isinstance(cell_widget, QCheckBox):
                    cell_widget.setChecked(True)

    def show_results(self):
        if not self.resultsShow.checkState():
            self.EarthquakeInfoText.setMaximumHeight(0)
        else:
            self.EarthquakeInfoText.setMaximumHeight(150)

    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        pass


    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path
    def _select_directory(self, bind: BindPyqtObject):

        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        return dir_path

    def on_click_select_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg, False))  # When launch, metadata path need show messsage to False.
    def onChange_metadata_path(self, value):

        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory: Inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
        except:
            raise FileNotFoundError("The metadata is not valid")

    def reloadMetadata(self):
        try:
            self.__metadata_manager = MetadataManager(self.datalessPathForm.text())
            self.inventory: Inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
        except:
            raise FileNotFoundError("The metadata is not valid")

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

    def get_nll_config(self):

        if self.loc_wavetypeCB.currentText() == "P & S":
            p_wave_type = True
            s_wave_type = True
        else:
            p_wave_type = True
            s_wave_type = True

        if self.loc_modelCB.currentText() == "1D":
            path_model1D = self.model_path_bind.value
            path_model3D = "NONE"
        else:
            path_model1D = "NONE"
            path_model3D = self.picks_path_bind.value

        nllconfig = NLLConfig(
            grid_configuration=GridConfiguration(
                latitude=self.grid_latitude_bind.value,
                longitude=self.grid_longitude_bind.value,
                depth=self.grid_depth_bind.value,
                x=self.grid_xnode_bind.value,
                y=self.grid_ynode_bind.value,
                z=self.grid_znode_bind.value,
                dx=self.grid_dxsize_bind.value,
                dy=self.grid_dysize_bind.value,
                dz=self.grid_dzsize_bind.value,
                geo_transformation=self.transCB.currentText(),
                grid_type=self.gridtype.currentText(),
                path_to_1d_model=path_model1D,
                path_to_3d_model=path_model3D,
                path_to_picks=self.picks_path_bind.value,
                p_wave_type=p_wave_type,
                s_wave_type=s_wave_type,
                model=self.loc_modelCB.currentText()),
            travel_times_configuration=TravelTimesConfiguration(
                distance_limit=self.distanceSB.value(),
                grid=self.grid_typeCB.currentText()[4:6]),
            location_parameters=LocationParameters(
                search=self.loc_searchCB.currentText(),
                method=self.loc_methodCB.currentText()))

        return nllconfig

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_run_vel_to_grid(self):
        nllconfig = self.get_nll_config()
        if isinstance(nllconfig, NLLConfig):
            nll_manager = NllManager(nllconfig, self.metadata_path_bind.value, self.loc_work_bind.value)
            nll_manager.vel_to_grid()

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_run_grid_to_time(self):
        nllconfig = self.get_nll_config()
        if isinstance(nllconfig, NLLConfig):
            nll_manager = NllManager(nllconfig, self.metadata_path_bind.value, self.loc_work_bind.value)
            nll_manager.grid_to_time()

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_run_loc(self):
        nllconfig = self.get_nll_config()
        if isinstance(nllconfig, NLLConfig):
            nll_manager = NllManager(nllconfig, self.metadata_path_bind.value, self.loc_work_bind.value)
            nll_manager.clean_output_folder()
            for i in range(self.iterationsSB.value()):
                print("Running Location iteration", i)
                nll_manager.run_nlloc()
            nll_catalog = Nllcatalog(self.loc_work_bind.value)
            nll_catalog.run_catalog(self.loc_work_bind.value)
            self.onChange_root_pathLoc("refress")

    def plot_pdf(self, file_hyp):

        #selected = pw.QFileDialog.getOpenFileName(self, "Select *hyp file")
        if isinstance(file_hyp, str) and os.path.isfile(file_hyp):
            ellipse = {}

            origin: Origin = ObspyUtil.reads_hyp_to_origin(file_hyp)
            latitude = origin.latitude
            longitude = origin.longitude
            smin = origin.origin_uncertainty.min_horizontal_uncertainty
            smax = origin.origin_uncertainty.max_horizontal_uncertainty
            azimuth = origin.origin_uncertainty.azimuth_max_horizontal_uncertainty
            ellipse['latitude'] = latitude
            ellipse['longitude'] = longitude
            ellipse['smin'] = smin
            ellipse['smax'] = smax
            ellipse['azimuth'] = azimuth
            if os.path.isfile(file_hyp):
                scatter_x, scatter_y, scatter_z, pdf = NllManager.get_NLL_scatter(file_hyp)
                plot_scatter(scatter_x, scatter_y, scatter_z, pdf, ellipse)

    def add_earthquake_info(self, origin: Origin):
        self.EarthquakeInfoText.clear()
        self.EarthquakeInfoText.setPlainText("  Origin time and RMS:     {origin_time}     {standard_error:.3f} s".
                                             format(origin_time=origin.time,
                                                    standard_error=origin.quality.standard_error))
        self.EarthquakeInfoText.appendPlainText("  Hypocenter Geographic Coordinates:     "
                                                "Latitude {lat:.3f}º "
                                                "Longitude {long:.3f}º     Depth {depth:.3f} km    "
                                                "Uncertainty {unc:.3f} km".
                                                format(lat=origin.latitude, long=origin.longitude,
                                                       depth=origin.depth / 1000,
                                                       unc=origin.depth_errors['uncertainty']))
        self.EarthquakeInfoText.appendPlainText("  Horizontal Ellipse:     Max Horizontal Err {:.3f} km     "
                                                "Min Horizontal Err {:.3f} km    "
                                                "Azimuth {:.3f} º"
                                                .format(origin.origin_uncertainty.max_horizontal_uncertainty,
                                                        origin.origin_uncertainty.min_horizontal_uncertainty,
                                                        origin.origin_uncertainty.azimuth_max_horizontal_uncertainty))

        self.EarthquakeInfoText.appendPlainText("  Quality Parameters:     Number of Phases {:.3f}     "
                                                "Azimuthal GAP {:.3f} º     Minimum Distance {:.3f} km     "
                                                "Maximum Distance {:.3f} km"
                                                .format(origin.quality.used_phase_count,
                                                        origin.quality.azimuthal_gap,
                                                        origin.quality.minimum_distance,
                                                        origin.quality.maximum_distance))

    def __get_last_hyp(self):

        file_last = os.path.join(self.loc_work_bind.value, "loc", "last.hyp")

        # check
        if os.path.isfile(file_last):

            return file_last
        else:
            return None

    def __get_last_focmec(self):
        file_last = os.path.join(self.loc_work_bind.value, "first_polarity/output", "focmec.lst")
        return file_last

    def on_click_plot_map(self):

        file_hyp = self.__selected_file()
        print("Plotting ", file_hyp)
        if file_hyp is not None:

            if self.inventory is None:
                metadata_manager = MetadataManager(self.metadata_path_bind.value)
                self.inventory: Inventory = metadata_manager.get_inventory()

            origin, event = ObspyUtil.reads_hyp_to_origin(file_hyp, modified=True)
            stations = StationUtils.get_station_location_dict(event, self.inventory)
            N = len(stations)
            self.add_earthquake_info(origin)
            self.cartopy_canvas.clear()

            # get ellipse points
            x_ellipse, y_ellipse = ObspyUtil.get_ellipse(file_hyp)


            if self.topoCB.isChecked():
                resolution = 'high'
            else:
                resolution = 'simple'


            self.cartopy_canvas.plot_map(origin.longitude, origin.latitude,0,
                                         resolution=resolution, stations=stations)
            self.cartopy_canvas.plot_ellipse(x_ellipse, y_ellipse, axes_index=0)


            if self.pdfCB.isChecked():
                self.plot_pdf(file_hyp)

            if N==0:
                md = MessageDialog(self)
                md.set_info_message("Warning, refresh your metadata, no stations matching the *.hyp file")


    def saveLoc(self):
        md = MessageDialog(self)
        dir_output_path = os.path.join(self.loc_work_bind.value, "savedLocs")
        if os.path.isdir(dir_output_path):
            pass
        else:
            try:
                os.makedirs(dir_output_path)
            except Exception as error:
                print("An exception occurred:", error)

        row_count = self.locFilesQTW.rowCount()
        column_count = self.locFilesQTW.columnCount()
        moved_files = []
        for row in range(row_count):
            file = None
            for column in range(column_count):
                # If the column contains a QTableWidgetItem
                item = self.locFilesQTW.item(row, column)
                if column == 0:
                    file = os.path.join(self.loc_work_bind.value, "loc", item.text())
                # If the column contains a widget (e.g., QCheckBox)

                if column == 1:
                    cell_widget = self.locFilesQTW.cellWidget(row, column)
                    #if isinstance(cell_widget, pw.QCheckBox):
                    if cell_widget.isChecked() and os.path.isfile(file):
                        shutil.copy(file, dir_output_path)
                        moved_files.append(file)

        if len(moved_files) > 0:
            md.set_info_message("Saved Loc files", dir_output_path)
        else:
            md.set_info_message("No files to save")

    def saveMec(self):
        md = MessageDialog(self)
        dir_output_path = os.path.join(self.loc_work_bind.value, "savedMecs")

        if os.path.isdir(dir_output_path):
            pass
        else:
            try:
                os.makedirs(dir_output_path)
            except Exception as error:
                print("An exception occurred:", error)

        row_count = self.focmecTW.rowCount()
        column_count = self.focmecTW.columnCount()
        moved_files = []
        for row in range(row_count):
            file = None
            for column in range(column_count):
                # If the column contains a QTableWidgetItem
                item = self.focmecTW.item(row, column)
                if column == 0:
                    file = os.path.join(self.loc_work_bind.value, "first_polarity/output", item.text())
                # If the column contains a widget (e.g., QCheckBox)

                if column == 1:
                    cell_widget = self.focmecTW.cellWidget(row, column)
                    # if isinstance(cell_widget, pw.QCheckBox):
                    if cell_widget.isChecked() and os.path.isfile(file):
                        shutil.copy(file, dir_output_path)
                        moved_files.append(file)

        if len(moved_files) > 0:
            md.set_info_message("Saved Loc files", dir_output_path)
        else:
            md.set_info_message("No files to save")

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def first_polarity(self):

        nllcatalog = Nllcatalog(self.loc_work_bind.value)
        nllcatalog.find_files()
        files_list = nllcatalog.obsfiles
        for file in files_list:
            try:
                header = FirstPolarity.set_head(file)
                if file is not None:
                    firstpolarity_manager = FirstPolarity()
                    file_input = firstpolarity_manager.create_input(file, header)
                    firstpolarity_manager.run_focmec(file_input, self.accepted_polarities.value())
            except:
                pass
        self.onChange_root_pathLoc("refress")


    def pltFocMec(self, set_page=None):

        if set_page is None:
            focmec_file = self.__focmec_file()
        else:
            self.SurfQuakeWidget.setCurrentIndex(2)
            focmec_file = self.__focmec_file(from_focmec=False)

        firstpolarity_manager = FirstPolarity()
        #Station, Az, Dip, Motion = firstpolarity_manager.get_dataframe(location_file)
        Station, Az, Dip, Motion = FirstPolarity.extract_station_data(focmec_file)
        cat, focal_mechanism = firstpolarity_manager.extract_focmec_info(focmec_file)
        # TODO MIGHT BE FOR PLOTTING ALL POSSIBLE FAUL PLANES
        # focmec_full_Data = parse_focmec_file(focmec_file)
        # file_output_name = FirstPolarity.extract_name(focmec_file)

        Plane_A = focal_mechanism.nodal_planes.nodal_plane_1
        strike_A = Plane_A.strike
        dip_A = Plane_A.dip
        rake_A = Plane_A.rake
        extra_info = firstpolarity_manager.parse_solution_block(focal_mechanism.comments[0]["text"])
        P_Trend = extra_info['P,T']['Trend']
        P_Plunge = extra_info['P,T']['Plunge']
        T_Trend = extra_info['P,N']['Trend']
        T_Plunge = extra_info['P,N']['Plunge']

        misfit_first_polarity = focal_mechanism.misfit
        azimuthal_gap = focal_mechanism.azimuthal_gap
        number_of_polarities = focal_mechanism.station_polarity_count
        #
        first_polarity_results = {"First_Polarity": ["Strike", "Dip", "Rake", "misfit_first_polarity", "azimuthal_gap",
                                                     "number_of_polarities", "P_axis_Trend", "P_axis_Plunge",
                                                     "T_axis_Trend", "T_axis_Plunge"],
                                  "results": [strike_A, dip_A, rake_A, misfit_first_polarity, azimuthal_gap,
                                              number_of_polarities, P_Trend, P_Plunge, T_Trend, T_Plunge]}

        self.add_first_polarity_info(first_polarity_results)
        self.focmec_canvas.clear()
        self.focmec_canvas.drawFocMec(strike_A, dip_A, rake_A, Station, Az, Dip, Motion, P_Trend, P_Plunge,
                                      T_Trend, T_Plunge)
        self.focmec_canvas.figure.subplots_adjust(left=0.250, bottom=0.105, right=0.725, top=0.937, wspace=0.0,
                                                  hspace=0.0)

    def add_first_polarity_info(self, first_polarity_results):
        self.FirstPolarityInfoText.clear()
        self.FirstPolarityInfoText.setPlainText("First Polarity Results")
        self.FirstPolarityInfoText.appendPlainText("Strike: {Strike:.3f}".format(Strike=first_polarity_results["results"][0]))
        self.FirstPolarityInfoText.appendPlainText("Dip: {Dip:.3f}".format(Dip=first_polarity_results["results"][1]))
        self.FirstPolarityInfoText.appendPlainText("Rake: {Rake:.3f}".format(Rake=first_polarity_results["results"][2]))
        self.FirstPolarityInfoText.appendPlainText("P axis trend & plunge: {Ptrend:.1f} {Pplunge:.1f}".
                                                   format(Ptrend=first_polarity_results["results"][6],
                                                          Pplunge=first_polarity_results["results"][7]))
        self.FirstPolarityInfoText.appendPlainText("T axis trend & plunge: {Ptrend:.1f} {Pplunge:.1f}".
                                                   format(Ptrend=first_polarity_results["results"][8],
                                                          Pplunge=first_polarity_results["results"][9]))
        self.FirstPolarityInfoText.appendPlainText("Misfit: {Misfit:.3f}".format(Misfit=first_polarity_results["results"][3]))
        self.FirstPolarityInfoText.appendPlainText("GAP: {GAP:.3f}".format(GAP=first_polarity_results["results"][4]))
        self.FirstPolarityInfoText.appendPlainText("Number of polarities: {NP:.3f}".format(NP=first_polarity_results["results"][5]))


    def onChange_root_pathLoc(self, value):

        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """

        self.locFilesQTW.clearContents()
        self.focmecTW.clearContents()


        if os.path.isdir(self.loc_work_bind.value):
            nllcatalog = Nllcatalog(self.loc_work_bind.value)
            nllcatalog.find_files()
            files_list = nllcatalog.obsfiles

            if len(files_list) > 0:
                self.locFilesQTW.setRowCount(0)

            files_focmec = FirstPolarity.find_files(self.loc_work_bind.value)

            if len(files_focmec) > 0:
                self.focmecTW.setRowCount(0)

            for file in files_list:
                try:
                    root_file = os.path.basename(file)
                    self.locFilesQTW.setRowCount(self.locFilesQTW.rowCount() + 1)

                    item = pw.QTableWidgetItem()
                    item.setData(0, root_file)

                    foc_mec_match = FirstPolarity.find_loc_mec_file(file)

                    if foc_mec_match is not None:
                        self.locFilesQTW.setItem(self.locFilesQTW.rowCount() - 1, 1, pw.QTableWidgetItem(foc_mec_match))
                    check = pw.QCheckBox()
                    self.locFilesQTW.setItem(self.locFilesQTW.rowCount() - 1, 0, pw.QTableWidgetItem(root_file))
                    self.locFilesQTW.setCellWidget(self.locFilesQTW.rowCount() - 1, 2, check)

                except Exception:
                    pass

            for file in files_focmec:
                try:
                    root_file = os.path.basename(file)
                    self.focmecTW.setRowCount(self.focmecTW.rowCount() + 1)

                    item = pw.QTableWidgetItem()
                    item.setData(0, root_file)
                    check = pw.QCheckBox()
                    self.focmecTW.setItem(self.focmecTW.rowCount() - 1, 0, pw.QTableWidgetItem(root_file))
                    self.focmecTW.setCellWidget(self.focmecTW.rowCount() - 1, 1, check)
                except Exception:
                    pass


    def setDefault(self):
        self.picksLE.setText(os.path.join(PICKING_DIR, "output.txt"))
        self.loc_workLE.setText(LOCATION_OUTPUT_PATH)
        self.modelLE.setText(os.path.join(LOC_STRUCTURE, "local_models"))

    def __selected_file(self):
        row = self.locFilesQTW.currentRow()
        file = os.path.join(self.loc_work_bind.value, "loc", self.locFilesQTW.item(row, 0).data(0))
        return file

    def __focmec_file(self, from_focmec=True):
        if from_focmec:
            row = self.focmecTW.currentRow()
            file = os.path.join(self.loc_work_bind.value, "first_polarity/output", self.focmecTW.item(row, 0).data(0))
        else:
            row = self.locFilesQTW.currentRow()
            file = os.path.join(self.loc_work_bind.value, "first_polarity/output", self.locFilesQTW.item(row, 1).data(0))
        return file



    ####### Source Parameters ########
    def run_automag(self):
        self.__send_run_automag()
        self.progress_dialog.exec()
        md = MessageDialog(self)
        md.set_info_message("Source Parameters estimation finished, Please see output directory and press "
                            "print results")

    @AsycTime.run_async()
    def __send_run_automag(self):

        self.__load_config_automag()
        # Running stage
        mg = Automag(self.sp, self.source_locs_bind.value, self.metadata_path_bind.value, source_config,
                     self.source_out_bind.value, scale="regional", gui_mod=self.config_automag)
        print("Estimating Source Parameters")
        mg.estimate_source_parameters()

        # write a txt summarizing the results
        rs = ReadSource(self.source_out_bind.value)
        summary = rs.generate_source_summary()
        summary_path = os.path.join(self.source_out_bind.value, "source_summary.txt")
        rs.write_summary(summary, summary_path)
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)

    def __load_config_automag(self):
        self.config_automag['epi_dist_ranges'] = [0, self.mag_max_distDB.value()]
        self.config_automag['p_arrival_tolerance'] = self.p_tolDB.value()
        self.config_automag['s_arrival_tolerance'] = self.s_tolDB.value()
        self.config_automag['noise_pre_time'] = self.noise_windowDB.value()
        self.config_automag['win_length'] = self.signal_windowDB.value()
        self.config_automag['spectral_win_length'] = self.spec_windowSB.value()
        self.config_automag['rho_source'] = self.automag_density_DB.value()
        self.config_automag['rpp'] = self.automag_rppDB.value()
        self.config_automag['rps'] = self.automag_rpsDB.value()

        if self.r_power_nRB.isChecked():
            self.config_automag['geom_spread_model'] = "r_power_n"
        else:
            self.config_automag['geom_spread_model'] = "boatwright"

        self.config_automag['geom_spread_n_exponent'] = self.geom_spread_n_exponentDB.value()
        self.config_automag['geom_spread_cutoff_distance'] = self.geom_spread_cutoff_distanceDB.value()

        self.config_automag['a'] = self.mag_aDB.value()
        self.config_automag['b'] = self.mag_bDB.value()
        self.config_automag['c'] = self.mag_cDB.value()
        print("Loaded Source Config from GUI")

    def print_source_results(self):
        import pandas as pd
        import math
        self.automagnitudesText.clear()
        summary_path = os.path.join(self.source_out_bind.value, "source_summary.txt")
        df = pd.read_csv(summary_path, sep=";", na_values='missing')

        for index, row in df.iterrows():
            self.automagnitudesText.appendPlainText("#####################################################")
            date = row['date_id']
            lat = str(row['lats'])
            lon = str(row['longs'])
            depth = str(row['depths'])
            if not math.isnan(row['Mw']):
                Mw = str("{: .2f}".format(row['Mw']))
            else:
                Mw = row['Mw']

            if not math.isnan(row['Mw_error']):
                Mw_std = str("{: .2f}".format(row['Mw_error']))
            else:
                Mw_std = row['Mw_error']

            if not math.isnan(row['Mo']):
                Mo = str("{: .2e}".format(row['Mo']))
            else:
                Mo = row['Mo']

            if not math.isnan(row['radius']):
                source_radius = str("{: .2f}".format(row['radius']))
            else:
                source_radius = row['radius']

            if not math.isnan(row['ML']):
                ML = str("{: .2f}".format(row['ML']))
            else:
                ML = row['ML']

            if not math.isnan(row['ML_error']):
                ML_std = str("{: .2f}".format(row['ML_error']))
            else:
                ML_std = row['ML_error']

            if not math.isnan(row['bsd']):
                bsd = str("{: .2f}".format(row['bsd']))
            else:
                bsd = row['bsd']

            if not math.isnan(row['Er']):
                Er = str("{: .2e}".format(row['Er']))
            else:
                Er = row['Er']

            if not math.isnan(row['Er_std']):
                Er_std = str("{: .2e}".format(row['Er_std']))
            else:
                Er_std = row['Er']

            if not math.isnan(row['fc']):
                fc = str("{: .2f}".format(row['fc']))
            else:
                fc = row['fc']

            if not math.isnan(row['fc_std']):
                fc_std = str("{: .2f}".format(row['fc_std']))
            else:
                fc_std = row['fc']

            if not math.isnan(row['Qo']):
                Qo = str("{: .2f}".format(row['Qo']))
            else:
                Qo = row['Qo']

            if not math.isnan(row['Qo_std']):
                Qo_std = str("{: .2f}".format(row['Qo_std']))
            else:
                Qo_std = row['Qo_std']

            if not math.isnan(row['t_star']):
                t_star = str("{: .2f}".format(row['t_star']))
            else:
                t_star = row['t_star']

            if not math.isnan(row['t_star_std']):
                t_star_std = str("{: .2f}".format(row['t_star_std']))
            else:
                t_star_std = row['t_star_std']


            self.automagnitudesText.appendPlainText(date + "    " + lat +"º    "+ lon+"º    "+ depth+" km")
            self.automagnitudesText.appendPlainText("Moment Magnitude: " " Mw {Mw} "
                                                             " std {std} ".format(Mw=Mw, std=Mw_std))

            self.automagnitudesText.appendPlainText("Seismic Moment and Source radius: " " Mo {Mo:} Nm"
                                                              ", R {std} km".format(Mo=Mo, std=source_radius))

            self.automagnitudesText.appendPlainText("Local Magnitude: " " ML {ML} "
                                                    " std {std} ".format(ML=ML, std=ML_std))

            self.automagnitudesText.appendPlainText("Brune stress Drop: " "{bsd} MPa".format(bsd=bsd))

            self.automagnitudesText.appendPlainText(
                 "Seismic Energy: " " Er {Er} juls" " Er_std {Er_std} ".format(Er=Er, Er_std=Er_std))

            self.automagnitudesText.appendPlainText(
                 "Corner Frequency: " " fc {fc} Hz" " fc_std {fc_std} ".format(fc=fc, fc_std=fc_std))

            self.automagnitudesText.appendPlainText(
                          "Quality factor: " " Qo {Qo} " " Q_std {Qo_std} ".format(Qo=Qo, Qo_std=Qo_std))

            self.automagnitudesText.appendPlainText(
                          "t_star: " "{t_star} s" " t_star_std {t_star_std} ".format(t_star=t_star,
                                                                                             t_star_std=t_star_std))
