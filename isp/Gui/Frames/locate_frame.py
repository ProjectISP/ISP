#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
locate_frame
"""

import os
import shutil

from PyQt5.QtWidgets import QPushButton, QVBoxLayout
from obspy import Inventory
from obspy.core.event import Origin
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import parse_excepts
from isp.Gui import pw
from isp.Gui.Frames import BaseFrame, MessageDialog, CartopyCanvas, FocCanvas
from isp.Gui.Frames.uis_frames import UiLocFlow
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.LocCore.pdf_plot import plot_scatter
from isp.LocCore.plot_tool_loc import StationUtils
from isp.Utils import ObspyUtil
from isp.earthquakeAnalysis import NllManager, FirstPolarity
from isp.earthquakeAnalysis.focmecobspy import parse_focmec_file
from isp.earthquakeAnalysis.run_nll import Nllcatalog
from isp.earthquakeAnalysis.structures import TravelTimesConfiguration, LocationParameters, NLLConfig, \
    GridConfiguration
from sys import platform

@add_save_load()
class Locate(BaseFrame, UiLocFlow):
    def __init__(self, inv_path: str):
        super(Locate, self).__init__()
        self.setupUi(self)

        """
        Locate Event Frame

        :param params required to initialize the class:

        """

        self.datalessPathForm.setText(inv_path)
        self.inventory = None
        ####### Metadata ##########
        # TODO ON SELECT METADATA CREATES SELF.INVENTORY
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


        # Map
        self.cartopy_canvas = CartopyCanvas(self.widget_map, constrained_layout=True)
        # FocMec
        self.focmec_canvas = FocCanvas(self.widget_focmec)

        self.resultsShow.stateChanged.connect(lambda: self.show_results())

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

    # def on_click_select_directory(self, bind: BindPyqtObject):
    #     dir_path = self._select_directory(bind)
    #     if dir_path:
    #         bind.value = dir_path

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

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg,
                                                              False))  # When launch, metadata path need show messsage to False.
    def onChange_metadata_path(self, value):

        try:
            self.__metadata_manager = MetadataManager(value)
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
                print(file_hyp)
                scatter_x, scatter_y, scatter_z, pdf = NllManager.get_NLL_scatter(file_hyp)
                plot_scatter(scatter_x, scatter_y, scatter_z, pdf, ellipse)

    def add_earthquake_info(self, origin: Origin):
        self.EarthquakeInfoText.clear()
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

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    def on_click_plot_map(self):

        file_hyp = self.__selected_file()
        print("Plotting ", file_hyp)
        if file_hyp is not None:

            if self.inventory is None:
                metadata_manager = MetadataManager(self.metadata_path_bind.value)
                self.inventory: Inventory = metadata_manager.get_inventory()

            origin, event = ObspyUtil.reads_hyp_to_origin(file_hyp, modified=True)
            stations = StationUtils.get_station_location_dict(event, self.inventory)
            self.add_earthquake_info(origin)
            self.cartopy_canvas.clear()
            if self.topoCB.isChecked():
                resolution = 'high'
            else:
                resolution = 'simple'
            self.cartopy_canvas.plot_map(origin.longitude, origin.latitude,0,
                                         resolution=resolution, stations=stations)
            self.plot_pdf(file_hyp)


    def saveLoc(self):

        md = MessageDialog(self)
        dir_output_path = os.path.join(self.loc_work_bind.value, "savedLocs")

        nllcatalog = Nllcatalog(self.loc_work_bind.value)
        nllcatalog.find_files()
        files_list = nllcatalog.obsfiles
        if files_list is not None:
            if os.path.isdir(dir_output_path):
                pass
            else:
                os.makedirs(dir_output_path)

            for file in files_list:
                shutil.move(file, dir_output_path)

            md.set_info_message("Saved Location files", dir_output_path)
        else:
            md.set_info_message("No files to save", "No *.hyp files inside " + self.loc_work_bind.value)




    def first_polarity(self):

        file_last = self.__get_last_hyp()

        if file_last is not None:
            print("Plotting Map")
            firstpolarity_manager = FirstPolarity()
            file_input = firstpolarity_manager.create_input(file_last)
            #if file_input is not None:
            firstpolarity_manager.run_focmec(file_input, self.accepted_polarities.value())


            #df = pd.DataFrame(first_polarity_results, columns=["First_Polarity", "results"])
            #df.to_csv(path_output, sep=' ', index=False)



    def pltFocMec(self):
        location_file = self.__get_last_hyp()
        focmec_file = self.__get_last_focmec()
        if location_file is not None:
            print("Plotting Map")
        firstpolarity_manager = FirstPolarity()
        Station, Az, Dip, Motion = firstpolarity_manager.get_dataframe(location_file)
        cat, focal_mechanism = firstpolarity_manager.extract_focmec_info(focmec_file)
        focmec_full_Data = parse_focmec_file(focmec_file)


        # #print(cat[0].focal_mechanisms[0])
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
        if os.path.isdir(self.loc_work_bind.value):
            nllcatalog = Nllcatalog(self.loc_work_bind.value)
            nllcatalog.find_files()
            files_list = nllcatalog.obsfiles
            for file in files_list:
                try:
                    root_file = os.path.basename(file)
                    self.locFilesQTW.setRowCount(self.locFilesQTW.rowCount() + 1)


                    item = pw.QTableWidgetItem()
                    item.setData(0, root_file)
                    self.locFilesQTW.setItem(self.locFilesQTW.rowCount() - 1, 0, pw.QTableWidgetItem(root_file))
                    #check = pw.QCheckBox()
                    #self.tw_files.setCellWidget(self.locFilesQTW.rowCount() - 1, 3, check)

                except Exception:
                    pass

    def __selected_file(self):
        row = self.locFilesQTW.currentRow()
        file = os.path.join(self.loc_work_bind.value, "loc", self.locFilesQTW.item(row, 0).data(0))
        return file