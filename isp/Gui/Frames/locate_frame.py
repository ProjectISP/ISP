#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
locate_frame
"""
import os

from obspy import Inventory

from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import parse_excepts
from isp.Gui import pw
from isp.Gui.Frames import BaseFrame, MessageDialog
from isp.Gui.Frames.uis_frames import UiLocFlow
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.earthquakeAnalysis import NllManager
from isp.earthquakeAnalysis.structures import TravelTimesConfiguration, LocationParameters, NLLConfig, \
    GridConfiguration
from sys import platform

@add_save_load()
class Locate(BaseFrame, UiLocFlow):
    def __init__(self, inv: Inventory):
        super(Locate, self).__init__()
        self.setupUi(self)

        """
        Locate Event Frame

        :param params required to initialize the class:

        """

        self.inv = inv

        ####### Metadata ##########
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm)
        self.setMetaBtn.clicked.connect(lambda: self.on_click_select_file(self.metadata_path_bind))
        self.loadMetaBtn.clicked.connect(lambda: self.onChange_metadata_path(self.metadata_path_bind.value))


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
        self.loc_work_bind = BindPyqtObject(self.loc_workLE)
        self.loc_work_dirBtn.clicked.connect(lambda: self.on_click_select_directory(self.loc_work_bind))
        self.model_path_bind = BindPyqtObject(self.modelLE, self.onChange_root_path)
        self.modelPathBtn.clicked.connect(lambda: self.on_click_select_directory(self.model_path_bind))
        self.picks_path_bind = BindPyqtObject(self.picksLE, self.onChange_root_path)
        self.picksBtn.clicked.connect(lambda: self.on_click_select_file(self.picks_path_bind))
        self.genvelBtn.clicked.connect(lambda: self.on_click_run_vel_to_grid())
        self.grdtimeBtn.clicked.connect(lambda: self.on_click_run_grid_to_time())
        self.runlocBtn.clicked.connect(lambda: self.on_click_run_loc())
        # self.plotmapBtn.clicked.connect(lambda: self.on_click_plot_map())
        # self.stationsBtn.clicked.connect(lambda: self.on_click_select_metadata_file())
        self.plotpdfBtn.clicked.connect(lambda: self.plot_pdf())


    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        pass

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = self._select_directory(bind)
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

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg, set_default_complete=False))
    def on_click_run_loc(self):
        nllconfig = self.get_nll_config()
        if isinstance(nllconfig, NLLConfig):
            nll_manager = NllManager(nllconfig, self.metadata_path_bind.value, self.loc_work_bind.value)
            for i in range(25):
                print("Running Location iteration", i)
                nll_manager.run_nlloc()
            #nll_catalog = Nllcatalog(self.loc_work_bind.value)
            #nll_catalog.run_catalog(self.loc_work_bind.value)

