import os
import pickle
import numpy as np
from platform import platform
from isp.Exceptions import parse_excepts
from isp.Gui import pw
from isp.Gui.Frames import MessageDialog, Pagination, MatplotlibCanvas, CartopyCanvas
from isp.Gui.Frames.uis_frames import UiDispersionMaps
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.Utils import AsycTime
from isp.ant.tomo_tools import tomotools
from isp.ant.process_ant import disp_maps_tools


@add_save_load()
class EGFDispersion(pw.QWidget, UiDispersionMaps):

    def __init__(self):
        super(EGFDispersion, self).__init__()
        self.setupUi(self)

        self.files = []
        self.total_items = 0
        self.items_per_page = 1
        self.stations_info = None
        self.root_path_bind = BindPyqtObject(self.rootPathForm)
        self.stations_path_bind = BindPyqtObject(self.StationsPathForm, self.onChange_stations_path)
        #self.output_path_bind = BindPyqtObject(self.OutputPathForm)

        self.read_pklBtn.clicked.connect(lambda: self.on_click_select_data_file(self.root_path_bind))
        self.stationsBtn.clicked.connect(lambda: self.on_click_select_data_file(self.stations_path_bind))
        #self.outputBtn.clicked.connect(lambda: self.on_click_select_directory(self.output_path_bind))

        self.createdispBtn.clicked.connect(self.traveltimes)
        self.pagination = Pagination(self.pagination_widget, self.total_items, self.items_per_page)
        self.pagination.set_total_items(0)
        self.pagination.set_lim_items_per_page()
        #self.pagination.items_per_page=6

        #self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=self.items_per_page, constrained_layout=False)
        self.cartopy_canvas = CartopyCanvas(self.plotMatWidget, nrows=self.items_per_page, constrained_layout=True)
        #self.canvas.figure.tight_layout()

        self.plotMapBtn.clicked.connect(self.plot_disp_maps)

    def get_files_at_page(self):
        n_0 = (self.pagination.current_page - 1) * self.pagination.items_per_page
        n_f = n_0 + self.pagination.items_per_page
        return self.files[n_0:n_f]

    def get_file_at_index(self, index):
        files_at_page = self.get_files_at_page()
        return files_at_page[index]

    def set_pagination_files(self, files_path):
        self.files = files_path
        self.total_items = len(self.files)
        self.pagination.set_total_items(self.total_items)


    def on_click_select_data_file(self, bind: BindPyqtObject):
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
                md.set_info_message("Loaded Data, please check your terminal for further details")

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    @AsycTime.run_async()
    def __get_disp_type(self):

        if self.wave_typeCB.currentText() == "Vertical":
            self.wave_type = ["ZZ"]
        elif self.wave_typeCB.currentText() == "Transversal":
            self.wave_type = ["TT"]
        elif self.wave_typeCB.currentText() == "Both":
            self.wave_type = ["TT", "ZZ"]

        if self.disp_typeCB.currentText() == "Group Velocity":
            self.dispersion_type = ["dsp"]
        elif self.disp_typeCB.currentText() == "Phase Velocity":
            self.dispersion_type = ["phv"]
        elif self.disp_typeCB.currentText() == "Both":
            self.dispersion_type = ["dsp","phv"]

        try:

            self.data_info = tomotools.read_dispersion(self.rootPathForm.text(), self.wave_type, self.dispersion_type)

        except:

            raise FileNotFoundError("The data info is not valid")


    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    @AsycTime.run_async()
    def onChange_stations_path(self, value):

        try:

            self.stations_info = tomotools.get_station_info(value)
            print(self.stations_info)

        except:

            raise FileNotFoundError("The stations file is not valid")


    def __gridding(self):

        md = MessageDialog(self)
        if isinstance(self.stations_info, tuple):
            try:
                self.grid = tomotools.create_grid(self.stations_info, gridy=self.grid_latSB.value(), gridx=self.grid_lonSB.value())
                print("Grid done!!")
                self.distance_matrix = tomotools.compute_distance_matrix(self.grid)
                print("Distance Matrix Done!!")
                md.set_info_message("Node Grids ready!!!!")
            except Exception as e:
                md.set_error_message("The Grid nodes is not completed")

        else:
            md.set_warning_message("Please load stations coordinates")

    def traveltimes(self):

        self.__get_disp_type()
        self.__gridding()

        for wave_type in self.wave_type:
            for dispersion_type in self.dispersion_type:
                for period in np.arange(self.min_periodCB.value(), self.max_periodCB.value()+1, self.periods_stepCB.value()):
                    result_dict = tomotools.traveltime_tomography(period, self.stations_info[0], self.stations_info[1], self.data_info
                    ,self.grid, self.distance_matrix, self.alphaCB.value(), self.betaCB.value(), self.gammaCB.value(), path_npts=150,
                            reg_lambda=0.1, density_pixel_size=0.05, checkerboard_test=self.checkerCB.isChecked())

                    tomotools.save_results(result_dict, period, wave_type=wave_type, dispersion_type=dispersion_type)


    def plot_disp_maps(self):

        self.files = disp_maps_tools.disp_maps_availables()
        self.set_pagination_files(self.files)
        files_at_page = self.get_files_at_page()
        files_at_page.sort()
        self.cartopy_canvas.clear()

        if len(self.cartopy_canvas.axes) != len(files_at_page):
            if len(files_at_page) == 1:
                 ncols = 1
            elif len(files_at_page) == 2:
                 ncols = 2
            else:
                ncols = 3
            self.cartopy_canvas.set_new_subplot_cartopy(nrows=int((len(files_at_page)-1)/3)+1, ncols=ncols)

        for k , grid_file in enumerate(files_at_page):
            #dispersion_ZZ_dsp_20.0s.pickle
             name = os.path.basename(grid_file)
             name_list = name.split("_")

             if name_list[1] == "ZZ":
                 wave_type = "Rayleigh"
             elif name_list[1] == "TT":
                 wave_type = "Love"

             if name_list[2] == "dsp":
                 vel_type = "Group Vel"
             elif name_list[2] == "TT":
                 vel_type = "Phase Vel"

             dsp_map = pickle.load(open(grid_file, "rb"))
             self.cartopy_canvas.plot_disp_map(k, dsp_map, interp=self.interpCB.currentText(),
                                               color=self.colorCB.currentText(), wave_type=wave_type, vel_type=vel_type,
                                               show_relief=self.reliefCB.isChecked(),
                                               map_type=self.mapTypeCB.currentText(), clip_scale=self.clipCB.isChecked(),
                                               low_limit=self.lowlimitDB.value(), up_limit=self.uplimitDB.value())

             period = str(dsp_map[0]["period"])
             vel_ref = str(dsp_map[0]["ref_velocity"])
             header = wave_type + " " + vel_type + " at period " + period + " s "
             self.cartopy_canvas.set_plot_title(k, header)

             ax = self.cartopy_canvas.get_axe(k)
             ax.spines["top"].set_visible(False)
             ax.spines["bottom"].set_visible(False)
             ax.spines["right"].set_visible(False)
             ax.spines["left"].set_visible(False)
             ax.tick_params(top=False)
             ax.tick_params(labeltop=False)


