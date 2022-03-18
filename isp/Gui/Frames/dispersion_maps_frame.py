import os
from isp.Exceptions import parse_excepts
from isp.Gui import pw
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiDispersionMaps
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.Utils import AsycTime
from isp.ant.tomo_tools import tomotools


@add_save_load()
class EGFDispersion(pw.QWidget, UiDispersionMaps):

    def __init__(self):
        super(EGFDispersion, self).__init__()
        self.setupUi(self)
        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.stations_path_bind = BindPyqtObject(self.StationsPathForm, self.onChange_stations_path)
        self.read_pklBtn.clicked.connect(lambda: self.on_click_select_data_file(self.root_path_bind))
        self.stationsBtn.clicked.connect(lambda: self.on_click_select_data_file(self.stations_path_bind))

    def on_click_select_data_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]

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
    def onChange_root_path(self, value):

        if self.wave_typeCB.currentText() == "Vertical":
            wave_type = ["ZZ"]
        elif self.wave_typeCB.currentText() == "Transversal":
            wave_type = ["TT"]
        elif self.wave_typeCB.currentText() == "Both":
            wave_type = ["TT", "ZZ"]

        if self.disp_typeCB.currentText() == "Group Velocity":
            dispersion_type = ["dsp"]
        elif self.disp_typeCB.currentText() == "Phase Velocity":
            dispersion_type = ["phv"]
        elif self.disp_typeCB.currentText() == "Both":
            dispersion_type = ["dsp","phv"]

        try:

            self.data_info = tomotools.read_dispersion(value, wave_type, dispersion_type)

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