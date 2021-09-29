from concurrent.futures import ThreadPoolExecutor

from isp.DataProcessing import DatalessManager
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import parse_excepts
from isp.Gui import pqg, pw, pyc
from isp.Gui.Frames import BaseFrame, Pagination, MatplotlibCanvas, MessageDialog
from isp.Gui.Frames.help_frame import HelpDoc
from isp.Gui.Frames.uis_frames import UiNoise
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from isp.Utils import AsycTime
from isp.Gui.Frames.setting_dialog_noise import SettingsDialogNoise
from isp.ant.ambientnoise import noise_organize
from isp.ant.process_ant import process_ant
from isp.ant.cross_stack import cross_stack

class NoiseFrame(BaseFrame, UiNoise):

    def __init__(self):
        super(NoiseFrame, self).__init__()
        self.setupUi(self)
        self.progressbar = pw.QProgressDialog(self)
        self.progressbar.setWindowTitle('Ambient Noise Tomography')
        self.progressbar.setLabelText(" Computing Auto-Picking ")
        self.progressbar.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.progressbar.close()
        self.setWindowTitle('Seismic Ambient Noise')
        self.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.settings_dialog = SettingsDialogNoise(self)
        self.inventory = {}
        self.files = []
        self.total_items = 0
        self.items_per_page = 1
        self.__dataless_manager = None
        self.__metadata_manager = None
        self.st = None
        self.output = None
        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.output_bind = BindPyqtObject(self.outPathForm, self.onChange_root_path)
        self.pagination = Pagination(self.pagination_widget, self.total_items, self.items_per_page)
        self.pagination.set_total_items(0)

        self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=self.items_per_page, constrained_layout=False)
        self.canvas.set_xlabel(0, "Time (s)")
        self.canvas.figure.tight_layout()

        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.metadata_path_bind))
        self.actionSet_Parameters.triggered.connect(lambda: self.open_parameters_settings())
        self.outputBtn.clicked.connect(lambda: self.on_click_select_directory(self.output_bind))

        # actions
        self.preprocessBtn.clicked.connect(self.run_preprocess)

        self.actionOpen_Settings.triggered.connect(lambda: self.settings_dialog.show())
        # Parameters settings

        self.parameters = ParametersSettings()

        # help Documentation

        self.help = HelpDoc()

        # shortcuts

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+L'), self)
        self.shortcut_open.activated.connect(self.open_parameters_settings)

    @pyc.Slot()
    def _increase_progress(self):
        self.progressbar.setValue(self.progressbar.value() + 1)

    def open_parameters_settings(self):
        self.parameters.show()

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def onChange_root_path(self, value):

        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        #self.read_files(value)
        pass


    @AsycTime.run_async()
    def onChange_metadata_path(self, value):
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
        except:
            pass

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                       pw.QFileDialog.Option.DontUseNativeDialog)

        if dir_path:
            bind.value = dir_path

    def read_files(self, dir_path, out_path):
        md = MessageDialog(self)
        md.hide()
        try:
            self.progressbar.reset()
            self.progressbar.setLabelText(" Reading Files ")
            self.progressbar.setRange(0,0)
            with ThreadPoolExecutor(1) as executor:
                self.ant = noise_organize(dir_path, self.inventory)
                self.ant.send_message.connect(self.receive_messages)
                def read_files_callback():
                    data_map, size, channels = self.ant.create_dict()

                    pyc.QMetaObject.invokeMethod(self.progressbar, 'accept')
                    return data_map, size, channels

                f = executor.submit(read_files_callback)
                self.progressbar.exec()
                self.data_map,self.size,self.channels = f.result()
                f.cancel()

            #self.ant.test()
            md.set_info_message("Readed data files Successfully")
        except:
            md.set_error_message("Something went wrong. Please check your data files are correct mseed files")

        md.show()



    # def process(self):
    #     md = MessageDialog(self)
    #     md.hide()
    #     try:
    #         self.progressbar.reset()
    #         self.progressbar.setLabelText(" Processing ")
    #         self.progressbar.setRange(0,0)
    #         with ThreadPoolExecutor(1) as executor:
    #
    #             self.ant.send_message.connect(self.receive_messages)
    #
    #             def read_files_callback():
    #                 list_raw = self.ant.get_all_values(self.results[0])
    #                 dict_matrix_list = self.ant.create_all_dict_matrix(list_raw, self.result[2])
    #
    #                 pyc.QMetaObject.invokeMethod(self.progressbar, 'accept')
    #                 return dict_matrix_list
    #
    #             f = executor.submit(read_files_callback)
    #             self.progressbar.exec()
    #             self.dict_matrix_list = f.dict_matrix_list()
    #             f.cancel()
    #
    #             #self.ant.test()
    #             md.set_info_message("Proceess Done, Please check the outout folder")
    #     except:
    #         md.set_error_message("Something went wrong. Please check your data files are correct mseed files")
    #
    #     md.show()

    def run_preprocess(self):
        self.params = self.settings_dialog.getParameters()
        self.read_files(self.root_path_bind.value, self.output_bind.value)
        self.process()


    def process(self):
        #
        self.process_ant = process_ant(self.output_bind.value, self.params, self.inventory)
        #self.process_ant.send_message.connect(self.receive_messages)
        list_raw = self.process_ant.get_all_values(self.data_map)
        dict_matrix_list = self.process_ant.create_all_dict_matrix(list_raw, self.channels)
        #post = self.cross_stack(self.output_bind.value, self.output_bind.value)
        #post.run_cross_stack()

    @pyc.pyqtSlot(str)
    def receive_messages(self, message):
        self.listWidget.addItem(message)