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


class NoiseFrame(BaseFrame, UiNoise):

    def __init__(self):
        super(NoiseFrame, self).__init__()
        self.setupUi(self)

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
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_dataless_path)
        self.pagination = Pagination(self.pagination_widget, self.total_items, self.items_per_page)
        self.pagination.set_total_items(0)

        self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=self.items_per_page, constrained_layout=False)
        self.canvas.set_xlabel(0, "Time (s)")
        self.canvas.figure.tight_layout()

        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        self.actionSet_Parameters.triggered.connect(lambda: self.open_parameters_settings())
        self.outputBtn.clicked.connect(lambda: self.on_click_select_directory(self.output_bind))

        # actions
        #self.readfilesBtn.clicked.connect(self.read_files)
        self.preprocessBtn.clicked.connect(self.run_preprocess)

        self.actionOpen_Settings.triggered.connect(lambda: self.settings_dialog.show())
        # Parameters settings

        self.parameters = ParametersSettings()

        # help Documentation

        self.help = HelpDoc()

        # shortcuts

        self.shortcut_open = pw.QShortcut(pqg.QKeySequence('Ctrl+L'), self)
        self.shortcut_open.activated.connect(self.open_parameters_settings)


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

    def set_dataless_dir(self, dir_path):
        self.__dataless_dir = dir_path
        self.output.set_dataless_dir(dir_path)


    def onChange_dataless_path(self, value):
        self.__dataless_manager = DatalessManager(value)
        self.set_dataless_dir(value)

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    @AsycTime.run_async()
    def onChange_metadata_path(self, value):
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
        except:
            raise FileNotFoundError("The metadata is not valid")

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                       pw.QFileDialog.Option.DontUseNativeDialog)

        if dir_path:
            bind.value = dir_path

    def read_files(self, dir_path, out_path):
        self.ant = noise_organize(dir_path, out_path, self.inventory, self.params)
        self.ant.send_message.connect(self.receive_messages)
        self.ant.test()

    def run_preprocess(self):
        self.params = self.settings_dialog.getParameters()
        self.read_files(self.root_path_bind.value, self.output_bind.value)

    @pyc.pyqtSlot(str)
    def receive_messages(self, message):
        self.listWidget.addItem(message)
