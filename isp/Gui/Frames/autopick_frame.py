import os.path

from PyQt5.QtCore import Qt
from isp import PICKING_DIR
from isp.Gui import pyc, pw
from isp.Gui.Frames import BaseFrame, MessageDialog
from isp.Gui.Frames.uis_frames import UiAutopick
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.Utils import AsycTime
from isp.Utils.os_utils import OSutils
from surfquakecore.phasenet.phasenet_handler import PhasenetISP, PhasenetUtils

@add_save_load()
class Autopick(BaseFrame, UiAutopick):

    signal = pyc.pyqtSignal()
    def __init__(self, project):
        super(Autopick, self).__init__()
        self.setupUi(self)

        """
        Locate Event Frame

        :param params required to initialize the class:

        """


        self.sp = project

        self.picking_bind = BindPyqtObject(self.picking_LE, self.onChange_root_path)
        self.output_path_pickBtn.clicked.connect(lambda: self.on_click_select_directory(self.picking_bind))

        self.phasenetBtn.clicked.connect(self.run_phasenet)
        self.setDefaultPickPath.clicked.connect(self.setDefaultPick)

        # Dialog
        self.progress_dialog = pw.QProgressDialog(self)
        self.progress_dialog.setRange(0, 0)
        self.progress_dialog.setWindowTitle('Processing.....')
        self.progress_dialog.setLabelText('Please Wait')
        self.progress_dialog.setWindowIcon(self.windowIcon())
        self.progress_dialog.setWindowTitle(self.windowTitle())
        self.progress_dialog.close()


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

    ## Picking ##

    def setDefaultPick(self):
        self.picking_LE.setText(PICKING_DIR)


    def run_phasenet(self):

        if self.sp is None:
            md = MessageDialog(self)
            md.set_error_message("Metadata run Picking, Please load a project first")
        else:
            self.send_phasenet()
            self.progress_dialog.exec()
            md = MessageDialog(self)
            md.set_info_message("Picking Done")
            self.send_signal()

    @AsycTime.run_async()
    def send_phasenet(self):
        print("Starting Picking")

        phISP = PhasenetISP(self.sp, amplitude=True, min_p_prob=self.p_wave_picking_thresholdDB.value(),
                            min_s_prob=self.s_wave_picking_thresholdDB.value())

        picks = phISP.phasenet()
        picks_ = PhasenetUtils.split_picks(picks)

        PhasenetUtils.write_nlloc_format(picks_, self.picking_bind.value)
        PhasenetUtils.convert2real(picks_, self.picking_bind.value)
        PhasenetUtils.save_original_picks(picks_, self.picking_bind.value)


        file_picks = os.path.join(self.picking_bind.value, "nll_picks.txt")
        OSutils.copy_and_rename_file(file_picks, PICKING_DIR, "output.txt")
        """ PHASENET OUTPUT TO REAL INPUT"""
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)

    def send_signal(self):
        self.signal.emit()

    ## End Picking ##