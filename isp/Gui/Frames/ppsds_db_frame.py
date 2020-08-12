import os
from isp.Gui import pw, pqg, qt, pyc
from isp.Gui.Frames import UiPPSDs_dialog, MessageDialog
from isp.Gui.Utils.pyqt_utils import add_save_load
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from isp.PPSDS_Utils.ppsds_utils import ppsdsISP
from concurrent.futures.thread import ThreadPoolExecutor

@add_save_load()
class PPSDsGeneratorDialog(pw.QDialog, UiPPSDs_dialog):
    def __init__(self, parent=None):
        super(PPSDsGeneratorDialog, self).__init__(parent)
        self.setupUi(self)
        self.inventory = {}
        self.ppsds = None
        self.progressbar = pw.QProgressDialog(self)
        self.progressbar.setWindowTitle('PPSD Running')
        self.progressbar.setLabelText(" Computing PPSDs ")
        self.progressbar.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.progressbar.close()
        # Binding
        self.root_path_bind = BindPyqtObject(self.rootPathForm)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)

        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.datalessBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))

        # Action Buttons
        self.processBtn.clicked.connect(lambda: self._ppsd_thread(self.process_ppsds))
        self.continueBtn.clicked.connect(lambda: self._ppsd_thread(self.ppsd_continue))
        self.saveBtn.clicked.connect(lambda: self.saveDB())
        self.loadBtn.clicked.connect(lambda: self.load_ppsd_db())

    def _stopBtnCallback(self):
        if self.ppsds is not None:
            self.ppsds.check = True

    def _ppsd_thread(self, callback):
        self.progressbar.reset()
        with ThreadPoolExecutor(1) as executor:
            f = executor.submit(callback)
            self.progressbar.exec()
            self._stopBtnCallback()
            f.cancel()
            self.ppsds.blockSignals(True)
            self.progressbar.close()
        self.continueBtn.setEnabled(True)


    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        if dir_path:
            bind.value = dir_path

    def onChange_metadata_path(self, value):
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
        except:
            pass

    def process_ppsds(self):
        file_path = self.root_path_bind.value
        self.ppsds = ppsdsISP(file_path, self.inventory, self.lenghtSB.value(), self.overlapSB.value(),
                         self.smoothingSB.value(), self.periodSB.value())

        self.ppsds.fileProcessed.connect(self.progressbar.setValue)
        ini_dict, size = self.ppsds.create_dict()
        pyc.QMetaObject.invokeMethod(self.progressbar, 'setMaximum', qt.AutoConnection, pyc.Q_ARG(int, size))
        pyc.QMetaObject.invokeMethod(self.progressbar, 'setValue', qt.AutoConnection, pyc.Q_ARG(int, 0))
        self.db = self.ppsds.get_all_values(ini_dict)
        print(self.db)


    def ppsd_continue(self):
        self.ppsds.check= False
        self.ppsds.processedFiles = 0

        size = ppsdsISP.size_db(self.db)
        self.ppsds.blockSignals(False)
        pyc.QMetaObject.invokeMethod(self.progressbar, 'setMaximum', qt.AutoConnection, pyc.Q_ARG(int, size))
        pyc.QMetaObject.invokeMethod(self.progressbar, 'setValue', qt.AutoConnection, pyc.Q_ARG(int, 0))
        self.db = self.ppsds.get_all_values(self.db)
        print(self.db)

    def saveDB(self):
        if self.db:
            params = self.db
            params['parameters']=[self.lenghtSB.value(),self.overlapSB.value(),
                                  self.smoothingSB.value(),self.periodSB.value()]
            path = pw.QFileDialog.getExistingDirectory(self,'Select Directory', self.root_path_bind.value)
            ppsdsISP.save_PPSDs(params, path, self.nameForm.text())


    def load_ppsd_db(self):
        file_path = self.root_path_bind.value
        selected = pw.QFileDialog.getOpenFileName(self, "Select DB file")
        head_tail = os.path.split(selected[0])
        params = ppsdsISP.load_PPSDs(head_tail[0], head_tail[1])
        self.db = {'nets' : params['nets']}
        self.lenghtSB.setValue(params['parameters'][0])
        self.overlapSB.setValue(params['parameters'][1])
        self.smoothingSB.setValue(params['parameters'][2])
        self.periodSB.setValue(params['parameters'][3])
        self.lenghtSB.setEnabled(False)
        self.overlapSB.setEnabled(False)
        self.smoothingSB.setEnabled(False)
        self.periodSB.setEnabled(False)
        self.processBtn.setEnabled(False)
        self.continueBtn.setEnabled(False)
        self.ppsds = ppsdsISP(file_path, self.inventory, self.lenghtSB.value(), self.overlapSB.value(),
                              self.smoothingSB.value(), self.periodSB.value())
        self.ppsds.fileProcessed.connect(self.progressbar.setValue)
        md = MessageDialog(self)
        md.set_info_message("PPSDs DB loaded")




