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
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.datalessBtn.clicked.connect(lambda: self.on_click_select_metadata_file(self.dataless_path_bind))

        # Action Buttons
        self.processBtn.clicked.connect(lambda: self._ppsd_thread(self.process_ppsds))
        self.continueBtn.clicked.connect(lambda: self._ppsd_thread(self.ppsd_continue))
        self.saveBtn.clicked.connect(lambda: self.saveDB())
        self.loadBtn.clicked.connect(lambda: self.load_ppsd_db())
        #self.load_metadataBtn.clicked.connect(lambda: self.load_metadata())
        self.add_dataBtn.clicked.connect(lambda: self.add_data())

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
        md = MessageDialog(self)
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
            md.set_info_message("Loaded Metadata, please check your terminal for further details")
        except:
            md.set_error_message("Something went wrong. Please check your metada file is a correct one")

    def on_click_select_metadata_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]

    # def load_metadata(self):
    #     md = MessageDialog(self)
    #     if self.dataless_path_bind.value is not "":
    #         try:
    #             self.__metadata_manager = MetadataManager(self.dataless_path_bind.value)
    #             self.inventory = self.__metadata_manager.get_inventory()
    #             print(self.inventory)
    #             md.set_info_message("Loaded Metadata, please check your terminal for further details")
    #         except:
    #             md.set_error_message("Something went wrong. Please check your metada file is a correct one")

    def process_ppsds(self):

        file_path = self.root_path_bind.value

        try:
            self.ppsds = ppsdsISP(file_path, self.inventory, self.lenghtSB.value(), self.overlapSB.value(),
                             self.smoothingSB.value(), self.periodSB.value())

            self.ppsds.fileProcessed.connect(self.progressbar.setValue)
            ini_dict, size = self.ppsds.create_dict(net_list = self.netsTx.text(),sta_list = self.stationsTx.text(),chn_list = self.chnTx.text())
            pyc.QMetaObject.invokeMethod(self.progressbar, 'setMaximum', qt.AutoConnection, pyc.Q_ARG(int, size))
            pyc.QMetaObject.invokeMethod(self.progressbar, 'setValue', qt.AutoConnection, pyc.Q_ARG(int, 0))
            self.db = self.ppsds.get_all_values(ini_dict)
            self.saveBtn.setEnabled(True)
            md = MessageDialog(self)
            md.set_info_message("PPSDs DB ready, now you can save your progress!!!")

        except:
            md = MessageDialog(self)
            md.set_error_message("PPSDs DB  couldn't be created, please check your metadata and "
                                 "the data files directory")



    def ppsd_continue(self):
        self.ppsds.check= False
        self.ppsds.processedFiles = 0
        size = ppsdsISP.size_db(self.db)
        self.ppsds.blockSignals(False)
        pyc.QMetaObject.invokeMethod(self.progressbar, 'setMaximum', qt.AutoConnection, pyc.Q_ARG(int, size))
        pyc.QMetaObject.invokeMethod(self.progressbar, 'setValue', qt.AutoConnection, pyc.Q_ARG(int, 0))
        self.db = self.ppsds.get_all_values(self.db)

    def add_data(self):
        file_path = self.root_path_bind.value
        self.ppsds = ppsdsISP(file_path, self.inventory, self.lenghtSB.value(), self.overlapSB.value(),
                              self.smoothingSB.value(), self.periodSB.value())

        self.ppsds.fileProcessed.connect(self.progressbar.setValue)

        if self.db:
            ini_dict, size = self.ppsds.add_db_files(self.db, net_list=self.netsTx.text(), sta_list=self.stationsTx.text(),
                                                    chn_list=self.chnTx.text())

            pyc.QMetaObject.invokeMethod(self.progressbar, 'setMaximum', qt.AutoConnection, pyc.Q_ARG(int, size))
            pyc.QMetaObject.invokeMethod(self.progressbar, 'setValue', qt.AutoConnection, pyc.Q_ARG(int, 0))
            self.db = self.ppsds.get_all_values(ini_dict)
            md = MessageDialog(self)
            md.set_info_message("Data incorporated and processed")
        else:
            md = MessageDialog(self)
            md.set_info_message("No data to add and process")


    def saveDB(self):
        if self.db:
            params = self.db
            params['parameters']=[self.lenghtSB.value(),self.overlapSB.value(),
                                  self.smoothingSB.value(),self.periodSB.value()]
            path = pw.QFileDialog.getExistingDirectory(self,'Select Directory', self.root_path_bind.value)
            ppsdsISP.save_PPSDs(params, path, self.nameForm.text())
            md = MessageDialog(self)
            md.set_info_message("DB saved successfully")
        else:
            md = MessageDialog(self)
            md.set_info_message("No data to save in DB")


    def load_ppsd_db(self):
        selected = []
        file_path = self.root_path_bind.value
        selected = pw.QFileDialog.getOpenFileName(self, "Select DB file")
        if isinstance(selected[0], str):
            try:
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
                self.continueBtn.setEnabled(True)
                self.add_dataBtn.setEnabled(True)
                self.saveBtn.setEnabled(True)
                self.ppsds = ppsdsISP(file_path, self.inventory, self.lenghtSB.value(), self.overlapSB.value(),
                                      self.smoothingSB.value(), self.periodSB.value())
                self.ppsds.fileProcessed.connect(self.progressbar.setValue)
                md = MessageDialog(self)
                md.set_info_message("PPSDs DB loaded")

            except:
                md = MessageDialog(self)
                md.set_error_message("PPSDs DB cannot be loaded")
        else:
            pass




