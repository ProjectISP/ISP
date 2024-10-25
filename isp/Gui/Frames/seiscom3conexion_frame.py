import os
from sys import platform
from PyQt5.QtWidgets import QDialogButtonBox
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Gui import pw
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiSeisComp3connexion
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.retrieve_events.seiscomp3connection import seiscompConnector as SC

@add_save_load()
class SeisCopm3connexion(pw.QDialog, UiSeisComp3connexion):
    def __init__(self, parent=None):
        super(SeisCopm3connexion, self).__init__(parent)
        self.setupUi(self)
        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

        self.cfg = None
        self.connectBtn.button(QDialogButtonBox.Ok).clicked.connect(self.get_connect_parameters)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)
        self.metadataBtn.clicked.connect(lambda: self.on_click_select_file(self.dataless_path_bind))
        self.loadBtn.clicked.connect(self.load_metadata_path_ext)
        self.dataoutputBtn.clicked.connect(self.set_output_path)

    def on_click_select_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]
            self.load_metadata_path(bind.value)

    def set_output_path(self):
        try:

            root_path = os.path.dirname(os.path.abspath(__file__))
            if "darwin" == platform:
                dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
            else:
                dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path,
                                                               pw.QFileDialog.DontUseNativeDialog)
            self.dataoutputLE.setText(dir_path)

        except:
            md = MessageDialog(self)
            md.set_info_message("Couldn't download event")

    def load_metadata_path(self, value):
        md = MessageDialog(self)
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
            md.set_info_message("Loaded Metadata, please check your terminal for further details")
        except:
            md.set_error_message("Something went wrong. Please check your metadata file is a correct one")

    def load_metadata_path_ext(self):
        md = MessageDialog(self)
        try:
            self.__metadata_manager = MetadataManager(self.datalessPathForm.text())
            self.inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
            md.set_info_message("Loaded Metadata, please check your terminal for further details")
        except:
            md.set_error_message("Something went wrong. Please check your metadata file is a correct one")


    def get_connect_parameters(self):
        """
             Obtiene los .mseed usando solo como filtro
             la red a la que pertenece la estacion
        """

        # Connexions DB and SFTP

        self.cfg = {'hostname': self.hostname.text(),
               'dbname': self.dbname.text(),
               'user': self.user.text(),
               'password': self.password.text(),
               'sdshost': self.sds_host.text(),
               'sdsuser': self.sds_user.text(),
               'sdspass': self.sds_password.text(),
               'sdsdir': self.sds_path.text(),
               'sdsport': self.sds_port.text(),
               'sdsout': self.dataoutputLE.text()+"/"}

        print(self.cfg)
        self.sc = SC(**self.cfg)

        md = MessageDialog(self)
        md.set_info_message("Loaded SeisComP3 DB and sftp connexion parameters!, proceed to download catalog")

    def getSeisComPdatabase(self):

        return self.sc

    def getMetadata(self):

        return self.inventory

