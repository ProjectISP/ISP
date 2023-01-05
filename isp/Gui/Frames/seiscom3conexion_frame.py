from PyQt5.QtWidgets import QDialogButtonBox
from isp.Gui import pw
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiSeisComp3connexion
from isp.Gui.Utils.pyqt_utils import add_save_load
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
               'sdsout': '/none'}

        print(self.cfg)
        self.sc = SC(**self.cfg)

        md = MessageDialog(self)
        md.set_info_message("Loaded SeisComP3 DB and sftp connexion parameters!, proceed to download catalog")

    def getParametersNow(self):

        return self.sc