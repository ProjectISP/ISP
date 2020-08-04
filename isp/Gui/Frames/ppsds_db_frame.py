from isp.Gui import pw
from isp.Gui.Frames import UiPPSDs_dialog
from isp.Gui.Utils.pyqt_utils import add_save_load

@add_save_load()
class PPSDsGeneratorDialog(pw.QDialog, UiPPSDs_dialog):
    def __init__(self, parent=None):
        super(PPSDsGeneratorDialog, self).__init__(parent)
        self.setupUi(self)

