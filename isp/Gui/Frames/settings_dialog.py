from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiSettingsDialog

class SettingsDialog(pw.QDialog, UiSettingsDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

    def getParameters(self):
        param_dict = {}
        for i in range(self.formLayout.rowCount()):
            widget = self.formLayout.itemAt(i, pw.QFormLayout.FieldRole).widget().value()
            label = self.formLayout.itemAt(i, pw.QFormLayout.LabelRole).widget().text()
            param_dict[label] = widget 

        return param_dict
