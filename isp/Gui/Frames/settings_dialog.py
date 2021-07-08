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
        param_dict["STA"] = self.staDB.value()
        param_dict["LTA"] = self.ltaDB.value()
        param_dict["N Tapers"] = self.tapersDB.value()
        param_dict["TW"] = self.twDB.value()
        param_dict["Win"] = self.winDB.value()
        param_dict["Num Cycles"] = self.ncyclesDB.value()
        param_dict["Fmin"] = self.fminDB.value()
        param_dict["Fmax"] = self.fminDB.value()
        param_dict["win_entropy"] = self.win_entropyDB.value()
        param_dict["stack type"] = self.typestackCB.currentText()
        param_dict["ThresholdDetect"] = self.ThresholdSB.value()
        param_dict["Coincidences"] = self.ConcidencesSB.value()
        param_dict["Cluster"] = self.ClusterSB.value()
        return param_dict

