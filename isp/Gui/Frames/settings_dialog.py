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
        param_dict["Num Cycles"] = self.ncyclesDB.value()
        param_dict["Fmin"] = self.fminDB.value()
        param_dict["Fmax"] = self.fminDB.value()
        param_dict["win_entropy"] = self.win_entropyDB.value()
        param_dict["stack type"] = self.typestackCB.currentText()
        param_dict["kurt_win"] = self.kurtosis_timewindowDB.value()
        param_dict["lineColor"] = self.colorCB.currentText()
        param_dict["amplitudeaxis"] = self.amplitudeaxisCB.isChecked()
        param_dict["prs_phases"] = self.phasesLE.text().split(",")
        return param_dict

