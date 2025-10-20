from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiSettingsDialogNoise
from isp.Gui.Utils.pyqt_utils import add_save_load


@add_save_load()
class SettingsDialogNoise(pw.QDialog, UiSettingsDialogNoise):
    def __init__(self, parent=None):
        super(SettingsDialogNoise, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

    def getParameters(self):

        param_dict = {}
        param_dict["f1"] = self.f1.value()
        param_dict["f2"] = self.f2.value()
        param_dict["f3"] = self.f3.value()
        param_dict["f4"] = self.f4.value()
        param_dict["waterlevel"] = self.waterlevelSB.value()
        param_dict["units"] = self.unitsCB.currentText()
        param_dict["factor"] = self.factor.value()
        param_dict["method"] = self.timenorm.currentText()
        param_dict["timewindow"] = self.timewindow.value()
        param_dict["freqbandwidth"] = self.freqbandwidth.value()
        param_dict["remove_responseCB"] = self.remove_responseCB.isChecked()
        param_dict["decimationCB"] = self.decimationCB.isChecked()
        param_dict["time_normalizationCB"] = self.time_normalizationCB.isChecked()
        param_dict["whitheningCB"] = self.whitheningCB.isChecked()
        param_dict["channels"] = self.componentsLE.text().split(',')
        param_dict["stations"] = [s.strip() for s in self.stationsLE.text().split(',') if s.strip()]
        param_dict["stack"] = self.stackCB.currentText()
        param_dict["power"] = self.powerSB.value()
        param_dict["processing_window"] = float(self.timeWindowCB.currentText())
        param_dict["nets_list"] = self.lineEditNets.text()
        param_dict["stations_list"] = self.lineEditStations.text()
        param_dict["channels_list"] = self.lineEditChannels.text()
        param_dict["autocorr"] = self.autocorrCB.isChecked()
        param_dict["max_distance"] = self.max_distanceBtn.value()
        param_dict["dailyStacks"] = self.dailyStacksCB.isChecked()
        param_dict["overlap"] = self.overlapSB.value()
        param_dict["prefilter"] = self.preFilterCB.isChecked()
        param_dict["filter_freqmin"] = self.freqminFilterDB.value()
        param_dict["filter_freqmax"] = self.freqmaxFilterDB.value()
        param_dict["filter_corners"] = self.cornersFilterDB.value()


        return param_dict