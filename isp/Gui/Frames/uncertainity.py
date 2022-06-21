from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiUncertainity


class UncertainityInfo(pw.QDialog, UiUncertainity):

    def __init__(self, parent=None):
        super(UncertainityInfo, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

    def getUncertainity(self):

        uncertainity_dict = {}

        if self.confirmPickCB.isChecked():
            uncertainity_dict["pickingUncertainity"] = self.uncertainitySB.value()
        else:
            uncertainity_dict["pickingUncertainity"] = 0.0

        return uncertainity_dict["pickingUncertainity"]