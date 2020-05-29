from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiMagnitudeFrame

class MagnitudeCalc(pw.QFrame, UiMagnitudeFrame):
    def __init__(self):
        super(MagnitudeCalc, self).__init__()
        self.setupUi(self)