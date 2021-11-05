from isp.Gui import pqg
from isp.Gui.Frames import BaseFrame
from isp.Gui.Frames.uis_frames import UiNoise
from isp.Gui.Frames.efg_frame import EGFFrame

class NoiseFrame(BaseFrame, UiNoise):

    def __init__(self):
        super(NoiseFrame, self).__init__()
        self.setupUi(self)

        self.setWindowTitle('Seismic Ambient Noise')
        self.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))

        egf_frame = EGFFrame()
        self.tabWidget.addTab(egf_frame, 'EGFs')

