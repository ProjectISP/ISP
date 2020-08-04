from isp.Gui.Frames import BaseFrame, UiPPSDs
from isp.Gui.Frames.ppsds_db_frame import PPSDsGeneratorDialog


class PPSDFrame(BaseFrame, UiPPSDs):

    def __init__(self):
        super(PPSDFrame, self).__init__()
        self.setupUi(self)
        self.ppsds_dialog = None
        self.ppsds_dialog = PPSDsGeneratorDialog(self)
        self.actionGenerate_synthetics.triggered.connect(self.run_ppsds)

    def run_ppsds(self):
        self.ppsds_dialog.show()