from isp.Gui.Frames import BaseFrame, UiEarthquakeAnalysisFrame
from isp.earthquakeAnalisysis import EarthquakeLocation


class EarthquakeAnalysisFrame(BaseFrame,UiEarthquakeAnalysisFrame):

    def __init__(self, ):
        super(EarthquakeAnalysisFrame, self).__init__()
        self.setupUi(self)

        self.test1Button.clicked.connect(self.onClick_test1)

    def onClick_test1(self):
        quake_location = EarthquakeLocation()
        quake_location.locate_earthquake(1)



