import os
import sys
from unittest import TestCase

from qtpy import QtWidgets

from isp import ROOT_DIR
from isp.Gui.Frames import MatplotlibFrame
from isp.earthquakeAnalisysis import EarthquakeLocation


class TestEarthquakeLocation(TestCase):

    @classmethod
    def setUpClass(cls):
        pass
        # app = QtWidgets.QApplication(sys.argv)

    def setUp(self):
        file_path = os.path.join(ROOT_DIR, "260", "Velocity")
        self.earquake_loacation = EarthquakeLocation(file_path)


    def test_pagination(self):
        npp = 3
        print(self.earquake_loacation.paginate(npp, 1))
        print(self.earquake_loacation.paginate(npp, 2))
        print(self.earquake_loacation.paginate(npp, 3))

    def test_plot(self):
        app = QtWidgets.QApplication(sys.argv)
        fig = self.earquake_loacation.plot(6,1)
        if fig:
            self.mpl = MatplotlibFrame(fig)
            self.mpl.show()
        sys.exit(app.exec_())

