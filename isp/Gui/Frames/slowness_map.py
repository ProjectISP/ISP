
from isp.Gui.Frames.uis_frames import UiSlowness_Map
from isp.Gui.Utils.pyqt_utils import add_save_load
from isp.Gui.Frames import BaseFrame


@add_save_load()
class SlownessMap(BaseFrame, UiSlowness_Map):

        """

        Search Catalog is build to facilitate the search of an Earthquake inside your project from a catalog

        :param params required to initialize the class:


        """
        def __init__(self):
            super(SlownessMap, self).__init__()
            self.setupUi(self)