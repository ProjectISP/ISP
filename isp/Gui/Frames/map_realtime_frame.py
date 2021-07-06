#from isp.Gui import pw, pyc
from isp.Gui.Frames import BaseFrame, CartopyCanvas
from isp.Gui.Frames.uis_frames import UiMapRealTime
from isp.retrieve_events import retrieve


class MapRealTime(BaseFrame, UiMapRealTime):

    def __init__(self, metadata):
        super(MapRealTime, self).__init__()
        self.setupUi(self)
        self.metadata = metadata
        # Map
        self.cartopy_canvas = CartopyCanvas(self.map)
        self.cartopy_canvas.global_map(0)
        self.cartopy_canvas.figure.subplots_adjust(left=0.00, bottom=0.055, right=0.97, top=0.920, wspace=0.0,
                                                   hspace=0.0)

        self.retrivetool = retrieve()
        coordinates = self.retrivetool.get_inventory_coordinates(self.metadata)
        self.cartopy_canvas.global_map(0, plot_earthquakes=False, show_colorbar=False, show_stations=True,
             show_station_names=False, clear_plot=False, coordinates=coordinates)

    def plot_set_stations(self, stations_list):
        extent = [min(stations_list[3]), max(stations_list[3]), min(stations_list[2]), max(stations_list[2])]
        #print(extent)
        self.cartopy_canvas.global_map(0, plot_earthquakes=False, show_colorbar=False, show_stations=True,
             show_station_names=False, clear_plot=False,real_time=True, coordinates=stations_list, extent = extent)

