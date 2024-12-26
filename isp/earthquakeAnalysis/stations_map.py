import cartopy

from isp import MAP_SERVICE_URL, MAP_LAYER
from isp.Gui.Frames import MatplotlibFrame


class StationsMap:

    def __init__(self, stations_dict):
        """
        Plot stations map fro dictionary (key = stations name, coordinates)

        :param
        """
        self.__stations_dict = stations_dict

    def _extrac_layers(self, wms):
        for layer in wms.contents:
            print(f" - {layer}: {wms[layer].title}")


    def plot_stations_map(self, **kwargs):
        from matplotlib.transforms import offset_copy
        import cartopy.crs as ccrs
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        from owslib.wms import WebMapService
        from matplotlib.patheffects import Stroke
        import cartopy.feature as cfeature
        import shapely.geometry as sgeom
        from matplotlib import pyplot as plt
        from isp import ROOT_DIR
        import os
        os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join(ROOT_DIR, "maps")
        # MAP_SERVICE_URL = 'https://gis.ngdc.noaa.gov/arcgis/services/gebco08_hillshade/MapServer/WMSServer'
        # MAP_SERVICE_URL = 'https://www.gebco.net/data_and_products/gebco_web_services/2020/mapserv?'
        # MAP_SERVICE_URL = 'https://gis.ngdc.noaa.gov/arcgis/services/etopo1/MapServer/WMSServer'
        geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
        #layer = 'GEBCO_08 Hillshade'
        # layer ='GEBCO_2020_Grid'
        #layer = 'shaded_relief'

        epi_lat = kwargs.pop('latitude')
        epi_lon = kwargs.pop('longitude')

        name_stations = []
        lat = []
        lon = []
        for name, coords in self.__stations_dict.items():
            name_stations.append(name)
            lat.append(float(coords[0]))
            lon.append(float(coords[1]))

        #
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj), figsize=(16, 12))
        self.mpf = MatplotlibFrame(fig, window_title="Stations Map")

        xmin = min(lon)-4
        xmax = max(lon)+4
        ymin = min(lat)-4
        ymax = max(lat)+4
        extent = [xmin, xmax, ymin, ymax]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        try:

            wms = WebMapService(MAP_SERVICE_URL)
            ax.add_wms(wms, MAP_LAYER)


        except Exception as e:
            print("Coudnt load web service")
            # Capture and print the exception
            print(f"An error occurred: {e}")
            ax.background_img(name='ne_shaded', resolution="high")


        geodetic_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        text_transform = offset_copy(geodetic_transform, units='dots', x=-25)
        ax.scatter(lon, lat, s=12, marker="^", color='red', alpha=0.7, transform=ccrs.PlateCarree())
        ax.scatter(epi_lon, epi_lat, s=18, marker="*", color='white', edgecolor='black')

        N=len(name_stations)
        for n in range(N):
            lon1=lon[n]
            lat1 = lat[n]
            name = name_stations[n]

            ax.text(lon1, lat1, name, verticalalignment='center', horizontalalignment='right', transform=text_transform,
                bbox=dict(facecolor='sandybrown', alpha=0.5, boxstyle='round'))

        # Create an inset GeoAxes showing the Global location
        sub_ax = self.mpf.canvas.figure.add_axes([0.70, 0.73, 0.28, 0.28], projection=ccrs.PlateCarree())
        sub_ax.set_extent([-179.9, 180, -89.9, 90], geodetic)

        # Make a nice border around the inset axes.
        #effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)
        #sub_ax.outline_patch.set_path_effects([effect])

        # Add the land, coastlines and the extent .
        sub_ax.add_feature(cfeature.LAND)
        sub_ax.coastlines()
        extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
        sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='none',
                              edgecolor='blue', linewidth=1.0)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.2, color='gray', alpha=0.2, linestyle='-')

        gl.top_labels = False
        gl.left_labels = False
        gl.xlines = False
        gl.ylines = False

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        #plt.savefig("/Users/robertocabieces/Documents/ISPshare/isp/mti/output/stations.png", bbox_inches='tight')
        self.mpf.show()

















