from obspy import Inventory
from obspy.core.event import Origin



class StationUtils:
    """
    Utility class for handling station data and metadata.
    """
    @staticmethod
    def get_station_location_dict(origin, inventory):
        """
        Creates a dictionary mapping station names to their latitude and longitude.

        Parameters:
        origin (obspy.core.event.origin.Origin): The Origin object containing pick information.
        inventory (obspy.core.inventory.inventory.Inventory): The Inventory object with station metadata.

        Returns:
        dict: A dictionary where keys are station names, and values are [latitude, longitude].
        """
        station_dict = {}

        # Extract station codes from the Origin's picks
        for pick in origin.picks:
            if pick.waveform_id and pick.waveform_id.station_code:
                station_code = pick.waveform_id.station_code

                # Search for the station in the Inventory
                for network in inventory:
                    for station in network:
                        if station.code == station_code:
                            # Add the station's name as key, and [latitude, longitude] as value
                            station_dict[station_code] = [station.longitude, station.latitude]
                            break  # Stop searching once the station is found

        return station_dict

def plot_real_map(networks,  **kwargs):
    import cartopy.crs as ccrs
    from matplotlib import pyplot as plt
    from owslib.wms import WebMapService
    import matplotlib
    import cartopy
    from isp import MAP_SERVICE_URL, MAP_LAYER
    matplotlib.use("Qt5Agg")
    ##Extract Area values##
    area = kwargs.pop('area', None)

    if area is not None:
        x = [area[0], area[1], area[2], area[3], area[4]]
        y = [area[5], area[6], area[7], area[8], area[9]]


    all_lon = []
    all_lat = []
    names = []
    for key in networks.keys():
        names += networks[key][0]
        all_lon += networks[key][1]
        all_lat += networks[key][2]

    MAP_SERVICE_URL = MAP_SERVICE_URL
    layer = MAP_LAYER
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj), figsize=(16, 12))

    xmin = min(all_lon) - 1
    xmax = max(all_lon) + 1
    ymin = min(all_lat) - 1
    ymax = max(all_lat) + 1
    extent = [xmin, xmax, ymin, ymax]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    try:
        wms = WebMapService(MAP_SERVICE_URL)
        ax.add_wms(wms, layer)
    except:
        coastline_10m = cartopy.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                                                            edgecolor='k', alpha=0.6, linewidth=0.5,
                                                            facecolor=cartopy.feature.COLORS['land'])
        ax.add_feature(coastline_10m)