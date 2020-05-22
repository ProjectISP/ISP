from isp.Structures.structures import StationsStats, StationCoordinates
from isp.Utils import MseedUtil, ObspyUtil
from obspy import read, read_inventory


class MetadataManager:

    def __init__(self, root_path):
        self.__root_path = root_path
        self.__dataless_files = []
        self.__stations_stats = None
        self.__dataless_file = MseedUtil.get_dataless_files(self.__root_path)

    def get_inventory(self):
        inv = read_inventory(self.__dataless_file[0])
        return inv

    def get_station_stats_by_mseed_file(self, file_path: str):
        mseed_stats = ObspyUtil.get_stats(file_path)
        return mseed_stats

    def extract_coordinates(self, inventory, file_path: str):

        stats = self.get_station_stats_by_mseed_file(file_path=file_path)
        selected_inv = inventory.select(network=stats.Network, station=stats.Station, channel=stats.Channel,
                                        starttime = stats.StartTime, endtime = stats.EndTime)
        cont = selected_inv.get_contents()
        coords = selected_inv.get_coordinates(cont['channels'][0])

        return StationCoordinates.from_dict(coords)

    def extrac_coordinates_from_trace(self, inventory, trace):
        stats = ObspyUtil.get_stats_from_trace(trace)
        selected_inv = inventory.select(network=stats['net'], station=stats['station'], channel=stats['channel'],
                                        starttime=stats['starttime'], endtime=stats['endtime'])
        cont = selected_inv.get_contents()
        coords = selected_inv.get_coordinates(cont['channels'][0])

        return StationCoordinates.from_dict(coords)



