from isp.Structures.structures import StationsStats
from isp.Utils import MseedUtil, ObspyUtil


class DatalessManager:

    def __init__(self, root_path):
        self.__root_path = root_path
        self.__dataless_files = []
        self.__stations_stats = None

    @property
    def dataless_files(self):
        return self.__dataless_files

    @property
    def stations_stats(self):
        if self.__stations_stats is None:
            self.__stations_stats = self.__get_stations_stats()
        return self.__stations_stats

    def __get_stations_stats(self):
        self.__dataless_files = MseedUtil.get_dataless_files(self.__root_path)
        stations_stats = []
        for file in self.dataless_files:
            stations_stats.append(StationsStats.from_dataless(file))

        return stations_stats

    def get_station_stats_by_name(self, name: str):
        for st_stats in self.stations_stats:
            if st_stats.Name.lower() == name.lower():
                return st_stats

        return None

    def get_station_stats_by_mseed_file(self, file_path: str):
        mseed_stats = ObspyUtil.get_stats(file_path)
        return self.get_station_stats_by_name(mseed_stats.Station)
