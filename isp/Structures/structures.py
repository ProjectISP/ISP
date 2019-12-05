import traceback
from typing import NamedTuple

from obspy import UTCDateTime
from obspy.io.xseed import Parser

from isp import app_logger


def validate_dictionary(cls: NamedTuple, dic: dict):
    """
     Force the dictionary to have the same type of the parameter's declaration.
    :param cls: Expect a NamedTuple derived class.
    :param dic: The dictionary to be validate.
    :return: A new dictionary that try to keeps the same data type from this class parameters.
    """
    valid_dic = {}
    fields_list_lower = [k.lower() for k in cls._fields]
    for k in dic.keys():
        index = fields_list_lower.index(k.lower())  # Compare all in lower case. Avoid Caps sensitive.
        safe_key = cls._fields[index]

        if cls._field_types.get(safe_key) == int:
            valid_dic[safe_key] = int(dic.get(k))
        elif cls._field_types.get(safe_key) == float:
            valid_dic[safe_key] = float(dic.get(k))
        elif cls._field_types.get(safe_key) == bool:
            if hasattr(dic.get(k), "capitalize"):
                valid_dic[safe_key] = True if dic.get(k).capitalize() == "True" else False
            else:
                valid_dic[safe_key] = dic.get(k)
        elif cls._field_types.get(safe_key) == str:
            if dic.get(k) == "null":
                valid_dic[safe_key] = None
            else:
                valid_dic[safe_key] = str(dic.get(k))
        else:
            valid_dic[safe_key] = dic.get(k)
    return valid_dic


class TracerStats(NamedTuple):
    """
    Class that holds a structure of mseed metadata.
    Fields:
        * Network = (string) network name.
        * Station = (string) station name.
        * Channel = (string) channel name.
        * StartTime = (UTCDateTime) start datetime.
        * EndTime = (UTCDateTime) stop datetime.
        * Location = (string) The location.
        * Sampling_rate = (float) Sample rate in hertz.
        * Delta = (float) Delta time.
        * Npts = (int) Number of points.
        * Calib = (float) Calibration of instrument.
        * Mseed = (dict) Information of mseed file.
        * Format = (string) File format
    """
    Network: str = None
    Station: str = None
    Channel: str = None
    StartTime: UTCDateTime = None
    EndTime: UTCDateTime = None
    Location: str = None
    Sampling_rate: float = None
    Delta: float = None
    Npts: int = None
    Calib: float = None
    Mseed: dict = None
    Format: str = None

    def to_dict(self):
        return self._asdict()

    # noinspection PyTypeChecker
    @classmethod
    def from_dict(cls, dictionary):
        try:
            from isp.Structures.obspy_stats_keys import ObspyStatsKeys
            file_format = dictionary.pop(ObspyStatsKeys.FORMAT)
            new_d = validate_dictionary(cls, dictionary)
            new_d["Format"] = file_format
            dictionary[ObspyStatsKeys.FORMAT] = file_format
            return cls(**new_d)

        except Exception as error:
            print(error)
            app_logger.error(traceback.format_exc())
            raise Exception


class StationsStats(NamedTuple):
    """
    Class that holds a structure of mseed metadata.
    Fields:
        * Name = (string) Station name.
        * Lon = (string) Longitude of station.
        * Lat = (string) Latitude of station.
        * Depth = (float) Depth of station.
    """
    Name: str = None
    Lon: float = None
    Lat: float = None
    Depth: float = None

    def to_dict(self):
        return self._asdict()

    # noinspection PyTypeChecker
    @classmethod
    def from_dict(cls, dictionary):
        try:
            from isp.Structures.obspy_stats_keys import ObspyStatsKeys
            new_d = validate_dictionary(cls, dictionary)
            return cls(**new_d)

        except Exception as error:
            print(error)
            app_logger.error(traceback.format_exc())
            raise Exception

    @classmethod
    def from_dataless(cls, file_path):
        parser = Parser()
        parser.read(file_path)
        # inventory = parser.get_inventory()
        station_blk = parser.stations[0][0]
        station_dict = {"Name": station_blk.station_call_letters, "Lon": station_blk.longitude,
                        "Lat": station_blk.latitude, "Depth": station_blk.elevation}
        return cls(**station_dict)

