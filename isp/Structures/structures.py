import traceback
from typing import NamedTuple

from obspy import UTCDateTime, read_inventory
from obspy.io.xseed import Parser

from isp import app_logger


def validate_dictionary(cls: NamedTuple, dic: dict):
    """
     Force the dictionary to have the same type of the parameter's declaration.

    :param cls: Expect a NamedTuple child class.
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
    Class that holds a structure of station metadata.

    Fields:
        * Name = (string) Station name.
        * Lon = (float) Longitude of station.
        * Lat = (float) Latitude of station.
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

    @classmethod
    def from_metadata(cls, file_path):

        inv = read_inventory(file_path)

        return inv



class PickerStructure(NamedTuple):
    """
    Class that holds a structure for the picker. This is used for re-plot the pickers keeping all
    necessary information in memory.

    Fields:
        * Time = (UTCDateTime) The time of the picker.
        * Station = (string) The station name.
        * XPosition = (float) The x position of the plot.
        * Amplitude = (float) The amplitude of the pick.
        * Color = (str) The color for the annotate box.
        * Label = (str) The abel for the annotate box.
        * FileName = (str) The file name (mseed file) that the picker was used.
    """
    Time: UTCDateTime
    Station: str
    XPosition: float
    Amplitude: float
    Color: str
    Label: str
    FileName: str

    def to_dict(self):
        return self._asdict()

    # noinspection PyTypeChecker
    @classmethod
    def from_dict(cls, dictionary):
        try:
            new_d = validate_dictionary(cls, dictionary)
            return cls(**new_d)

        except Exception as error:
            print(error)
            app_logger.error(traceback.format_exc())
            raise Exception

class StationCoordinates(NamedTuple):
    """
        Class that holds a structure for the picker. This is used for re-plot the pickers keeping all
        necessary information in memory.

        Fields:
            * Latitude = (float)

        """

    Latitude: float
    Longitude: float
    Elevation: float
    Local_depth: float

    def to_dict(self):
        return self._asdict()

    # noinspection PyTypeChecker
    @classmethod
    def from_dict(cls, dictionary):
        try:
            new_d = validate_dictionary(cls, dictionary)
            return cls(**new_d)

        except Exception as error:
            print(error)
            app_logger.error(traceback.format_exc())
            raise Exception


class Search(NamedTuple):
    """
    Class that holds a structure to perform search and paginate it. This structure can
    be used by any :class:`BaseModel`, since SearchBy and OrderBy are valid column names.
    Use the method from_dict to create an instance from a dictionary.

    Fields:
        SearchBy: A table's column's name to search. You can pass multiple values by using comma separation.
            e.g: "username, name", it will perform a search in this two columns.

        SearchValue: The value to search. You can pass multiple values by using comma separation.
            e.g: "John, Sara", it will perform a search for this values for the given columns.

        Page: The current page to return.

        PerPage: Number of items per page.

        OrderBy: A table's column's name to order.

        OrderDesc: True if the order must be descendant.

        MapColumnAndValue (default = False): If True it will consider a 1:1 mapping for SearchBy:SearchValue.
            e.g: Column -> "username, name", Values -> "admin, Sara".

            If True: This will search for username = like(%admin%) and name = like(%Sara%).

            If False: This will search for username = like(%admin%, %Sara%) and name = like(%admin%, %Sara%).

        Use_AND_Operator (default = False): Makes the search with AND instead of OR.

        TextualQuery: Use a textual query, i.e: "id<1111"
    """

    SearchBy: str
    SearchValue: str
    Page: int
    PerPage: int
    OrderBy: str
    OrderDesc: bool = False
    MapColumnAndValue: bool = False
    Use_AND_Operator: bool = False
    TextualQuery: str = None

    def to_dict(self):
        return self._asdict()

    # noinspection PyTypeChecker
    @classmethod
    def from_dict(cls, dictionary):
        new_d = validate_dictionary(cls, dictionary)
        return cls(**new_d)


class SearchResult(NamedTuple):
    """
    Class that holds a structure to return a search result.

    Fields:
        result: Expect a list of entities. However, it can be any object list that implements the method to_dict().

        total: The total number of entities found.
    """

    result: any
    total: int

    def to_dict(self) -> dict:
        """
        Map this object to a dictionary.

        :return: The dictionary representation of this object.
        """
        search_result_asdict = self._asdict()
        search_result_asdict["result"] = [entity.to_dict() for entity in self.result]
        return search_result_asdict

    # noinspection PyTypeChecker
    @classmethod
    def from_dict(cls, dictionary):
        new_d = validate_dictionary(cls, dictionary)
        return cls(**new_d)
