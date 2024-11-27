import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import NamedTuple

from obspy import UTCDateTime, read_inventory
from obspy.io.xseed import Parser
from sqlalchemy import func, Column

from isp import app_logger
from isp.Structures import BaseDataClass


def validate_dictionary(cls: NamedTuple, dic: dict):
    """
    Force the dictionary to have the same type of the parameter's declaration.

    :param cls: Expect a NamedTuple child class.
    :param dic: The dictionary to be validated.
    :return: A new dictionary that tries to keep the same data type from this class parameters.
    """
    valid_dic = {}
    fields_list_lower = [k.lower() for k in cls._fields]

    for k in dic.keys():
        if k.lower() in fields_list_lower:
            try:
                # Find the correct field name in the NamedTuple (case insensitive)
                index = fields_list_lower.index(k.lower())
                safe_key = cls._fields[index]

                # Access the type annotation for the field
                field_type = cls.__annotations__.get(safe_key)

                # Cast the dictionary value to the correct type based on the field type
                if field_type == int:
                    valid_dic[safe_key] = int(dic.get(k))
                elif field_type == float:
                    valid_dic[safe_key] = float(dic.get(k))
                elif field_type == bool:
                    value = dic.get(k)
                    if isinstance(value, str):
                        valid_dic[safe_key] = value.strip().lower() == "true"
                    else:
                        valid_dic[safe_key] = bool(value)
                elif field_type == str:
                    # Handle 'null' as None
                    value = dic.get(k)
                    valid_dic[safe_key] = None if value == "null" else str(value)
                else:
                    valid_dic[safe_key] = dic.get(k)

            except (KeyError, ValueError, TypeError) as e:
                # Log or handle the error here
                print(f"Error processing key '{k}': {e}")
                pass

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
        * Dataquality = (string)
        * numsamples = 236
        * samplecnt = (float)
        * sampletype: (string)
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
    # Calib: float = None
    # Mseed: dict = None
     #Format: str = None
    # Dataquality: str = None
    # numsamples: float = None
    # samplecnt:float = None
    # sampletype: str = None

    def to_dict(self):
        return self._asdict()

    # noinspection PyTypeChecker
    @classmethod
    def from_dict(cls, dictionary):
        try:
            from isp.Structures.obspy_stats_keys import ObspyStatsKeys
            if "processing" in dictionary:
                del dictionary['processing']
            file_format = dictionary.pop(ObspyStatsKeys.FORMAT, "mseed") #ISP 1.0
            new_d = validate_dictionary(cls, dictionary) #ISP 1.0
            #new_d["Format"] = file_format
            #dictionary[ObspyStatsKeys.FORMAT] = file_format
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
        * Uncertainty = (float) The uncertainty associated to the pick
        * Amplitude = (float) The amplitude of the pick.
        * Color = (str) The color for the annotate box.
        * Label = (str) The abel for the annotate box.
        * FileName = (str) The file name (mseed file) that the picker was used.
    """
    Time: UTCDateTime
    Station: str
    XPosition: float
    Uncertainty: float
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


class QueryOperators(Enum):
    INCLUDE = "CONTAINS", lambda col, val: col.like(f"%{val}%")
    EQUAL = "EQUAL", lambda col, val: col.__eq__(f"{val}")
    BEGINS_WITH = "BEGINSWITH", lambda col, val: col.like(f"{val}%")
    EQUAL_ALIAS = "==", lambda col, val: col.__eq__(f"{val}")
    INCLUDE_ALIAS = "[]", lambda col, val: col.like(f"%{val}%")
    BEGINS_WITH_ALIAS = "]", lambda col, val: col.like(f"{val}%")
    BIGGER = ">", lambda col, val: col.__gt__(f"{val}")
    BIGGER_EQUAL = ">=", lambda col, val: col.__ge__(f"{val}")
    SMALLER = "<", lambda col, val: col.__lt__(f"{val}")
    SMALLER_EQUAL = "<=", lambda col, val: col.__le__(f"{val}")
    NOT_EQUAL = "!=", lambda col, val: col.__ne__(f"{val}")
    LOWER_INCLUDES = "lower_CONTAINS", lambda col, val: col.ilike(f"%{val}%")
    LOWER_INCLUDES_ALIAS = "lower[]", lambda col, val: col.ilike(f"%{val}%")
    LOWER_BEGINS_WITH = "lower_BEGINSWITH", lambda col, val: col.ilike(f"{val}%")
    LOWER_EQUAL = "lower_EQUAL", lambda col, val: func.lower(col).__eq__(f"{val}")

    def __new__(cls, value, apply_filter=None):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.apply_filter = apply_filter
        return obj


@dataclass
class ColumnOperator:
    Column: Column
    QueryOp: QueryOperators


@dataclass
class SearchDefault(BaseDataClass):
    """
    Class that holds a structure to perform search and paginate it. This structure can
    be used by any :class:`BaseModel`, since SearchBy and OrderBy are valid column names.
    Use the method from_dict to create an instance from a dictionary.
    Fields:
        SearchBy: A table's column's name to search. You can pass multiple values by using comma separation,
        e.g:
            "username, name".
        This will perform a search in this to columns. SearchBy can also override
        the operator, e.g:
            "username: ==, name: !="
        Valid operator: "==, <,>,<=,>=,!=,[],]"
        SearchValue: The value to search. You can pass multiple values by using comma separation or a List, e.g:
            "John, Sara" or [ "John, Sara"]. This are safer than just comma separation.
        This will perform a search for this values for the given columns.
        Page: The current page to return.
        PerPage: Number of items per page.
        OrderBy: A table's column's name to order.
        OrderDesc: True if the order must be descendant.
        MapColumnAndValue (default = True): If True it will consider a 1:1 mapping for SearchBy:SearchValue.
            e.g: Column -> "username, name", Values -> "admin, Sara".
            If True: This will search for username = like(%admin%) and name = like(%Sara%).
            If False: This will search for username = like(%admin%, %Sara%) and name = like(%admin%, %Sara%).
        Use_AND_Operator (default = False): Makes the search with AND instead of OR.
        TextualQuery: Use a textual query, i.e: "id<1111"
        Join: Used to join query. This must be a tuple where the first element must be a child class
            of BaseModel and the second element must be an object from Search, i.e (UserModel, search)
        Operator (default =  QueryOperators.INCLUDE): A valid QueryOperator. This is used to apply a
            given operator in filter.
    """

    SearchBy: str = " "
    SearchValue: list = field(default_factory=list)
    Page: int = 1
    PerPage: int = 1000
    OrderBy: str = ""
    OrderDesc: bool = False
    MapColumnAndValue: bool = True
    Use_AND_Operator: bool = False
    TextualQuery: str = None
    Join: tuple = None
    Operator: QueryOperators = QueryOperators.INCLUDE

    @property
    def per_page(self):
        return self.PerPage

    @property
    def page(self):
        return self.Page


@dataclass
class Search(BaseDataClass):
    """
    Class that holds a structure to perform search and paginate it. This structure can
    be used by any :class:`BaseModel`, since SearchBy and OrderBy are valid column names.
    Use the method from_dict to create an instance from a dictionary.
    Fields:
        SearchBy: A table's column's name to search. You can pass multiple values by using comma separation,
        e.g:
            "username, name".
        This will perform a search in this to columns. SearchBy can also override
        the operator, e.g:
            "username: ==, name: !="
        Valid operator: "==, <,>,<=,>=,!=,[],]"
        SearchValue: The value to search. You can pass multiple values by using comma separation or a List, e.g:
            "John, Sara" or [ "John, Sara"]. This are safer than just comma separation.
        This will perform a search for this values for the given columns.
        Page: The current page to return.
        PerPage: Number of items per page.
        OrderBy: A table's column's name to order.
        OrderDesc: True if the order must be descendant.
        MapColumnAndValue (default = True): If True it will consider a 1:1 mapping for SearchBy:SearchValue.
            e.g: Column -> "username, name", Values -> "admin, Sara".
            If True: This will search for username = like(%admin%) and name = like(%Sara%).
            If False: This will search for username = like(%admin%, %Sara%) and name = like(%admin%, %Sara%).
        Use_AND_Operator (default = False): Makes the search with AND instead of OR.
        TextualQuery: Use a textual query, i.e: "id<1111"
        Join: Used to join query. This must be a tuple where the first element must be a child class
            of BaseModel and the second element must be an object from Search, i.e (UserModel, search)
        Operator (default =  QueryOperators.INCLUDE): A valid QueryOperator. This is used to apply a
            given operator in filter.
    """

    SearchBy: str
    SearchValue: list
    Page: int
    PerPage: int
    OrderBy: str
    OrderDesc: bool = False
    MapColumnAndValue: bool = True
    Use_AND_Operator: bool = False
    TextualQuery: str = None
    Join: tuple = None
    Operator: QueryOperators = QueryOperators.INCLUDE


@dataclass
class SearchResult(BaseDataClass):
    """
    Class that holds a structure to return a search result.
    Fields:
        result: Expect a list of entities. However, it can be any object list that implements the method to_dict().
        total: The total number of entities found.
    """

    resultList: list = field(default_factory=list)
    totalCount: int = 0

    def to_dict(self) -> dict:
        """
        Map this object to a dictionary.
        :return: The dictionary representation of this object.
        """
        search_result_asdict = asdict(self)
        search_result_asdict["resultList"] = [entity.to_dict() for entity in self.resultList]
        return search_result_asdict
