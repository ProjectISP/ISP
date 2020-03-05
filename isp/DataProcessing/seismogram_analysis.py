from enum import Enum, unique

from obspy import read, UTCDateTime
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

import numpy as np


from isp.Structures.structures import TracerStats
from isp.Utils import ObspyUtil, Filters
import matplotlib.dates as mdt


@unique
class ArrivalsModels(Enum):
    Iasp91 = "iasp91"
    Ak135 = "ak135"


class SeismogramAnalysis:

    def __init__(self, station_latitude, station_longitude):

        self.station_latitude = station_latitude
        self.station_longitude = station_longitude

    def get_phases_and_arrivals(self, event_lat, event_lon, depth):
        dist = locations2degrees(event_lat, event_lon, self.station_latitude, self.station_longitude)
        model = TauPyModel(model=ArrivalsModels.Iasp91.value)

        travel_times = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist)

        arrival = [(tt.time, tt.name) for tt in travel_times]
        arrivals, phases = zip(*arrival)

        return phases, arrivals


class SeismogramData:

    def __init__(self, file_path):
        if file_path:
            st = read(file_path)
            self.__tracer = st[0]
            self.stats = TracerStats.from_dict(self.tracer.stats)

    @classmethod
    def from_tracer(cls, tracer):
        sd = cls(None)
        sd.set_tracer(tracer)
        return sd

    @property
    def tracer(self):
        return self.__tracer

    def set_tracer(self, tracer):
        self.__tracer = tracer
        self.stats = TracerStats.from_dict(self.__tracer.stats)

    def __send_filter_error_callback(self, func, msg):
        if func:
            func(msg)

    def get_waveform(self, filter_error_callback=None, **kwargs):
        filter_value = kwargs.get("filter_value", Filters.Default)
        f_min = kwargs.get("f_min", 0.)
        f_max = kwargs.get("f_max", 0.)
        start_time = kwargs.get("start_time", self.stats.StartTime)
        end_time = kwargs.get("end_time", self.stats.EndTime)

        tr = self.tracer

        tr.detrend(type="demean")

        try:
            if not ObspyUtil.filter_trace(tr, filter_value, f_min, f_max):
                self.__send_filter_error_callback(filter_error_callback,
                                                  "Lower frequency {} must be "
                                                  "smaller than Upper frequency {}".format(f_min, f_max))
        except ValueError as e:
            print(e)
            self.__send_filter_error_callback(filter_error_callback, str(e))

        sample_rate = self.stats.Sampling_rate
        dt = 1/sample_rate

        try:
            tr.trim(starttime=start_time, endtime=end_time)
        except:
            print("Please Check Starttime and Endtime")

        t = [UTCDateTime(start_time + n * dt).matplotlib_date for n in range(0, len(tr.data))]
        t_sec = np.arange(0, len(tr.data) / sample_rate, 1. / sample_rate)
        return t, t_sec, tr.data