from enum import Enum, unique

from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel


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
