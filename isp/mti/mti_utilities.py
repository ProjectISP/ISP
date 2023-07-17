import os
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth


class MTIManager:

    def __init__(self, st, inv, lat0, lon0, min_dist, max_dist):
        """
        Manage MTI files for run isola class program.
        st: stream of seismograms
        in: inventory
        """
        self.__st = st
        self.__inv = inv
        self.lat = lat0
        self.lon = lon0
        self.min_dist = min_dist
        self.max_dist = max_dist

    @staticmethod
    def __validate_file(file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(file_path))

    @staticmethod
    def __validate_dir(dir_path):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(dir_path))

    @property
    def root_path(self):
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.__validate_dir(root_path)
        return root_path

    @property
    def get_stations_dir(self):
        stations_dir = os.path.join(self.root_path, "input")
        self.__validate_dir(stations_dir)
        return stations_dir


    def get_stations_index(self):

        ind = []
        file_list = []
        dist1 = []
        for tr in self.__st:
            net = tr.stats.network
            station = tr.stats.station
            channel = tr.stats.channel
            coords = self.__inv.get_coordinates(tr.id)
            lat = coords['latitude']
            lon = coords['longitude']
            if ind.count(station):
                pass
            else:
                [dist, _, _] = gps2dist_azimuth(self.lat, self.lon, lat, lon, a=6378137.0, f=0.0033528106647474805)
                ind.append(station)
                item = '{net}:{station}::{channel}    {lat}    {lon}'.format(net=net,
                        station=station, channel=channel[0:2], lat=lat, lon=lon)

            # filter by distance
                if self.min_dist and self.max_dist > 0:
                    # do the distance filter
                    if dist >= self.min_dist:
                        file_list.append(item)
                        dist1.append(dist)
                        keydict = dict(zip(file_list, dist1))
                        file_list.sort(key=keydict.get)
                    if dist <= self.max_dist:
                        file_list.append(item)
                        dist1.append(dist)
                        keydict = dict(zip(file_list, dist1))
                        file_list.sort(key=keydict.get)
                # do not filter by distance
                else:
                    file_list.append(item)
                    dist1.append(dist)
                    keydict = dict(zip(file_list, dist1))
                    file_list.sort(key=keydict.get)

        self.stations_index = ind
        self.stream = self.sort_stream(dist1)


        deltas = self.get_deltas()

        data = {'item': file_list}

        df = pd.DataFrame(data, columns=['item'])
        #print(df)
        outstations_path = os.path.join(self.get_stations_dir, "stations.txt")
        #print(outstations_path)
        df.to_csv(outstations_path, header=False, index=False)
        return self.stream , deltas, outstations_path


    def sort_stream(self, dist1):
        stream = []
        stream_sorted_order = []

        for station in self.stations_index:
            st2 = self.__st.select(station=station)
            stream.append(st2)

        # Sort by Distance

        stream_sorted = [x for _, x in sorted(zip(dist1, stream))]
        # reverse from E N Z --> Z N E
        # reverse from 1 2 Z --> Z 2 1

        for stream_sort in stream_sorted:
            stream_sorted_order.append(stream_sort.reverse())

        return stream_sorted_order

    def get_deltas(self):
        deltas = []
        n =len(self.stream)
        for j in range(n):
            stream_unique = self.stream[j]
            delta_unique = stream_unique[0].stats.delta
            deltas.append(delta_unique)

        return deltas




