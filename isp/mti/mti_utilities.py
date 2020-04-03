import os
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read_events
from obspy.core.event import Origin

from isp import ROOT_DIR
from isp.DataProcessing import DatalessManager
from isp.Gui import pw
from isp.Utils.subprocess_utils import exc_cmd


class MTIManager:

    def __init__(self, st, inv):
        """
        Manage MTI files for run isola class program.

        """
        self.__st = st
        self.__inv = inv

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
                ind.append(station)
                item = '{net}:{station}::{channel}    {lat}    {lon}'.format(net=net,
                        station=station, channel=channel[0:2],lat=lat,lon=lon)

                file_list.append(item)

        self.stations_index = ind
        self.stream = self.sort_stream()
        deltas = self.get_deltas()

        data = {'item': file_list}

        df = pd.DataFrame(data, columns=['item'])
        print(df)
        outstations_path = os.path.join(self.get_stations_dir, "stations.txt")
        print(outstations_path)
        df.to_csv(outstations_path, header=False, index=False)
        return self.stream , deltas, outstations_path


    def sort_stream(self):
        stream = []

        for station in self.stations_index:
            st2 = self.__st.select(station=station)
            stream.append(st2)

        return stream

    def get_deltas(self):
        deltas = []
        n =len(self.stream)
        for j in range(n):
            stream_unique = self.stream[j]
            delta_unique = stream_unique[0].stats.delta
            deltas.append(delta_unique)

        return deltas




