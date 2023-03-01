import os
import pickle

import pandas as pd

# CH1,CH2,HH1,HH2,BH1,BH2,CN1,CN2
from obspy import read, read_inventory
from obspy.io.mseed.core import _is_mseed

class test_dict:

    def __init__(self, data_path, metadata):
        self.data_path = data_path
        self.metadata = metadata

    def list_directory(self):

        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.data_path):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles


    def create_dict(self, **kwargs):

        net_list = kwargs.pop('net_list', "").split(',')
        sta_list = kwargs.pop('sta_list', "").split(',')
        chn_list = kwargs.pop('chn_list', "").split(',')

        obsfiles = self.list_directory()

        data_map = {}
        info = {}
        info_starts = []
        data_map['nets'] = {}
        size = 0

        for paths in obsfiles:

            if _is_mseed(paths):

                header = read(paths, headlonly=True)
                net = header[0].stats.network
                network = {net: {}}
                sta = header[0].stats.station
                stations = {sta: {}}
                chn = header[0].stats.channel
                starttime_ini = header[0].stats.starttime
                endtime_ini = header[0].stats.endtime
                meta = [net, sta, chn]
                key_meta = meta[0] + meta[1] + meta[2]
                ## Filter per nets
                # 1. Check if the net exists, else add
                if net not in data_map['nets']:
                    # 1.1 Check the filter per network

                    if net in net_list:
                        data_map['nets'].update(network)
                    # 1.2 the filter per network is not activated
                    if net_list[0] == "":
                        data_map['nets'].update(network)

                # 2. Check if the station exists, else add
                # try:
                if sta not in data_map['nets'][net]:
                    if sta in sta_list:
                        data_map['nets'][net].update(stations)
                    if sta_list[0] == "":
                        data_map['nets'][net].update(stations)
                # except:
                #    pass

                # 3. Check if the channels exists, else add
                # try:
                # 3.1 if already exists just add
                if chn in data_map['nets'][net][sta]:
                    if chn in chn_list:
                        data_map['nets'][net][sta][chn].append(paths)

                        size = size + 1
                        info[key_meta][2].append(starttime_ini)
                        if endtime_ini - info[key_meta][0][1] > 0:
                            info[key_meta][0][1] = endtime_ini


                        elif info[key_meta][0][0] - starttime_ini > 0:
                            info[key_meta][0][0] = starttime_ini


                    if chn_list[0] == "":
                        data_map['nets'][net][sta][chn].append(paths)

                        if endtime_ini - info[key_meta][0][1] > 0:
                            info[key_meta][0][1] = endtime_ini
                            info[key_meta][2].append(starttime_ini)

                        elif info[key_meta][0][0] - starttime_ini > 0:
                            info[key_meta][0][0] = starttime_ini

                        size = size + 1
                else:
                    # 3.2 if does't exist create a list
                    if chn in chn_list:
                        data_map['nets'][net][sta][chn] = [meta, paths]
                        starttime_chn = starttime_ini
                        endtime_chn = endtime_ini

                        info[key_meta] = [[starttime_chn, endtime_chn],
                                          self.metadata.select(channel=chn, station=sta), [starttime_chn]]

                        size = size + 1
                    if chn_list[0] == "":
                        data_map['nets'][net][sta][chn] = [meta, paths]
                        starttime_chn = starttime_ini
                        endtime_chn = endtime_ini
                        info[key_meta] = [[starttime_chn, endtime_chn],
                                          self.metadata.select(channel=chn, station=sta), [starttime_chn]]
                        size = size + 1

                # except:
                #    pass

        return data_map, size, info

if __name__ == "__main__":
    chn_list = "CH1,CH2,HH1,HH2,BH1,BH2,CN1,CN2"
    data_path = "/Volumes/NO NAME/UPFLOW/new_stuff/UP_test1"
    metadata = read_inventory("/Volumes/NO NAME/UPFLOW/new_stuff/metadata/METADATA_WITH_OREINTATIONS _CHANGED_BH.xml")
    test = test_dict(data_path,metadata)
    data_map, size, info = test.create_dict(chn_list=chn_list)
    print("end")