#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 09:04:24 2021

@author: robertocabieces
"""
import os
import pickle
from obspy import read
import obspy
from obspy.io.mseed.core import _is_mseed
from obspy import read_inventory
from obspy import Stream
import numpy as np
import math
from multiprocessing import Pool

from isp.Gui import pyc
from isp.ant.signal_processing_tools import noise_processing

#print_messages = True if (config_object["OPTIONS"]["print_messages"] == "True") else False
# compress_data = True if (config_object["OPTIONS"]["compress_data"] == "True") else False
# output_files_path = config_object["PATHS"]["output_files_path"]
# data_domain = config_object["OPTIONS"]["data_domain"]
# gaps_tol = int(config_object["OPTIONS"]["gaps_tol"])
# taper_max_percent = float(config_object["OPTIONS"]["taper_max_percent"])
# trim_interval = int(config_object["OPTIONS"]["trim_interval"])
# save_files = True if (config_object["OPTIONS"]["save_files"] == "True") else False
# num_hours_dict_matrix = int(config_object["OPTIONS"]["num_hours_dict_matrix"])
# whiten_freq_min = float(config_object["OPTIONS"]["whiten_freq_min"])
# whiten_freq_max = float(config_object["OPTIONS"]["whiten_freq_max"])
# num_minutes_dict_matrix = int(config_object["OPTIONS"]["num_minutes_dict_matrix"])


# @numba.jitclass([('width', numba.float64), ('height', numba.float64)])
class noise_organize(pyc.QObject):

    send_message = pyc.pyqtSignal(str)

    def __init__(self, data_path, output_files_path, metadata, param_dict):
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata
        self.dict_matrix = None
        self.list_item = None
        self.inc_time = None
        self.num_rows = None
        self.parameters = param_dict
        self.data_domain = "frequency"
        self.save_files = "True"
        self.taper_max_percent = 0.05
        self.num_hours_dict_matrix = 6
        self.num_minutes_dict_matrix = 15
        self.gaps_tol = 120
        self.output_files_path = output_files_path


        # extraxting parameters

        self.f1 = param_dict["f1"]
        self.f2 = param_dict["f2"]
        self.f3 = param_dict["f3"]
        self.f4 = param_dict["f4"]
        self.waterlevelSB = param_dict["waterlevel"]
        self.unitsCB = param_dict["units"]
        self.factor =param_dict["factor"]
        self.timenorm = param_dict["method"]
        self.timewindow = param_dict["timewindow"]
        self.freqbandwidth = param_dict["freqbandwidth"]
        self.remove_responseCB = param_dict["remove_responseCB"]
        self.decimationCB = param_dict["decimationCB"]
        self.time_normalizationCB = param_dict["time_normalizationCB"]
        self.whitheningCB = param_dict["whitheningCB"]

    def test(self):

        self.send_message.emit("message")

    def list_directory(self):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.data_path):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

    #def get_inventory(self):

    #    inv = read_inventory(self.metadata_path)
    #    return inv

    def create_dict(self, **kwargs):

        net_list = kwargs.pop('net_list', "").split(',')
        sta_list = kwargs.pop('sta_list', "").split(',')
        chn_list = kwargs.pop('chn_list', "").split(',')

        obsfiles = self.list_directory()

        data_map = {}
        info = {}
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
                try:
                    if sta not in data_map['nets'][net]:
                        if sta in sta_list:
                            data_map['nets'][net].update(stations)
                        if sta_list[0] == "":
                            data_map['nets'][net].update(stations)
                except:
                    pass

                # 3. Check if the channels exists, else add
                try:
                    # 3.1 if already exists just add
                    if chn in data_map['nets'][net][sta]:
                        if chn in chn_list:
                            data_map['nets'][net][sta][chn].append(paths)

                            size = size + 1

                            if endtime_ini - info[key_meta][0][1] > 0:
                                info[key_meta][0][1] = endtime_ini

                            elif info[key_meta][0][0] - starttime_ini > 0:
                                info[key_meta][0][0] = starttime_ini

                        if chn_list[0] == "":
                            data_map['nets'][net][sta][chn].append(paths)

                            if endtime_ini - info[key_meta][0][1] > 0:
                                info[key_meta][0][1] = endtime_ini

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
                                              self.inventory.select(channel=chn, station=sta)]

                            size = size + 1
                        if chn_list[0] == "":
                            data_map['nets'][net][sta][chn] = [meta, paths]
                            starttime_chn = starttime_ini
                            endtime_chn = endtime_ini
                            info[key_meta] = [[starttime_chn, endtime_chn],
                                              self.inventory.select(channel=chn, station=sta)]
                            size = size + 1

                except:
                    pass

        return data_map, size, info

    def fftzeropad(self, data):
        current_size = len(data)
        new_size = 2 ** int(np.log2(current_size) + 1)
        pad_size = new_size - current_size
        if pad_size % 2 == 0:
            left_pad = int(pad_size / 2)
            right_pad = left_pad
        else:
            left_pad = int(pad_size / 2)
            right_pad = pad_size - left_pad
        datapad = np.pad(data, [left_pad, right_pad], 'constant')
        # print(str(np.log2(left_pad+right_pad+current_size)))
        return datapad

    def process_col_matrix(self, j, fill_gaps=True):

        res = []
        tr = obspy.read(self.list_item[1 + j])[0]

        if self.remove_responseCB:
            tr = self.__remove_response(tr,self.f1, self.f2, self.f3, self.f4, self.water_level, self.unitsCB)

        if self.decimationCB:
            tr.decimate(factor=self.factor, no_filter = False)

        # for i in range(self.num_rows):
        for i in range(len(self.inc_time) - 1):
            tr_test = tr.copy()
            tr_test.trim(starttime=tr.stats.starttime + self.inc_time[i],
                         endtime=tr.stats.starttime + self.inc_time[i + 1])
            # TODO IMPLEMENT THE DECONVOLUTION AND THE CORRECT QUALITY CONTROL --> SEND TO BLACKLIST
            if fill_gaps:
                st = self.fill_gaps(Stream(traces=tr_test), tol=self.gaps_tol)
                if st == []:
                    tr_test.data = np.zeros(len(tr_test.data),dtype=np.complex64)
                else:
                    tr_test = st[0]
            if (self.data_domain == "frequency") and len(tr[:]) > 0:
                n = tr_test.count()
                if n > 0:
                    D = 2 ** math.ceil(math.log2(n))
                    print(tr_test)
                    # tr_test.plot()
                    # Prefilt
                    tr_test.detrend(type='simple')
                    tr_test.taper(max_percentage=0.05)
                    process = noise_processing(tr_test)
                    if self.time_normalizationCB:
                        process.normalize(norm_win=3, norm_method='ramn')
                    if self.whitheningCB:
                        process.whiten_new(freq_width=self.freqbandwidth, taper_edge=True)
                    try:
                        # self.dict_matrix['data_matrix'][i, j, :] = np.fft.rfft(process.tr.data, D)
                        res.append(np.fft.rfft(process.tr.data, D))
                    except:
                        res.append(None)
                        print("dimensions does not agree")
                else:
                    res.append(None)
            else:
                res.append(None)

        return res

    def create_dict_matrix(self, list_item, info_item):
        # create an object to compress

        self.dict_matrix = {'data_matrix': [], 'metadata_list': [], 'date_list': []}
        print(" -- Matrix: " + list_item[0][0] + list_item[0][1] + list_item[0][2])

        # 1.- dict_matrix['date_list']
        date_ini = info_item[0][0].julday
        date_end = info_item[0][1].julday
        self.dict_matrix['date_list'] = [date_ini + d for d in range(date_end - date_ini + 1)]

        # 2.- dict_matrix['metadata_list']
        self.dict_matrix['metadata_list'] = info_item[1]

        # 3.- dict_matrix['data_matrix']

        num_minutes = self.num_minutes_dict_matrix  # 15min 96 incrementos
        self.num_rows = int((24 * 60) / num_minutes)
        num_columns = len(list_item) - 1
        N = num_minutes * 60 * 5 + 1  # segundos*fs
        DD = 2 ** math.ceil(math.log2(N))
        self.list_item = list_item
        # ······
        # f = [0, 1, ..., n / 2 - 1, n / 2] / (d * n) if n is even
        # f = [0, 1, ..., (n - 1) / 2 - 1, (n - 1) / 2] / (d * n) if n is odd
        # ······
        DD_half_point = int(((DD) / 2) + 1)
        self.dict_matrix['data_matrix'] = np.zeros((self.num_rows, num_columns, DD_half_point), dtype=np.complex64)
        self.inc_time = [i * 60 * num_minutes for i in range(self.num_rows + 1)]
        with Pool(processes=6) as pool:
            r = pool.map(self.process_col_matrix, range(num_columns))

        j = 0
        for col in r:
            i = 0
            for row in col:
                if row is not None and row.size == DD_half_point:
                    self.dict_matrix['data_matrix'][i, j, :] = row
                i += 1
            j += 1

        print("Finalizado", info_item)

        # compressing matrix
        # dict_matrix['data_matrix']=xr.DataArray(dictmatrix)
        if self.save_files:
            print(" -- File: " + self.output_files_path + '/' + list_item[0][0] + list_item[0][1] + list_item[0][2])
            path = self.output_files_path + '/' + list_item[0][0] + list_item[0][1] + list_item[0][2]
            print("Saving to ", path)
            file_to_store = open(path, "wb")
            pickle.dump(self.dict_matrix, file_to_store)

        print(self.dict_matrix)
        return self.dict_matrix

    def create_all_dict_matrix(self, list_raw, info):

        all_dict_matrix = []

        for list_item in list_raw:
            key_info = list_item[0][0] + list_item[0][1] + list_item[0][2]  # 'XTCAPCBHE', ...
            dict_matrix = self.create_dict_matrix(list_item, info[key_info])
            all_dict_matrix.append(dict_matrix)

        return all_dict_matrix

    def get_all_values(self, nested_dictionary):
        list_raw = []
        for key, value in nested_dictionary.items():

            if type(value) is dict:
                list_raw.extend(self.get_all_values(value))

            else:
                list_raw.append(value)

        return list_raw

    def __check_starttimes(self, st1, st2, st3):

        endtime1 = st1[0].stats.endtime
        starttime1 = st2[0].stats.starttime
        endtime2 = st2[0].stats.endtime
        starttime2 = st3[0].stats.starttime
        if abs(starttime1 - endtime1) and abs(starttime2 - endtime2) < 600:
            check = True
        else:
            check = False
        return check

    def check_gaps(self, gaps, tol):

        time_gaps = []
        for i in gaps:
            time_gaps.append(i[6])

        sum_total = sum(time_gaps)

        if sum_total > tol:
            check = True
        else:
            check = False

        return check

    def fill_gaps(self, st, tol):

        gaps = st.get_gaps()

        if len(gaps) > 0 and self.check_gaps(gaps, tol):
            st.print_gaps()
            st = []

        elif len(gaps) > 0 and self.check_gaps(gaps, tol) == False:
            st.print_gaps()
            st.merge(fill_value="interpolate", interpolation_samples=-1)

        elif len(gaps) == 0 and self.check_gaps(gaps, tol) == False:
            pass

        return st

    def temporal_window(self, channel_list):

        starts = []
        ends = []

        for path in range(len(channel_list)):
            header = read(path, headlonly=True)
            starttime = header[0].stats.starttime
            endtime = header[0].stats.endtime
            starts.append(starttime)
            ends.append(endtime)
        min_start = min(starts)
        max_ends = max(ends)

        return min_start, max_ends

    # def chop_all_data(self, list_raw):
    #
    #     for chn in list_raw:
    #         n = len(chn[1:]) - 2
    #         for k in range(1, n, 1):
    #             st1 = read(chn[k])
    #             st2 = read(chn[k + 1])
    #             st3 = read(chn[k + 2])
    #             check = self.__check_starttimes(st1, st2, st3)
    #             if (check):
    #                 tr = st1[0] + st2[0] + st3[0]
    #                 dt = trim_interval * 3600
    #                 starttime = st2[0].stats.starttime - dt
    #                 endtime = st2[0].stats.endtime + dt
    #                 st = self.fill_gaps(Stream(traces=tr), tol=gaps_tol)
    #                 if len(st) > 0:
    #                     st.trim(starttime=starttime, endtime=endtime)
    #                     st.taper(max_percentage=taper_max_percent)

    def __remove_response(self, tr, f1, f2 , f3, f4, water_level, units):


        try:

            tr.remove_response(inventory=self.inventory, pre_filt = (f1, f2, f3, f4), output=units, water_level=water_level)

        except:

            print("Coudn't deconvolve", tr.stats)

        return tr