#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Aug  9 11:00:00 2020
    @author: robertocabieces
"""

import os
from isp.Gui import pyc
from os import listdir
from os.path import isfile, join
from obspy import read
from obspy.io.mseed.core import _is_mseed
from obspy.signal import PPSD
import pickle

class ppsdsISP(pyc.QObject):
    fileProcessed = pyc.pyqtSignal(int)

    def __init__(self, files_path, metadata, length, overlap, smoothing, period):
        """
                PPSDs utils for ISP.



                :param Path to the files to be processed
                       Metadata dataless or xml with stations metadata
        """
        super().__init__()
        self.files_path = files_path
        self.metadata = metadata
        self.length = length
        self.overlap = overlap/100
        self.smoothing = smoothing
        self.period = period
        self.check = False
        self.processedFiles = 0
        #self.selection = kwargs.pop('selection')
        #self.nets = kwargs.pop('nets').split(',')
        #self.stations = kwargs.pop('stations').split(',')
        #self.channels = kwargs.pop('channels').split(',')


    def create_dict(self):

        obsfiles = [f for f in listdir(self.files_path) if isfile(join(self.files_path, f))]
        obsfiles.sort()
        data_map = {}
        data_map['nets'] = {}
        size = 0
        for i in obsfiles:

            paths = os.path.join(self.files_path, i)
            if _is_mseed(paths):

                header = read(paths, headlonly=True)
                net = header[0].stats.network
                network = {net: {}}
                sta = header[0].stats.station
                stations = {sta: {}}
                chn = header[0].stats.channel
                # 1. Check if the net exists, else add
                #if self.selection[0] and self.nets.count(net) == 1:
                if net not in data_map['nets']:
                    data_map['nets'] = network
                #else:
                #    data_map['nets'] = network

                # 2. Check if the station exists, else add

                if sta not in data_map['nets'][net]:
                    data_map['nets'][net].update(stations)

                # 3. Check if the channels exists, else add

                if chn in data_map['nets'][net][sta]:
                    data_map['nets'][net][sta][chn].append(paths)

                else:
                    data_map['nets'][net][sta][chn] = [paths]

                size = size + 1

        return data_map, size

    def get_all_values(self,nested_dictionary):
        for key, value in nested_dictionary.items():
            if self.check == False:
                if type(value) is dict:
                    nested_dictionary[key] = self.get_all_values(value)
                else:
                    files = []
                    process_list = []
                    if type(value[0]) == list:
                        for j in value[0]:
                            st = read(j)
                            files.append(st[0])

                        ppsd = value[1]
                    else:
                        for j in value:
                            st = read(j)
                            files.append(st[0])
                        try:
                            ppsd = PPSD(files[0].stats, metadata=self.metadata, ppsd_length =self.length,
                                        overlap = self.overlap, period_smoothing_width_octaves = self.smoothing,
                                        period_step_octaves = self.period)
                        except:
                            pass

                    for i,j in zip(files, value):
                        try:
                            if self.check == False:
                                ppsd.add(i)
                                self.processedFiles = self.processedFiles + 1
                                print(i," processed")
                                self.fileProcessed.emit(self.processedFiles)
                            else:
                                process_list.append(j)
                        except:
                            process_list.append(j)

                    nested_dictionary[key] = [process_list, ppsd]

        return nested_dictionary

    @staticmethod
    def save_PPSDs(ppsds_dictionary, dir_path, name):
        with open(os.path.join(dir_path, name), 'wb') as f:
            pickle.dump(ppsds_dictionary, f)

    @staticmethod
    def load_PPSDs(dir_path, name):
        with open(os.path.join(dir_path, name), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def size_db(data_base):

        k = 0
        for key, value in data_base.items():

                if type(value) is dict:
                    k=k + ppsdsISP.size_db(value)
                else:
                    if type(value[0]) == list:
                        k = k + len(value[0])
                    else:
                        k = k + len(value)
        return k

    def add_db_files(self, data_map):

        obsfiles = [f for f in listdir(self.files_path) if isfile(join(self.files_path, f))]
        obsfiles.sort()

        size = 0
        for i in obsfiles:

            paths = os.path.join(self.files_path, i)
            if _is_mseed(paths):

                header = read(paths, headlonly=True)
                net = header[0].stats.network
                network = {net: {}}
                sta = header[0].stats.station
                stations = {sta: {}}
                chn = header[0].stats.channel
                # 1. Check if the net exists, else add
                #if self.selection[0] and self.nets.count(net) == 1:
                if net not in data_map['nets']:
                    data_map['nets'] = network
                #else:
                #    data_map['nets'] = network

                # 2. Check if the station exists, else add

                if sta not in data_map['nets'][net]:
                    data_map['nets'][net].update(stations)

                # 3. Check if the channels exists, else add

                if chn in data_map['nets'][net][sta] and type(data_map['nets'][net][sta][chn][0]) == list:
                    data_map['nets'][net][sta][chn][0].append(paths)

                else:
                    data_map['nets'][net][sta][chn][0] = [paths]

                size = size + 1

        return data_map, size





