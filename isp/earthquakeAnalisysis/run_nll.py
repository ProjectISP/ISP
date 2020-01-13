#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:26:28 2019

@author: robertocabieces
"""

import os
import pandas as pd
import subprocess as sb
from obspy.io.xseed import Parser
from os.path import isfile, join
from os import listdir
from obspy import read_events
from isp.DataProcessing import DatalessManager
from obspy.io.nlloc.util import read_nlloc_scatter

class NllManager:
    def __init__(self, obs_file_path,dataless_path):
        """
        Manage nll files for run nll program.

        Important: The  obs_file_path is provide by the class :class:`PickerManager`.

        :param obs_file_path: The file path of pick observations.
        """
        self.__data_less_path = dataless_path
        self.__obs_file_path = obs_file_path
        self.__create_dirs()
        self.stations_to_NLL()
    @property
    def root_path(self):
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_output")
        if not os.path.isdir(root_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(root_path))
        return root_path

    @property
    def get_run_template_file_path(self):
        run_file_path = os.path.join(self.get_run_dir, "run_template")
        self.__validate_file(run_file_path)
        return run_file_path

    @property
    def get_vel_template_file_path(self):
        v2g_file_path = os.path.join(self.get_run_dir, "v2g_template")
        self.__validate_file(v2g_file_path)
        return v2g_file_path

    @property
    def get_time_template_file_path(self):
        g2t_file_path = os.path.join(self.get_run_dir, "g2t_template")
        self.__validate_file(g2t_file_path)
        return g2t_file_path

    @property
    def get_stations_template_file_path(self):
        stations_file_path = os.path.join(self.get_stations_dir, "stations.txt")
        self.__validate_file(stations_file_path)
        return stations_file_path

    @property
    def get_models_files_path(self):
        model_path_p = os.path.join(self.get_model_dir, "modelP")
        model_path_s = os.path.join(self.get_model_dir, "modelS")
        self.__validate_file(model_path_s)
        self.__validate_file(model_path_p)

        return model_path_p, model_path_s

    @property
    def get_run_dir(self):
        run_dir = os.path.join(self.root_path, "run")
        self.__validate_dir(run_dir)
        return run_dir

    @property
    def get_loc_dir(self):
        loc_dir = os.path.join(self.root_path, "loc")
        self.__validate_dir(loc_dir)
        return loc_dir

    @property
    def get_temp_dir(self):
        temp_dir = os.path.join(self.root_path, "temp")
        self.__validate_dir(temp_dir)
        return temp_dir

    @property
    def get_model_dir(self):
        model_dir = os.path.join(self.root_path, "model")
        self.__validate_dir(model_dir)
        return model_dir

    @property
    def get_time_dir(self):
        time_dir = os.path.join(self.root_path, "time")
        self.__validate_dir(time_dir)
        return time_dir

    @property
    def get_stations_dir(self):
        stations_dir = os.path.join(self.root_path, "stations")
        self.__validate_dir(stations_dir)
        return stations_dir

    @staticmethod
    def __validate_file(file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(file_path))

    @staticmethod
    def __validate_dir(dir_path):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(dir_path))

    def __create_dir(self, dir_name):
        """
        Create a directory inside the root_path with a dir_name if it doesn't exists.

        :param dir_name: The name of the directory to be created. Only the name NOT the full path.

        :return:
        """
        dir_path = os.path.join(self.root_path, dir_name)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    def __create_dirs(self):
        """
        Create the necessary directories for this class.

        :return:
        """
        # temporary dir.
        self.__create_dir("temp")

        # station dir.
        self.__create_dir("stations")

        # model dir.
        self.__create_dir("model")

        # time dir.
        self.__create_dir("time")

        # loc dir.
        self.__create_dir("loc")

    def set_run_template(self, latitude, longitude, depth):
        run_path = self.get_run_template_file_path
        data = pd.read_csv(run_path)
        travetimepath = os.path.join(self.get_time_dir, "layer")
        locationpath = os.path.join(self.get_loc_dir, "location")
        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}'.format(lat=latitude, lon=longitude, depth=depth)
        df.iloc[3, 0] = 'LOCFILES {obspath} NLLOC_OBS {timepath} {locpath}'.format(obspath=self.__obs_file_path,
                                                                                   timepath=travetimepath,
                                                                                   locpath=locationpath)
        output = os.path.join(self.get_temp_dir, "run_temp.txt")
        df.to_csv(output, index=False, header=True, encoding='utf-8')
        return output

    def set_vel2grid_template(self, latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz, grid_type, wave_type):
        file_path = self.get_vel_template_file_path
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}'.format(lat=latitude, lon=longitude, depth=depth)
        df.iloc[2, 0] = 'VGGRID {xnd} {ynd} {znd} 0.0 0.0 -1.0  {dx:.2f} ' \
                        '{dy:.2f} {dz:.2f} {type}'.format(xnd=x_node, ynd=y_node, znd=z_node, dx=dx,
                                                          dy=dy, dz=dz, type=grid_type)
        df.iloc[3, 0] = 'VGOUT {}'.format(os.path.join(self.get_model_dir, "layer"))
        df.iloc[4, 0] = 'VGTYPE {wavetype}'.format(wavetype=wave_type)

        output = os.path.join(self.get_temp_dir, "input.txt")
        df.to_csv(output, index=False, header=True, encoding='utf-8')
        return output

    def set_grid2time_template(self, latitude, longitude, depth, dimension, option, wave_type):
        file_path = self.get_time_template_file_path
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)

        df.iloc[1, 0] = "TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}".format(lat=latitude, lon=longitude, depth=depth)
        df.iloc[2, 0] = "GTFILES {modelpath} {timepath} {wavetype}".\
            format(modelpath=os.path.join(self.get_model_dir, "layer"),
                   timepath=os.path.join(self.get_time_dir, "layer"), wavetype=wave_type)
        df.iloc[3, 0] = "GTMODE {grid} {angles}".format(grid=dimension, angles=option)

        output = os.path.join(self.get_temp_dir, "G2T_temp.txt")
        df.to_csv(output, index=False, header=True, encoding='utf-8')
        return output

    def vel_to_grid(self, latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz, grid_type, wave_type):

        output = self.set_vel2grid_template(latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz,
                                            grid_type, wave_type)
        model_path_p, model_path_s = self.get_models_files_path
        command = "cat " + model_path_p + " >> " + output
        sb.Popen(command, shell=True)
        command = "Vel2Grid " + output
        sb.Popen(command, shell=True)
        print("Velocity Grid Generated")

    def grid_to_time(self, latitude, longitude, depth, dimension, option, wave):

        output = self.set_grid2time_template(latitude, longitude, depth, dimension, option, wave)
        command = "cat " + self.get_stations_template_file_path + " >> " + output
        sb.Popen(command, shell=True)
        command = "Grid2Time " + output
        sb.Popen(command, shell=True)

    def run_nlloc(self, latitude, longitude, depth):
        output = self.set_run_template(latitude, longitude, depth)
        command = "NLLoc " + output
        sb.call(command, shell=True)
        print("Location Completed")

    def stations_to_NLL(self):
        dataless_directory= self.__data_less_path
        outstations_path = os.path.join(self.root_path,"stations")
        dm = DatalessManager(dataless_directory)
        station_names = []
        station_latitudes = []
        station_longitudes = []
        station_depths = []
        for st in dm.stations_stats:
            station_names.append(st.Name)
            station_latitudes.append(st.Lat)
            station_longitudes.append(st.Lon)
            station_depths.append(st.Depth/1000)

        data={'Code': 'GTSRCE', 'Name': station_names, 'Type': 'LATLON', 'Lon': station_longitudes,
              'Lat': station_latitudes, 'Z': '0.000', 'Depth': station_depths}

        df = pd.DataFrame(data, columns=['Code', 'Name', 'Type', 'Lat', 'Lon', 'Z', 'Depth'])

        df.to_csv(outstations_path + '/stations.txt', sep=' ', header=False, index=False)

    def get_NLL_info(self):

        location_file = os.path.join(self.root_path, "loc", "last.hyp")
        cat = read_events(location_file)
        event = cat[0]
        origin = event.origins[0]
        latitude = origin.latitude
        longitude = origin.longitude

        return latitude,longitude

    def get_NLL_scatter(self,latOrig,lonOrig):

        import math as mt
        import numpy as np

        location_file = os.path.join(self.root_path, "loc", "last.scat")
        data = read_nlloc_scatter(location_file)
        L = len(data)
        x = []
        y = []
        z = []
        pdf = []

        for i in range(L):
            x.append(data[i][0])
            y.append(data[i][1])
            z.append(data[i][2])
            pdf.append(data[i][3])
        x = np.array(x)
        y = np.array(y)

        conv = 111.111 * mt.cos(latOrig * 180 / mt.pi)
        x = (x / conv) + lonOrig
        y = (y / 111.111) + latOrig
        pdf = np.array(pdf) / np.max(pdf)

        return x,y,pdf

