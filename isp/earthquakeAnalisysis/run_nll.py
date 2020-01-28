#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:26:28 2019

@author: robertocabieces
"""

import math as mt
import os
import subprocess as sb

import numpy as np
import pandas as pd
from obspy import read_events
from obspy.core.event import Origin
from obspy.io.nlloc.util import read_nlloc_scatter

from isp import ROOT_DIR
from isp.DataProcessing import DatalessManager


class NllManager:

    def __init__(self, obs_file_path, dataless_path):
        """
        Manage nll files for run nll program.

        Important: The  obs_file_path is provide by the class :class:`PickerManager`.

        :param obs_file_path: The file path of pick observations.
        """
        self.__dataless_dir = dataless_path
        self.__obs_file_path = obs_file_path
        self.__create_dirs()

    @property
    def nll_bin_path(self):
        bin_path = os.path.join(ROOT_DIR, "NLL7", "bin")
        if not os.path.isdir(bin_path):
            raise FileNotFoundError("The dir {} doesn't exist. Please make sure to run: "
                                    "python setup.py build_ext --inplace. These should create a bin folder fo nll."
                                    .format(bin_path))
        return bin_path

    @property
    def root_path(self):
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_output")
        self.__validate_dir(root_path)
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
        model_path_p = os.path.join(self.get_local_models_dir, "modelP")
        model_path_s = os.path.join(self.get_local_models_dir, "modelS")
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
    def get_local_models_dir(self):
        models_dir = os.path.join(self.root_path, "local_models")
        self.__validate_dir(models_dir)
        return models_dir

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

    def set_dataless_dir(self, dir_path):
        self.__dataless_dir = dir_path

    def set_observation_file(self, file_path):
        self.__obs_file_path = file_path

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

    def get_bin_file(self, file_name):
        bin_file = os.path.join(self.nll_bin_path, file_name)
        if not os.path.isfile(bin_file):
            raise FileNotFoundError("The file {} doesn't exist. Check typos in file_name or make sure to run: "
                                    "python setup.py build_ext --inplace. These should create a bin folder fo nll with "
                                    "the binary files."
                                    .format(bin_file))
        return bin_file

    def vel_to_grid(self, latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz, grid_type, wave_type):

        output = self.set_vel2grid_template(latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz,
                                            grid_type, wave_type)
        model_path_p, model_path_s = self.get_models_files_path
        command = "cat " + model_path_p + " >> " + output
        sb.Popen(command, shell=True)
        command = "{} {}".format(self.get_bin_file("Vel2Grid"), output)
        sb.Popen(command, shell=True)
        print("Velocity Grid Generated")

    def grid_to_time(self, latitude, longitude, depth, dimension, option, wave):
        self.stations_to_nll()
        output = self.set_grid2time_template(latitude, longitude, depth, dimension, option, wave)
        print(self.get_stations_template_file_path, output)
        command = "cat " + self.get_stations_template_file_path + " >> " + output
        sb.Popen(command, shell=True, stderr=sb.PIPE)
        command = "{} {}".format(self.get_bin_file("Grid2Time"), output)
        p = sb.Popen(command, shell=True, stdout=sb.PIPE, stderr=sb.PIPE)
        stdout, stderr = p.communicate()
        if stderr:
            error_msg = (str(stderr, encoding="utf-8"))
            raise RuntimeError(error_msg)

    def run_nlloc(self, latitude, longitude, depth):
        output = self.set_run_template(latitude, longitude, depth)
        command = "{} {}".format(self.get_bin_file("NLLoc"), output)
        sb.call(command, shell=True)
        print("Location Completed")

    def stations_to_nll(self):
        dm = DatalessManager(self.__dataless_dir)
        if len(dm.stations_stats) == 0:
            raise FileNotFoundError("No dataless found at location {}.".format(self.__dataless_dir))
        station_names = []
        station_latitudes = []
        station_longitudes = []
        station_depths = []
        for st in dm.stations_stats:
            station_names.append(st.Name)
            station_latitudes.append(st.Lat)
            station_longitudes.append(st.Lon)
            station_depths.append(st.Depth/1000)

        data = {'Code': 'GTSRCE', 'Name': station_names, 'Type': 'LATLON', 'Lon': station_longitudes,
                'Lat': station_latitudes, 'Z': '0.000', 'Depth': station_depths}

        df = pd.DataFrame(data, columns=['Code', 'Name', 'Type', 'Lat', 'Lon', 'Z', 'Depth'])

        outstations_path = os.path.join(self.get_stations_dir, "stations.txt")
        df.to_csv(outstations_path, sep=' ', header=False, index=False)
        return outstations_path

    def get_NLL_info(self) -> Origin:
        location_file = os.path.join(self.get_loc_dir, "last.hyp")
        if os.path.isfile(location_file):
            cat = read_events(location_file)
            event = cat[0]
            origin = event.origins[0]
            return origin
        else:
            raise FileNotFoundError("The file {} doesn't exist. Please, run location".format(location_file))

    def get_NLL_scatter(self, lat_orig, lon_orig):

        location_file = os.path.join(self.get_loc_dir, "last.scat")
        if os.path.isfile(location_file):
            data = read_nlloc_scatter(location_file)
            data_size = len(data)
            x = []
            y = []
            z = []
            pdf = []

            for i in range(data_size):
                x.append(data[i][0])
                y.append(data[i][1])
                z.append(data[i][2])
                pdf.append(data[i][3])
            x = np.array(x)
            y = np.array(y)

            conv = 111.111 * mt.cos(lat_orig * 180 / mt.pi)
            x = (x / conv) + lon_orig
            y = (y / 111.111) + lat_orig
            pdf = np.array(pdf) / np.max(pdf)

            return x, y, pdf
        else:
            raise FileNotFoundError("The file {} doesn't exist. Please, run location".format(location_file))

    def ger_NLL_residuals(self):
        location_file = os.path.join(self.get_loc_dir, "last.hyp")
        if os.path.isfile(location_file):
            df = pd.read_csv(location_file, delim_whitespace=True, skiprows=16)
            xp = []
            yp = []
            xs = []
            ys = []
            for i in range(len(df)):
                phase = df.iloc[i].On

                if df.iloc[i].Weight > 0.01 and phase[0].upper() == "P":
                    yp.append(df.iloc[i].Res)
                    xp.append(df.iloc[i].PHASE)

            for i in range(len(df)):
                phase = df.iloc[i].On

                if df.iloc[i].Weight > 0.01 and phase[0].upper() == "S":
                    ys.append(df.iloc[i].Res)
                    xs.append(df.iloc[i].PHASE)

            return xp, yp, xs, ys
        else:
            raise FileNotFoundError("The file {} doesn't exist. Please, run location".format(location_file))

