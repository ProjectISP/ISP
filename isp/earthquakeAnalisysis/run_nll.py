#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:26:28 2019

@author: robertocabieces
"""

import os
from pathlib import Path
import fnmatch
import numpy as np
import pandas as pd
import shutil
from obspy.core.event import Origin
from obspy.geodetics import gps2dist_azimuth
from isp import ROOT_DIR
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Utils import ObspyUtil
from isp.Utils.subprocess_utils import exc_cmd


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
        self.__metadata_manager = None


    def find_files(self,base, pattern):
        '''Return list of files matching pattern in base folder.'''
        return [n for n in fnmatch.filter(os.listdir(base), pattern) if os.path.isfile(os.path.join(base, n))]



    @property
    def nll_bin_path(self):
        bin_path = os.path.join(ROOT_DIR, "NLL7", "src/bin")
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
    def get_run_template_global_file_path(self):
        run_template_global_file_path = os.path.join(self.get_run_dir, "global_template")
        self.__validate_file(run_template_global_file_path)
        return run_template_global_file_path

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

    def get_model_file_path(self, wave_type: str, model):
        """
        Gets the model path for S or P wave.

        :param wave_type: Either S or P wave.
        :return: The path for the model S or P.
        """

        if wave_type.upper() == "P":
            model_path_p = os.path.join(self.get_local_models_dir, "modelP")
            self.__validate_file(model_path_p)
            return model_path_p

        elif wave_type.upper() == "S":
            model_path_s = os.path.join(self.get_local_models_dir, "modelS")
            self.__validate_file(model_path_s)
            return model_path_s
        else:
            raise AttributeError("Wrong wave type. The wave type {} is not valid".format(wave_type))


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
    def get_local_models_dir3D(self):
        models_dir = os.path.join(self.root_path, "model3D")
        self.__validate_dir(models_dir)
        return models_dir

    @property
    def get_time_dir(self):
        time_dir = os.path.join(self.root_path, "time")
        self.__validate_dir(time_dir)
        return time_dir

    @property
    def get_time_global_dir(self):
        time_dir = os.path.join(self.root_path, "ak135")
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
        files = self.find_files(self.get_time_dir, 'layer.P.mod.hdr')
        file_name = os.path.join(self.get_time_dir, files[0])
        fa = open(file_name)
        # Reads the header file
        hline = fa.readline().split()
        xNum, yNum, zNum = map(int, hline[:3])
        xOrig, yOrig, zOrig, dx, dy, dz = map(float, hline[3:-2])
        fa.close()

        run_path = self.get_run_template_file_path
        data = pd.read_csv(run_path, header = None)
        travetimepath = os.path.join(self.get_time_dir, "layer")
        locationpath = os.path.join(self.get_loc_dir, "location")
        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}'.format(lat=latitude, lon=longitude, depth=0.0)
        df.iloc[3, 0] = 'LOCFILES {obspath} NLLOC_OBS {timepath} {locpath}'.format(obspath=self.__obs_file_path,
                            timepath=travetimepath,locpath=locationpath)
        if xNum == 2: # 1D Grid

           xNum = int(yNum)
           yNum = int(yNum)
           df.iloc[6, 0] = 'LOCGRID  {x} {y} {z} {xo} {yo} {zo} {dx} {dy} {dz} PROB_DENSITY  SAVE'.format(x=xNum,
                            y=yNum, z=zNum, xo=xOrig, yo=yOrig, zo=zOrig, dx=dx, dy=dy, dz=dz)
        else:

            #xNum = int(yNum / 2)
            #yNum = int(yNum / 2)
            df.iloc[6, 0] = 'LOCGRID  {x} {y} {z} {xo} {yo} {zo} {dx} {dy} {dz} PROB_DENSITY  SAVE'.format(x=xNum,
                        y=yNum, z=zNum, xo=xOrig, yo=yOrig, zo=zOrig, dx=dx, dy=dy, dz=dz)

        output = os.path.join(self.get_temp_dir, "run_temp.txt")
        df.to_csv(output, index=False, header=False, encoding='utf-8')
        return output

    def set_run_template_global(self):
        run_path = self.get_run_template_global_file_path
        data = pd.read_csv(run_path)
        travetimepath = os.path.join(self.get_time_global_dir, "ak135")
        locationpath = os.path.join(self.get_loc_dir, "location")+" 1"
        stations_path = os.path.join(self.get_stations_template_file_path)
        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS GLOBAL'
        df.iloc[3, 0] = 'INCLUDE {obspath} '.format(obspath=stations_path)
        df.iloc[4, 0] = 'LOCFILES {obspath} NLLOC_OBS {timepath} {locpath}'.format(obspath=self.__obs_file_path,
                                                                                   timepath=travetimepath,
                                                                                   locpath=locationpath)
        output = os.path.join(self.get_temp_dir, "run_temp.txt")
        df.to_csv(output, index=False, header=True, encoding='utf-8')
        return output

    def set_vel2grid_template(self, latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz,
                              grid_type, wave_type):
        file_path = self.get_vel_template_file_path
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}'.format(lat=latitude, lon=longitude, depth=0.0)
        df.iloc[2, 0] = 'VGGRID {xnd} {ynd} {znd} 0.0 0.0 {depth:.2f}  {dx:.2f} ' \
                        '{dy:.2f} {dz:.2f} {type}'.format(xnd=x_node, ynd=y_node, znd=z_node, depth=depth, dx=dx,
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

        df.iloc[1, 0] = "TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}".format(lat=latitude, lon=longitude, depth=0.0)
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

    @staticmethod
    def __append_files(file_path_to_cat: str, file_path_to_append: str):
        command = "cat {}".format(file_path_to_cat)
        exc_cmd(command, stdout=open(file_path_to_append, 'a'), close_fds=True)

    def vel_to_grid(self, latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz, grid_type, wave_type, model):
        if model == "2D":
            x_node = 2 # mandatory for 2D models
            output = self.set_vel2grid_template(latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz,
                                                grid_type, wave_type)
            model_path = self.get_model_file_path(wave_type,model)
            self.__append_files(model_path, output)
            output_path = Path(output)
            command = "{} {}".format(self.get_bin_file("Vel2Grid"), output_path.name)
            exc_cmd(command, cwd=output_path.parent)
        elif model == "3D":
             self.__write_header(latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz, wave_type)
             self.grid3d(wave_type)


    def grid_to_time(self, latitude, longitude, depth, dimension, option, wave, limit):

        self.stations_to_nll_v2(latitude, longitude, depth, limit)
        output = self.set_grid2time_template(latitude, longitude, depth, dimension, option, wave)
        self.__append_files(self.get_stations_template_file_path, output)
        output_path = Path(output)
        command = "{} {}".format(self.get_bin_file("Grid2Time"), output_path.name)
        exc_cmd(command, cwd=output_path.parent)


    def run_nlloc(self, latitude, longitude, depth, transform):

        if transform == "SIMPLE":
            output = self.set_run_template(latitude, longitude, depth)
            output_path = Path(output)
            command = "{} {}".format(self.get_bin_file("NLLoc"), output_path.name)


        elif transform == "GLOBAL":

            self.stations_to_nll_v2(latitude, longitude, depth, limit = 20000, transform="GLOBAL")
            stations_path = os.path.join(self.get_stations_template_file_path)
            temp_path = self.get_temp_dir
            shutil.copy(stations_path, temp_path)
            output = self.set_run_template_global()
            output_path = Path(output)
            command = "{} {}".format(self.get_bin_file("NLLoc"), output_path.name)

        return exc_cmd(command, cwd=output_path.parent)


    def stations_to_nll_v2(self, latitude_f, longitude_f, depth, limit, transform="SIMPLE"):

        try:
            metadata_manager = MetadataManager(self.__dataless_dir)
            inv = metadata_manager.get_inventory()
        except:
            raise FileNotFoundError("No dataless found at location {}.".format(self.__dataless_dir))

        station_names = []
        station_latitudes = []
        station_longitudes = []
        station_depths = []

        if inv:
            number_of_nets = len(inv)
            for j in range(number_of_nets):
                net = inv[j]
                for k in range(len(net)):
                    sta = net[k]
                    code = sta.code
                    latitude = sta.latitude
                    longitude = sta.longitude
                    elevation = sta.elevation/1000

                    # filter minimum distance
                    dist, _, _ = gps2dist_azimuth(latitude, longitude, latitude_f,longitude_f)
                    dist = dist/1000
                    if dist<limit:
                        station_names.append(code)
                        station_latitudes.append(latitude)
                        station_longitudes.append(longitude)
                        station_depths.append(elevation)

            if transform == "SIMPLE":

                data = {'Code': 'GTSRCE', 'Name': station_names, 'Type': 'LATLON', 'Lat': station_latitudes,
                    'Lon': station_longitudes, 'Z': '0.000', 'Depth': station_depths}

            if transform == "GLOBAL":
                data = {'Code': 'LOCSRCE', 'Name': station_names, 'Type': 'LATLON', 'Lat': station_latitudes,
                        'Lon': station_longitudes, 'Z': '0.000', 'Depth': station_depths}


            df = pd.DataFrame(data, columns=['Code', 'Name', 'Type', 'Lat', 'Lon', 'Z', 'Depth'])

            outstations_path = os.path.join(self.get_stations_dir, "stations.txt")
            df.to_csv(outstations_path, sep=' ', header=False, index=False)

        return outstations_path

    def get_NLL_info(self) -> Origin:
        location_file = os.path.join(self.get_loc_dir, "last.hyp")
        return ObspyUtil.reads_hyp_to_origin(location_file)

    def get_NLL_scatter(self):

        location_file = os.path.join(self.get_loc_dir, "last")
        command = "{} {} {} {}".format(self.get_bin_file("scat2latlon"), 1, self.get_loc_dir, location_file)
        exc_cmd(command)

        location_file_check = os.path.join(self.get_loc_dir, "last.hyp.scat.xyz")
        if os.path.isfile(location_file_check):
            my_array = np.genfromtxt(location_file_check, skip_header=3)
            y = my_array[:, 0]
            x = my_array[:, 1]
            z = my_array[:, 2]
            pdf = my_array[:, 4]

            pdf = np.array(pdf) / np.max(pdf)

            return x, y, z, pdf
        else:
            raise FileNotFoundError("The file {} doesn't exist. Please, run location".format(location_file))

    def ger_NLL_residuals(self):
        location_file = os.path.join(self.get_loc_dir, "last.hyp")
        if os.path.isfile(location_file):
            df = pd.read_csv(location_file, delim_whitespace=True, skiprows=17)
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

    def check_stations(self):
        location_file = os.path.join(self.get_loc_dir, "last.hyp")
        if os.path.isfile(location_file):
            df = pd.read_csv(location_file, delim_whitespace=True, skiprows=17)
            xp = []
            for i in range(len(df)):
                phase = df.iloc[i].On
                if df.iloc[i].Res > 0 and phase[0].upper() == "P":
                    xp.append(df.iloc[i].PHASE)

        return xp

    def load_stations_file(self):
        stations_path = os.path.join(self.get_stations_dir, "stations.txt")
        stations = np.loadtxt(stations_path, dtype = str)
        name = stations[:,1]
        lon = stations[:,4]
        lat = stations[:, 3]
        all = [name, lon, lat]
        return all

    def stations_match(self):
        stations_located = self.check_stations()
        all_stations_names = self.load_stations_file()

        stations = {}
        for name,lon,lat in zip(all_stations_names[0],all_stations_names[1],all_stations_names[2]):
            if name in stations_located:
                stations[name] = [lon, lat]

        return stations

    # read NLL 3D grids

    def grid3d(self, wave_type):
        if wave_type == "P":
            file_name = "layer.P.mod"
        elif wave_type == "S":
            file_name = "layer.S.mod"
        path = os.path.join(self.get_local_models_dir3D, file_name)
        aslow, xNum, yNum, zNum = self.read_modfiles(path)
        output_name = file_name+".buf"
        output= os.path.join(self.get_model_dir, output_name)
        with open(file_name, 'wb'):
            aslow.astype('float32').tofile(output)


    def read_modfiles(self, file_name):

        xNum, yNum, zNum, xOrig, yOrig, zOrig, dx, dy, dz = self.__read_header(file_name)
        aslow = np.empty([xNum, yNum, zNum])

        for k in range(zNum):
            new_k = k * int(dz)
            depth = int(zOrig) + new_k
            strdepth = str(depth)
            fm_name = file_name + strdepth + '.mod'
            fm = open(fm_name)
            for j in range(yNum):
                line = fm.readline().split()
                for i in range(xNum):
                    vel = float(line[i])
                    slow = dx * 1. / vel
                    aslow[i, j, k] = slow
        fm.close()
        return aslow, xNum, yNum, zNum

    def __read_header(self, file_name):

        fa = open(file_name + '.hdr')

        # Reads the header file
        hline = fa.readline().split()
        xNum, yNum, zNum = map(int, hline[:3])
        xOrig, yOrig, zOrig, dx, dy, dz = map(float, hline[3:-2])
        hline = fa.readline().split()
        Transf, lat0, lon0, Rot0 = hline[1], hline[3], hline[5], hline[7]
        fa.close()
        return xNum, yNum, zNum, xOrig, yOrig, zOrig, dx, dy, dz



    def __write_header(self, latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz, wave_type):
        """
        Take the parameters of the 3D_Grid, and writes the header file in model
        """
        if wave_type == "P":
            file_name = os.path.join(self.get_local_models_dir3D,"layer.P.mod.hdr")
        elif wave_type == "S":
            file_name = os.path.join(self.get_local_models_dir3D,"layer.S.mod.hdr")


        shift_x = -0.5*float((x_node-1)*dx)
        shift_y = -0.5*float((y_node-1)*dy)

        coords = '{xnd} {ynd} {znd} {shift_x} {shift_y}  {depth} {dx:.2f} {dy:.2f} {dz:.2f} SLOW_LEN FLOAT\n'.format(xnd=x_node,
                    ynd=y_node,znd=z_node,shift_x=shift_x, shift_y=shift_y, depth=depth, dx=dx, dy=dy, dz=dz)
        transf = 'TRANSFORM SIMPLE LatOrig {xorig:.2f} LongOrig {yorig:.2f} RotCW 0.000000'.format(xorig=latitude, yorig=longitude)
        new_file = open(file_name, mode="w+", encoding="utf-8")
        new_file.write(coords)
        new_file.close()
        new_file = open(file_name, mode="a+", encoding="utf-8")
        new_file.write(transf)
        new_file.close()
        shutil.copy(file_name, self.get_model_dir)
