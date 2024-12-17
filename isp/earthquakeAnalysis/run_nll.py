# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: run_nll.py
# Program: surfQuake & ISP
# Date: January 2024
# Purpose: Manage Event Locator
# Author: Roberto Cabieces & Thiago C. Junqueira
#  Email: rcabdia@roa.es
# --------------------------------------------------------------------

import re
import os
import stat
from pathlib import Path
import fnmatch
import platform
import random
import string
from typing import Union
import numpy as np
import pandas as pd
import shutil
from obspy.core.event import Origin
from obspy.geodetics import gps2dist_azimuth
from isp import nll_templates, nll_ak135, BINARY_NLL_DIR
from isp.LocCore.geodetic_conversion import calculate_destination_coordinates
from isp.Utils import ObspyUtil
from isp.earthquakeAnalysis.nll_parse import load_nll_configuration
from isp.earthquakeAnalysis.structures import NLLConfig
from isp.Utils.subprocess_utils import exc_nll
from isp.DataProcessing.metadata_manager import MetadataManager

_os = platform.system()

class NllManager:

    def __init__(self, nll_config: Union[str, NLLConfig], metadata_path, working_directory):
        """
        Manage NonLinLoc program to locate seismic events.
        :param nll_config: Path to nll_config.ini file or to NLLConfig object.
        :param metadata_path: Path to metadata file.
        :param working_dirctory: Root path to folder to establish the working and output structure.
        """
        self.__get_nll_config(nll_config)
        self.__location_output = working_directory
        self.__create_dirs()
        self.__dataless_dir = metadata_path
        self.__metadata_manager = None

    def __get_nll_config(self, nll_config):
        if isinstance(nll_config, str) and os.path.isfile(nll_config):
            self.nll_config: NLLConfig = load_nll_configuration(nll_config)
        elif isinstance(nll_config, NLLConfig):
            self.nll_config = nll_config
        else:
            raise ValueError(f"mti_config {nll_config} is not valid. It must be either a "
                             f" valid real_config.ini file or a NLLConfig instance.")


    def find_files(self,base, pattern):
        """
        Return list of files matching pattern in base folder
        """
        return [n for n in fnmatch.filter(os.listdir(base), pattern) if os.path.isfile(os.path.join(base, n))]


    def clean_output_folder(self):

        """
        Cleans the destination folder and creates symbolic links for all files in the source folder.

        Args:
        destination_folder (str): Path to the destination folder.
        source_folder (str): Path to the source folder.

        """
        dir_path = os.path.join(self.__location_output, "loc")

        # Clean the destination folder
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove files or symbolic links
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove

    @property
    def root_path(self):
        root_path = self.__location_output
        self.__validate_dir(root_path)
        return root_path

    @property
    def get_run_template_file_path(self):
        temp_run_file_old = os.path.join(self.root_path, 'temp/run_temp.txt')
        if os.path.isfile(temp_run_file_old):
            os.remove(temp_run_file_old)
        run_file_path = os.path.join(nll_templates, "run_template")
        self.__validate_file(run_file_path)
        return run_file_path

    @property
    def get_run_template_global_file_path(self):
        run_template_global_file_path = os.path.join(nll_templates, "global_template")
        self.__validate_file(run_template_global_file_path)
        return run_template_global_file_path

    @property
    def get_vel_template_file_path(self):
        v2g_file_path = os.path.join(nll_templates, "v2g_template")
        self.__validate_file(v2g_file_path)
        return v2g_file_path

    @property
    def get_time_template_file_path(self):
        g2t_file_path = os.path.join(nll_templates, "g2t_template")
        self.__validate_file(g2t_file_path)
        return g2t_file_path

    @property
    def get_stations_template_file_path(self):
        stations_file_path = os.path.join(self.get_stations_dir, "stations.txt")
        self.__validate_file(stations_file_path)
        return stations_file_path

    def get_model_file_path(self, wave_type: str):
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
        models_dir = self.nll_config.grid_configuration.path_to_1d_model
        self.__validate_dir(models_dir)
        return models_dir

    @property
    def get_local_models_dir3D(self):
        models_dir = self.nll_config.grid_configuration.path_to_3d_model
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

        # model dir.
        self.__create_dir("model3D")

        # time dir.
        self.__create_dir("time")

        # loc dir.
        self.__create_dir("loc")

    def set_observation_file(self):

        self.__obs_file_path = self.nll_config.grid_configuration.path_to_picks

    def set_run_template(self, latitude, longitude):
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

        # Grid search inside for location robustness 1% inside the grid
        yNum = yNum*dy - round(0.02 * (yNum*dy) + 1)
        xNum = yNum*dy
        zNum = zNum*dz - round(0.02 * (zNum*dz) + 1)

        if self.nll_config.grid_configuration.model == "1D":
            xOrig = round(0.01*(xNum*dx) + 1)
            yOrig = round(0.01*(yNum*dy) + 1)
            zOrig = zOrig + round(0.01*zNum + 1)
        else:
            xOrig = round(0.01 * xOrig + 1)
            yOrig = round(0.01 * yOrig + 1)
            zOrig = zOrig + round(0.01 * zOrig + 1)


        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}'.format(lat=latitude, lon=longitude, depth=0.0)
        df.iloc[3, 0] = 'LOCFILES {obspath} NLLOC_OBS {timepath} {locpath}'.format(
            obspath=self.nll_config.grid_configuration.path_to_picks, timepath=travetimepath, locpath=locationpath)

        xNum = int(yNum)
        yNum = int(yNum)
        df.iloc[6, 0] = 'LOCGRID  {x} {y} {z} {xo} {yo} {zo} {dx} {dy} {dz} PROB_DENSITY  SAVE'.format(x=int(xNum),
                        y=int(yNum), z=int(zNum), xo=xOrig, yo=yOrig, zo=zOrig, dx=dx, dy=dy, dz=dz)

        if self.nll_config.location_parameters.method == 'GAU_ANALYTIC':
            df.iloc[8, 0] = ('LOCMETH GAU_ANALYTIC {maxDistStaGrid} {minNumberPhases} {maxNumberPhases} '
                             '{minNumberSphases} {VpVsRatio} {maxNum3DGridMemory} {minDistStaGrid} '
                             '{iRejectDuplicateArrivals}'.format(maxDistStaGrid=9999.0, minNumberPhases=4,
                                                                 maxNumberPhases=-1, minNumberSphases=-1,
                                                                 VpVsRatio=1.68, maxNum3DGridMemory=6, minDistStaGrid=5,
                                                                 iRejectDuplicateArrivals=0))

        #GAU_ANALYTIC 9999.0 4 - 1 - 1 1.68 6
        #LOCMETH EDT_OT_WT 9999.0 4 -1 -1 1.68 6 -1.0 1
        output = os.path.join(self.get_temp_dir, "run_temp.txt")
        df.to_csv(output, index=False, header=False, encoding='utf-8')
        return output

    @staticmethod
    def __secure_exec(bin_file:str):
        st = os.stat(bin_file)
        os.chmod(bin_file, st.st_mode | stat.S_IEXEC)

    def set_run_template_global(self):
        run_path = self.get_run_template_global_file_path
        data = pd.read_csv(run_path)
        travetimepath = os.path.join(nll_ak135, "ak135")
        locationpath = os.path.join(self.get_loc_dir, "location") + " 1"
        stations_path = os.path.join(self.get_stations_template_file_path)
        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS GLOBAL'
        df.iloc[3, 0] = 'INCLUDE {obspath} '.format(obspath=stations_path)
        df.iloc[4, 0] = 'LOCFILES {obspath} NLLOC_OBS {timepath} {locpath}'.format(
            obspath=self.nll_config.grid_configuration.path_to_picks, timepath=travetimepath, locpath=locationpath)
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
        bin_file = os.path.join(BINARY_NLL_DIR, file_name)
        self.__secure_exec(bin_file)
        if not os.path.isfile(bin_file):
            raise FileNotFoundError("The file {} doesn't exist. Check typos in file_name or make sure to run: "
                                    "python setup.py build_ext --inplace. These should create a binaries folder fo nll with "
                                    "the binary files."
                                    .format(bin_file))
        return bin_file

    @staticmethod
    def __append_files(file_path_to_cat: str, file_path_to_append: str):
        #command = "cat {}".format(file_path_to_cat)
        command = ["cat", file_path_to_cat]
        #command = "cat /Users/admin/Documents/iMacROA/SurfQuakeCore/examples/earthquake_locate/model1D/modelP"
        exc_nll(command, stdout=open(file_path_to_append, 'a'), close_fds=True)

    def vel_to_grid(self):
        """
        # Method to generate the velocity grid #
        :return: Extracts the velocity grid as layer*.buf and layer*.hdr inside working_dir/model
        template file temp.txt in working_dir/temp.txt
        """
        waves = []
        latitude = self.nll_config.grid_configuration.latitude
        longitude = self.nll_config.grid_configuration.longitude
        depth = self.nll_config.grid_configuration.depth
        x_node = int(self.nll_config.grid_configuration.x)
        y_node = int(self.nll_config.grid_configuration.y)
        z_node = int(self.nll_config.grid_configuration.z)
        dx = self.nll_config.grid_configuration.dx
        dy = self.nll_config.grid_configuration.dy
        dz = self.nll_config.grid_configuration.dz
        grid_type = self.nll_config.grid_configuration.grid_type
        p_wave_type = self.nll_config.grid_configuration.p_wave_type
        s_wave_type = self.nll_config.grid_configuration.s_wave_type
        model = self.nll_config.grid_configuration.model

        if model == "1D":
            x_node = 2 # mandatory for 1D models
            if p_wave_type:
                waves.append("P")
            if s_wave_type:
                waves.append("S")
            for wave in waves:
                    output = self.set_vel2grid_template(latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz,
                                                        grid_type, wave)
                    model_path = self.get_model_file_path(wave)
                    self.__append_files(model_path, output)
                    output_path = Path(output)
                    command = [self.get_bin_file("Vel2Grid"), output_path]
                    exc_nll(command, cwd=output_path.parent)

        elif model == "3D":
            if p_wave_type:
                waves.append("P")
            if s_wave_type:
                waves.append("S")
            for wave in waves:
             self.__write_header(latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz, wave)
             self.grid3d(wave)


    def grid_to_time(self):
        """
        # Method to generate the travel-time tables file #
        :return: Extracts the travel-times per wave type as
        [layer.P.STA.angle.buf, layer.P.STA.time.buf, layer.P.STA.time.hdr]
        inside ./working_dir/time
        template file at ./work_dir/temp/G2T_temp.txt
        """
        waves = []
        latitude = self.nll_config.grid_configuration.latitude
        longitude = self.nll_config.grid_configuration.longitude
        depth = self.nll_config.grid_configuration.depth
        grid = self.nll_config.travel_times_configuration.grid

        limit = self.nll_config.travel_times_configuration.distance_limit
        option = "ANGLES_YES"
        dimension = "GRID2D"

        if grid == "1D":
            dimension = "GRID2D"
        elif grid == "3D":
            dimension = "GRID3D"

        if self.nll_config.grid_configuration.p_wave_type:
            waves.append("P")
        if self.nll_config.grid_configuration.s_wave_type:
            waves.append("S")
        self.stations_to_nll_v2(latitude, longitude, depth, limit)

        for wave in waves:

            output = self.set_grid2time_template(latitude, longitude, depth, dimension, option, wave)
            self.__append_files(self.get_stations_template_file_path, output)
            output_path = Path(output)
            command = [self.get_bin_file("Grid2Time"), output_path]
            exc_nll(command, cwd=output_path.parent)


    def run_nlloc(self, num_iter=1):
        """
        # Method to run the event locations from the picking file and config_file.ini #
        :return: locations files *hyp inside ./working_dir/loc
        template file at ./work_dir/temp/run_temp.txt
        """
        latitude = self.nll_config.grid_configuration.latitude
        longitude = self.nll_config.grid_configuration.longitude
        transform = self.nll_config.grid_configuration.geo_transformation

        if transform == "SIMPLE":
            output = self.set_run_template(latitude, longitude)
            output_path = Path(output)
            command = [self.get_bin_file("NLLoc"), output_path]


        elif transform == "GLOBAL":
            if num_iter == 1:
                self.stations_to_nll_v2(latitude, longitude, limit=20000, transform="GLOBAL")

            stations_path = os.path.join(self.get_stations_template_file_path)
            temp_path = self.get_temp_dir
            shutil.copy(stations_path, temp_path)
            output = self.set_run_template_global()
            output_path = Path(output)
            command = [self.get_bin_file("NLLoc"), output]

        # include statistics
        stats_file = os.path.join(self.root_path, 'loc/last.stat_totcorr')
        temp_run_file = os.path.join(self.root_path, 'temp/run_temp.txt')
        if os.path.isfile(stats_file) and os.path.isfile(temp_run_file):
            self.__append_files(stats_file, temp_run_file)

        return exc_nll(command, cwd=output_path.parent)


    def stations_to_nll_v2(self, latitude_f, longitude_f, depth=0, limit=20000, transform="SIMPLE"):

        try:
            metadata_manager = MetadataManager(self.__dataless_dir)
            inv = metadata_manager.get_inventory()
            print(inv)
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
                    dist, _, _ = gps2dist_azimuth(latitude, longitude, latitude_f, longitude_f)
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

    @staticmethod
    def get_NLL_scatter(location_file):
        # method to be implemented by GUI
        # Usage: scat2latlon < decim_factor > < output_dir > < hyp_file_list >
        # firs modify file with no hyp extension

        location_file_cut = location_file[0:-4]
        output_folder = os.path.dirname(location_file)

        bin_file = os.path.join(BINARY_NLL_DIR, "scat2latlon")
        st = os.stat(bin_file)
        os.chmod(bin_file, st.st_mode | stat.S_IEXEC)
        if not os.path.isfile(bin_file):
            raise FileNotFoundError("The file {} doesn't exist. Check typos in file_name or make sure to run: "
                                    "python setup.py build_ext --inplace. These should create a binaries folder fo nll with "
                                    "the binary files."
                                    .format(bin_file))

        command = [bin_file, "1", output_folder, location_file_cut]
        exc_nll(command)
        #name = "location.20211001.021028.grid0.loc.hdr"
        location_file_name = os.path.basename(location_file)
        location_file_name_list = location_file_name.split(".")
        scat_file_name = (location_file_name_list[0]+"."+location_file_name_list[1] + "." +
                          location_file_name_list[2]+"."+location_file_name_list[3] + "." +
                          location_file_name_list[4]+"."+location_file_name_list[5]+"." +
                          "scat"+"."+"xyz")

        location_file_check = os.path.join(output_folder, scat_file_name)
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

    def check_stations(self):
        location_file = os.path.join(self.get_loc_dir, "last.hyp")
        if os.path.isfile(location_file):
            df = pd.read_csv(location_file, delim_whitespace=True, skiprows=16)
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
        for name, lon, lat in zip(all_stations_names[0],all_stations_names[1],all_stations_names[2]):
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
            file_name = os.path.join(self.get_local_models_dir3D, "layer.P.mod.hdr")
        elif wave_type == "S":
            file_name = os.path.join(self.get_local_models_dir3D, "layer.S.mod.hdr")

        x_width = float((x_node - 1) * dx)
        y_width = float((x_node - 1) * dx)
        shift_x = -0.5 * x_width
        shift_y = -0.5 * y_width
        lat_geo, lon_geo = calculate_destination_coordinates(latitude, longitude, abs(shift_x), abs(shift_y))

        # shift_x = -0.5*float((x_node-1)*dx)
        # shift_y = -0.5*float((y_node-1)*dy)

        coords = '{xnd} {ynd} {znd} {shift_x} {shift_y}  {depth} {dx:.2f} {dy:.2f} {dz:.2f} SLOW_LEN FLOAT\n'.format(xnd=x_node,
                    ynd=y_node, znd=z_node, shift_x=shift_x, shift_y=shift_y, depth=depth, dx=dx, dy=dy, dz=dz)
        transf = 'TRANSFORM SIMPLE LatOrig {xorig:.2f} LongOrig {yorig:.2f} RotCW 0.000000'.format(xorig=lat_geo, yorig=lon_geo)
        new_file = open(file_name, mode="w+", encoding="utf-8")
        new_file.write(coords)
        new_file.close()
        new_file = open(file_name, mode="a+", encoding="utf-8")
        new_file.write(transf)
        new_file.close()
        shutil.copy(file_name, self.get_model_dir)

class Nllcatalog:

    def __init__(self, working_directory):
        self.working_directory = os.path.join(working_directory, "loc")


    def find_files(self):

        self.obsfiles = []
        pattern = re.compile(r'.*\.grid0\.loc\.hyp$')  # Match files ending with ".grid0.loc.hyp"

        for top_dir, _, files in os.walk(self.working_directory):
            for file in files:
                # Exclude files starting with "._" or containing "sum"
                if file.startswith("._") or "sum" in file:
                    continue

                # If the file matches the desired pattern, add it to the list
                if pattern.match(file):
                    self.obsfiles.append(os.path.join(top_dir, file))

        # Remove specific file "location.sum.grid0.loc.hyp" from the results, if it exists
        self.obsfiles = [file for file in self.obsfiles if not file.endswith("location.sum.grid0.loc.hyp")]

    def generate_id(self, length: int) -> str:
        """
        Generate a random string with the combination of lowercase and uppercase letters.

        :param length: The size of the id key

        :return: An id of size length formed by lowe and uppercase letters.
        """
        letters = string.ascii_letters
        return "".join(random.choice(letters) for _ in range(length))

    def __create_from_origin(self, origin: Origin):

        # origin_id = generate_id_from_origin(origin)
        event_dict = {"id": self.generate_id(16), "origin_time": origin.time.datetime, "transformation": "SIMPLE",
                      "rms": origin.quality.standard_error, "latitude": origin.latitude,
                      "longitude": origin.longitude, "depth": origin.depth,
                      "uncertainty": origin.depth_errors["uncertainty"],
                      "max_horizontal_error": origin.origin_uncertainty.max_horizontal_uncertainty,
                      "min_horizontal_error": origin.origin_uncertainty.min_horizontal_uncertainty,
                      "ellipse_azimuth": origin.origin_uncertainty.azimuth_max_horizontal_uncertainty,
                      "number_of_phases": origin.quality.used_phase_count,
                      "azimuthal_gap": origin.quality.azimuthal_gap,
                      "max_distance": origin.quality.maximum_distance,
                      "min_distance": origin.quality.minimum_distance}

        return event_dict
    def run_catalog(self, summary_path):
        # TODO INCLUDE TRANSFORMATION
        dates = []
        transformations = []
        rmss = []
        lats = []
        longs = []
        depths = []
        uncertainties = []
        max_hor_errors = []
        min_hor_errors = []
        ellipses_azs = []
        no_phases = []
        azs_gap = []
        max_dists = []
        min_dists = []
        print("Creating Catalog")
        self.find_files()
        for event_file in self.obsfiles:
            origin: Origin = ObspyUtil.reads_hyp_to_origin(event_file)
            origin_time_formatted_string = origin.time.datetime.strftime("%m/%d/%Y, %H:%M:%S.%f")
            dates.append(origin_time_formatted_string)
            transformations.append("SIMPLE")
            rmss.append(origin.quality.standard_error)
            longs.append(origin.longitude)
            lats.append(origin.latitude)
            depths.append(origin.depth/1000)
            uncertainties.append(origin.depth_errors["uncertainty"])
            max_hor_errors.append(origin.origin_uncertainty.max_horizontal_uncertainty)
            min_hor_errors.append(origin.origin_uncertainty.min_horizontal_uncertainty)
            ellipses_azs.append(origin.origin_uncertainty.azimuth_max_horizontal_uncertainty)
            no_phases.append(origin.quality.used_phase_count)
            azs_gap.append(origin.quality.azimuthal_gap)
            max_dists.append(origin.quality.maximum_distance)
            min_dists.append(origin.quality.minimum_distance)

        events_dict = {'Origin Time': dates, 'RMS': rmss, 'lats': lats, 'longs': longs,
        'depths': depths, 'Uncertainty':uncertainties, 'Max_Hor_Error': max_hor_errors, 'Min_Hor_Error': min_hor_errors,
        'Ellipse_Az': ellipses_azs, 'No_phases': no_phases, 'Az_gap': azs_gap, 'Max_Dist': max_dists, 'Min Dist': min_dists}

        self.__write_dict(events_dict, summary_path)

    def __write_dict(self, events_dict, output):
        output = os.path.join(output, "catalog.txt")
        df_magnitudes = pd.DataFrame.from_dict(events_dict)
        df_magnitudes.to_csv(output, sep=";", index=False)

