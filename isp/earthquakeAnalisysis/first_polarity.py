#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:26:28 2019

@author: robertocabieces
"""

import pandas as pd
import os
from obspy import read_events
from isp.earthquakeAnalisysis.focmecobspy import _read_focmec
from isp.Utils.subprocess_utils import exc_cmd

class FirstPolarity:

    def __init__(self):
        """
        Manage FOCMEC files for run nll program.

        Important: The  obs_file_path is provide by the class :class:`PickerManager`.

        :param obs_file_path: The file path of pick observations.
        """
        #self.__dataless_dir = dataless_path
        #self.__obs_file_path = obs_file_path
        #self.__create_dirs()

    @staticmethod
    def __validate_dir(dir_path):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(dir_path))

    @property
    def root_path(self):
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_output")
        self.__validate_dir(root_path)
        return root_path

    @property
    def get_loc_dir(self):
        loc_dir = os.path.join(self.root_path, "loc")
        self.__validate_dir(loc_dir)
        return loc_dir

    @property
    def get_foc_dir(self):
        first_polarity_dir = os.path.join(self.root_path, "first_polarity")
        self.__validate_dir(first_polarity_dir)
        return first_polarity_dir

    def get_dataframe(self):
        Station = []
        Az = []
        Dip = []
        Phase = []
        Motion = []
        location_file = os.path.join(self.get_loc_dir, "last.hyp")
        df = pd.read_csv(location_file, delim_whitespace=True, skiprows=16)
        for i in range(len(df)):
            if df.iloc[i].Weight > 0.01:
                sta = str(df.iloc[i].PHASE)
                if len(sta) >= 5:
                    sta = sta[0:4]
                az = df.iloc[i].SAzim
                dip = df.iloc[i].RAz
                m = df.iloc[i].Pha
                ph = str(df.iloc[i].On)
                if dip >= 90:
                    dip = 180 - dip
                if ph[0] == "P" and m != "?":
                    Az.append(az)
                    Dip.append(dip)
                    Motion.append(m)
                    Station.append(sta)
        return Station, Az, Dip, Motion

    def get_NLL_info(self):
        location_file = os.path.join(self.get_loc_dir, "last.hyp")
        if os.path.isfile(location_file):
            cat = read_events(location_file)
            event = cat[0]
            origin = event.origins[0]
            return origin
        else:
            raise FileNotFoundError("The file {} doesn't exist. Please, run location".format(location_file))

    def create_input(self):

        Station,Az,Dip,Motion= self.get_dataframe()
        N = len(Station)
        fmt = "%4s  %6.2f  %6.2f%1s\n"
        #
        with open(os.path.join(self.get_foc_dir,'test.inp'), 'wt') as f:
            f.write("\n")  # first line is ignored!
            for j in range(N):
                f.write(fmt % (Station[j], Az[j], Dip[j], Motion[j]))

    def run_focmec(self):
        command=os.path.join(self.get_foc_dir,'rfocmec_UW')
        exc_cmd(command)

    def extract_focmec_info(self):
        cat = _read_focmec(os.path.join(self.get_foc_dir,'focmec.lst'))
        Plane_A = cat[0].focal_mechanisms[0].nodal_planes.nodal_plane_1
        strike_A = Plane_A.strike
        dip_A = Plane_A.dip
        rake_A = Plane_A.rake
        Plane_B = cat[0].focal_mechanisms[0].nodal_planes.nodal_plane_1
        strike_B = Plane_B.strike
        dip_B = Plane_B.dip
        rake_B = Plane_B.rake
        return cat,Plane_A


