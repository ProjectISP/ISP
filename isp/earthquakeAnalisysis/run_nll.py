#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:26:28 2019

@author: robertocabieces
"""

import os
import pandas as pd
import subprocess as sb

class NllManager:
    def __init__(self):
        pass

    @property
    def root_path(self):
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_output")
        if not os.path.isdir(root_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(root_path))
        return root_path

    def get_template_file_path(self):
        file_path = os.path.join(self.root_path,"run","template")
        if not os.path.isfile(file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(file_path))
        return file_path

    def vel_to_grid(self,latitude, longitude,depth=0):
        # Example of modification

        file_path=self.get_template_file_path()

        output = os.path.join(self.root_path,"temp","input.txt")
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}'.format(lat=latitude,lon=longitude, depth=depth)
        # Copy the data frame
        df.to_csv(output, index = False, header=True,encoding='utf-8')
        print(df)
    #
    #
         #command="cat ./model/modelP >> ./temp/input.txt"
    #     sb.Popen(command,shell=True)
    #
    #
    # ######
    # ##Vel2Grid
    # ######
    # command="Vel2Grid ./temp/input.txt"
    # sb.Popen(command,shell=True)
    #
    # ######
    # ##Grid2Time
    # #######
    # file=path+"/"+"run"+"/G2T"
    # output=path+"/"+"temp"+"/G2T_temp.txt"
    # data=pd.read_csv(file)
    # df = pd.DataFrame(data)
    # df2=df
    # df2.to_csv(output, index = False, header=True,encoding='utf-8')
    # command="cat ./stations/stations.txt >> ./temp/G2T_temp.txt"
    # sb.Popen(command,shell=True)
    # command= "Grid2Time /Users/robertocabieces/Documents/NLL/programPython/temp/G2T_temp.txt"
    # #sb.Popen(command,shell=True)
    # #sb.run([command],capture_output=True)
    # sb.call(command,shell=True)
    # ##
    # #NonLinLoc
    # ##
    # command="NLLoc ./run/run"
    # sb.call(command,shell=True)




