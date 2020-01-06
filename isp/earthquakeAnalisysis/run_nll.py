#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:26:28 2019

@author: robertocabieces
"""

import os
import pandas as pd
import subprocess as sb

class NLL:
    def __init__(self):
        ##relative path
        self.path = self.__get_default_output_path()

        ##First read the Vel2Grid template


    ##Function to get the full standard location path
    def __get_default_output_path(self):
        run_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_output")
        if not os.path.isdir(run_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(run_path))

        return run_path

    @staticmethod
    def get_template_file_path():
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_output")
        if not os.path.isdir(root_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(root_path))
        file_path = os.path.join(root_path,"run","template")
        if not os.path.isfile(file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(file_path))


    ###Necessary add the info the user wants to modify

    def Vel2Grid_mod(self):
         #Example of modification
         #df.iloc[1, 0]='TRANS SIMPLE 33 -10 0.0'
         file = self.path + "/" + "run" + "/template"
         output = self.path + "/" + "temp" + "/input.txt"
         data = pd.read_csv(file)

         df = pd.DataFrame(data)

         ##Copy the data frame
         df2 = df
         df2.to_csv(output, index = False, header=True,encoding='utf-8')
         print(df2)
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




