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

    def get_model_file_path(self):
        model_path_P = os.path.join(self.root_path,"model","modelP")
        model_path_S = os.path.join(self.root_path, "model", "modelS")
        if not os.path.isfile(model_path_P) and os.path.isfile(model_path_S):
            raise FileNotFoundError("The file {} doesn't exist.".format(model_path_P))
            raise FileNotFoundError("The file {} doesn't exist.".format(model_path_S))
        return model_path_P,model_path_S

    def vel_to_grid(self,latitude, longitude, depth, x_node, y_node, z_node,dx,dy,dz,grid_type,wave):

        file_path=self.get_template_file_path()
        output = os.path.join(self.root_path,"temp","input.txt")
        model_path=self.get_model_file_path()
        modelP_path=model_path[0]
        modelS_path = model_path[1]
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}'.format(lat=latitude,lon=longitude, depth=depth)
        df.iloc[2, 0] = 'VGGRID {xnd} {ynd} {znd} 0.0 0.0 -1.0  {dxsize:.2f} ' \
                        '{dysize:.2f} {dzsize:.2f} {type}'.format(xnd=x_node, ynd=y_node, znd=z_node,dxsize=dx,dysize=dy,dzsize=dz,type=grid_type)
        df.iloc[3, 0] = 'VGOUT '+ os.path.join(self.root_path,"model","layer")
        df.iloc[4, 0] = 'VGTYPE {wavetype}'.format(wavetype=wave)

        df.to_csv(output, index = False, header=True,encoding='utf-8')
        print(df)
        command="cat "+modelP_path+ " >> " +output
        sb.Popen(command, shell=True)
    # ######
    # ##Vel2Grid
    # ######
        command="Vel2Grid "+output
        sb.Popen(command,shell=True)
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




