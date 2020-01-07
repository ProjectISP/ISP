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

    def get_run_template_file_path(self):
        run_file_path = os.path.join(self.root_path,"run","run")
        if not os.path.isfile(run_file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(run_file_path))
        return run_file_path

    def get_vel_template_file_path(self):
        v2g_file_path = os.path.join(self.root_path,"run","v2g_template")
        if not os.path.isfile(v2g_file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(v2g_file_path))
        return v2g_file_path

    def get_time_template_file_path(self):
        g2t_file_path = os.path.join(self.root_path,"run","g2t_template")
        if not os.path.isfile(g2t_file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(g2t_file_path))
        return g2t_file_path

    def get_stations_template_file_path(self):
        stations_file_path = os.path.join(self.root_path,"stations","stations.txt")
        if not os.path.isfile(stations_file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(stations_file_path))
        return stations_file_path

    def get_model_file_path(self):
        model_path_P = os.path.join(self.root_path,"model","modelP")
        model_path_S = os.path.join(self.root_path, "model", "modelS")
        if not os.path.isfile(model_path_P) and os.path.isfile(model_path_S):
            raise FileNotFoundError("The file {} doesn't exist.".format(model_path_P))
            raise FileNotFoundError("The file {} doesn't exist.".format(model_path_S))
        return model_path_P,model_path_S

    def vel_to_grid(self,latitude, longitude, depth, x_node, y_node, z_node, dx, dy, dz, grid_type, wave):

        file_path=self.get_vel_template_file_path()
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
        command="Vel2Grid "+output
        sb.Popen(command,shell=True)
        print("Generated Velocity Grid")

    def grid_to_time(self, latitude, longitude, depth, dimension, option, wave):
        #nll_manager.grid_to_time(self.grid_latitude_bind.value, self.grid_longitude_bind.value,
                                 #self.grid_depth_bind.value, self.comboBox_griddimension.currentText(),
                                # self.comboBox_angles.currentText(), self.comboBox_ttwave.currentText())

        file_path = self.get_time_template_file_path()
        output = os.path.join(self.root_path, "temp", "G2T_temp.txt")
        data=pd.read_csv(file_path)
        df = pd.DataFrame(data)


        df.iloc[1, 0] = 'TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}'.format(lat=latitude, lon=longitude, depth=depth)
        df.iloc[2, 0] = 'GTFILES '+ os.path.join(self.root_path,"model","layer")+" "+os.path.join(self.root_path,"time","layer")+" "+wave
        df.iloc[3, 0] = 'GTMODE {grid} {angles}'.format(grid=dimension,angles=option)
        print(df)

        df.to_csv(output, index = False, header=True,encoding='utf-8')
        command = "cat " + self.get_stations_template_file_path() + " >> " + output
        sb.Popen(command,shell=True)
        command= "Grid2Time "+ output
        sb.Popen(command,shell=True)



    def NLLoc(self,latitude, longitude, depth):
        run_path = self.get_run_template_file_path()
        data = pd.read_csv(run_path)
        obsfile = os.path.join(self.root_path, "obs", "output.txt")
        travetimepath=os.path.join(self.root_path, "time", "layer")
        locationpath=os.path.join(self.root_path, "loc", "location")
        output = os.path.join(self.root_path, "temp", "run_temp.txt")
        df = pd.DataFrame(data)
        df.iloc[1, 0] = 'TRANS SIMPLE {lat:.2f} {lon:.2f} {depth:.2f}'.format(lat=latitude, lon=longitude, depth=depth)
        df.iloc[3, 0] = 'LOCFILES {obspath} {NLLOC_OBS} {timepath} {locpath}'.format(obspath=obsfile, NLLOC_OBS="NLLOC_OBS",
                                                                                     timepath=travetimepath,locpath=locationpath)
        df.to_csv(output, index=False, header=True, encoding='utf-8')
        print(df)
        command="NLLoc "+ output
        sb.call(command,shell=True)
        print("Location Completed")