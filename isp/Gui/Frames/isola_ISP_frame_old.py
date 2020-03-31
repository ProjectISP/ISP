#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:11:13 2020

@author: olivar
"""

import os
import shutil
#from PyQt5 import uic, QtGui, QtCore, QtWidgets, QtWebEngineWidgets

from isp.Gui.Frames import BaseFrame
from isp.mti import ISP_invert
from isp.Gui import pw,qteng
from isp.Gui.Frames.uis_frames import UiMomentTensor
from isp import ROOT_DIR

# class MTIFrame(BaseFrame, UiMomentTensor):
#
#     def __init__(self):
#         super(MTIFrame, self).__init__()
#         self.setupUi(self)
#         #uic.loadUi("MomentTensor.ui", self)
#         #self.show()
#
#         #self.ISOLA_input = os.path.join(os.getcwd(), "mti/input")
#         #self.ISOLA_sac = os.path.join(os.getcwd(), "mti/invert/sac")
#         #self.ISOLA_pzfiles = os.path.join(os.getcwd(), "mti/invert/pzfiles")
#
#         self.connect_menu_actions()
#         self.connect_buttons()
#
#     @property
#     def root_isola_path(self):
#         root_isola_path = os.path.join(ROOT_DIR, "mti")
#         if not os.path.isdir(root_isola_path):
#             raise FileNotFoundError("The dir {} doesn't exist"
#                                     .format(root_isola_path))
#         return root_isola_path
#
#
#
#
#     def connect_menu_actions(self):
#         self.actionWaveforms.triggered.connect(self.get_waveforms_dir)
#         self.actionStation_information.triggered.connect(self.get_station_info_file)
#         self.actionEvent_information.triggered.connect(self.get_event_information)
#         self.actionEarth_models_Read.triggered.connect(self.get_earth_model)
#         self.actionSet_paths.triggered.connect(self.set_path)
#
#     def connect_buttons(self):
#         self.pushButton_mti.clicked.connect(self.mti)
#
#
#     def set_path(self):
#         root = self.root_isola_path
#
#         self.ISOLA_input = os.path.join(root, "input")
#         print(self.ISOLA_input)
#         self.ISOLA_sac = os.path.join(root, "invert/sac")
#         self.ISOLA_pzfiles = os.path.join(root, "invert/pzfiles")
#
#     def get_waveforms_dir(self):
#         self.waveform_dir = pw.QFileDialog.getExistingDirectory()
#         if self.waveform_dir:
#             self.checkBox_waveforms.setCheckable(True)
#             self.checkBox_waveforms.setChecked(True)
#             self.checkBox_waveforms.setEnabled(False)
#
#     def get_station_info_file(self):
#         self.stinfo_file = pw.QFileDialog.getOpenFileName()[0]
#         if self.stinfo_file:
#             self.checkBox_station.setCheckable(True)
#             self.checkBox_station.setChecked(True)
#             self.checkBox_station.setEnabled(False)
#
#     def get_event_information(self):
#         self.evinfo_file = pw.QFileDialog.getOpenFileName()[0]
#         if self.evinfo_file:
#             self.checkBox_event.setCheckable(True)
#             self.checkBox_event.setChecked(True)
#             self.checkBox_event.setEnabled(False)
#
#     def get_earth_model(self):
#         self.model_file = pw.QFileDialog.getOpenFileName()[0]
#         if self.model_file:
#             self.checkBox_model.setCheckable(True)
#             self.checkBox_model.setChecked(True)
#             self.checkBox_model.setEnabled(False)
#
#     def copy_temp_files(self):
#
#         # delete old temp files
#         for temp_dir in [self.ISOLA_input, self.ISOLA_sac]:
#             for file in os.listdir(temp_dir):
#                 file_path = os.path.join(temp_dir, file)
#                 os.unlink(file_path)
#
#         # copy new temp files
#         shutil.copyfile(self.evinfo_file, os.path.join(self.ISOLA_input, "event.isl"))
#         shutil.copyfile(self.stinfo_file, os.path.join(self.ISOLA_input, "network.stn"))
#         shutil.copyfile(self.model_file, os.path.join(self.ISOLA_input, "crustal.dat"))
#
#         for file in os.listdir(self.waveform_dir):
#             file_path = os.path.join(self.waveform_dir, file)
#             shutil.copyfile(file_path, os.path.join(self.ISOLA_sac, file))
#
#     def mti(self):
#         self.copy_temp_files()
#         ISP_invert.perform_mti()
#         url = qteng.QUrl.fromLocalFile(os.path.join(os.getcwd(), "mti/output/index.html"))
#         self.widget.load(url)
