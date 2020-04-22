# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 04:06:28 2020

@author: olivar
"""

import sys
from PyQt5 import uic, QtGui, QtCore, QtWidgets
import obspy
import pickle
from functools import partial
import isp.receiverfunctions.dialogs_utils as utils
from isp.Gui.Frames import UiReceiverFunctionsCut, BaseFrame


class CutEarthquakesDialog(BaseFrame, UiReceiverFunctionsCut):
    def __init__(self):
        super(CutEarthquakesDialog, self).__init__()
        self.setupUi(self)
        # connectionsx
        self.pushButton_2.clicked.connect(partial(self.get_path, 2))
        self.pushButton_3.clicked.connect(partial(self.get_path, 3))
        self.pushButton_4.clicked.connect(partial(self.get_path, 4))
        self.pushButton_5.clicked.connect(partial(self.get_path, 5))        
        self.pushButton_6.clicked.connect(self.cut_earthquakes)
        self.pushButton_7.clicked.connect(self.close)
    
    def get_path(self, pushButton):
        if pushButton == 2:
            path = QtWidgets.QFileDialog.getExistingDirectory()
            self.lineEdit.setText(path)
        elif pushButton == 3:
            path = QtWidgets.QFileDialog.getOpenFileName()[0]
            self.lineEdit_3.setText(path)
        elif pushButton == 4:
            path = QtWidgets.QFileDialog.getExistingDirectory()
            self.lineEdit_2.setText(path)
        elif pushButton == 5:
            path = QtWidgets.QFileDialog.getSaveFileName()[0]
            self.lineEdit_4.setText(path)

    def cut_earthquakes(self):
        data_path = self.lineEdit.text()
        station_metadata_path = self.lineEdit_3.text()
        earthquake_output_path = self.lineEdit_2.text()
        event_metadata_output_path = self.lineEdit_4.text()
        starttime = self.dateTimeEdit.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z")
        endtime = self.dateTimeEdit_2.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z")
        min_mag = self.doubleSpinBox_2.value()
        min_snr = self.doubleSpinBox.value()
        min_dist = self.doubleSpinBox_3.value()
        max_dist = self.doubleSpinBox_4.value()
        client = self.comboBox.currentText()
        model = self.comboBox_2.currentText()
        
        catalog = utils.get_catalog(starttime, endtime, client=client, min_magnitude=min_mag)
        arrivals = utils.taup_arrival_times(catalog, station_metadata_path, earth_model=model,
                                            min_distance_degrees=min_dist,
                                            max_distance_degrees=max_dist)
        pickle.dump(arrivals, open(event_metadata_output_path, "wb"))
        
        """catalog = utils.get_catalog(starttime, endtime, client=client, min_magnitude=min_mag)
        #data_map = utils.data_map(data_path) <- sin implementar
        arrivals = utils.taup_arrival_times(catalog, station_metadata_path, earth_model=model,
                                            min_distance_degrees=min_dist,
                                            max_distance_degrees=max_dist)
        pickle.dump(arrivals, open(event_metadata_output_path, "wb"))
        utils.cut_earthquakes(data_map, arrivals, station_metadata_path)"""