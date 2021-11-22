# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 04:06:28 2020

@author: olivar

This file is part of Rfun, a toolbox for the analysis of teleseismic receiver
functions.
Copyright (C) 2020-2021 Andrés Olivar-Castaño

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

For questions, bug reports, or to suggest new features, please contact me at
olivar.ac@gmail.com.
"""

import io
import os
import sys
from PyQt5 import uic, QtGui, QtCore, QtWidgets
import obspy
import pickle
from functools import partial
import isp.receiverfunctions.rf_dialogs_utils as du
from isp.Gui.Frames import UiReceiverFunctionsCut, UiReceiverFunctionsSaveFigure, \
                           UiReceiverFunctionsCrossSection, UiReceiverFunctionsAbout, \
                           UiReceiverFunctionsShowEarthquake, BaseFrame
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from isp.Gui import pyqt, pqg, pw, pyc, qt

import numpy as np
from sys import platform

class ShowEarthquakeDialog(QtWidgets.QDialog, UiReceiverFunctionsShowEarthquake):
    def __init__(self, file, bandpass):
        super(ShowEarthquakeDialog, self).__init__()
        self.setupUi(self)
        
        # Button connections
        self.pushButton_2.clicked.connect(self.close)
        
        self.mplwidget.figure.subplots(nrows=3)
        st = obspy.read(file)
        st.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1])
        
        try:
            l = st.select(component="L")
            maxy = np.max(l[0].data)
            miny = np.min(l[0].data)
        except IndexError: # If this happens there is no L component, so there should be a Z component
            z = st.select(component="Z")
            maxy = np.max(z[0].data)
            miny = np.min(z[0].data)
        
        for i, tr in enumerate(st):
            self.mplwidget.figure.axes[i].plot(tr.times("matplotlib"), tr.data, color="black")
            self.mplwidget.figure.axes[i].set_ylim(miny, maxy)
    
    def close(self):
        self.done(0)

class CutEarthquakesDialog(QtWidgets.QDialog, UiReceiverFunctionsCut):
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
            if "darwin" == platform:
                path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', '')
            else:
                path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', '',
                                                           pw.QFileDialog.DontUseNativeDialog)
            self.lineEdit.setText(path)
        elif pushButton == 3:
            path = QtWidgets.QFileDialog.getOpenFileName()[0]
            self.lineEdit_3.setText(path)
        elif pushButton == 4:
            if "darwin" == platform:
                path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', '')
            else:
                path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', '',
                                                           pw.QFileDialog.DontUseNativeDialog)
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
        
        catalog = du.get_catalog(starttime, endtime, client=client, min_magnitude=min_mag)
        arrivals = du.taup_arrival_times(catalog, station_metadata_path, earth_model=model,
                                            min_distance_degrees=min_dist,
                                            max_distance_degrees=max_dist)
        pickle.dump(arrivals, open(event_metadata_output_path, "wb"))
        data_map = du.map_data(data_path, quick=self.checkBox.isChecked())
        
        time_before = self.doubleSpinBox_5.value()
        time_after = self.doubleSpinBox_6.value()
        rotation = self.comboBox_3.currentText()
        remove_instrumental_responses = self.checkBox_2.isChecked()

        du.cut_earthquakes(data_map, arrivals, time_before, time_after, min_snr,
                    station_metadata_path, earthquake_output_path)

class SaveFigureDialog(QtWidgets.QDialog, UiReceiverFunctionsSaveFigure):
    def __init__(self, figure):
        super(SaveFigureDialog, self).__init__()
        self.setupUi(self)
        
        self.figure = figure
        
        self.pushButton_2.clicked.connect(self.save_figure)
    
    def save_figure(self):
        output_path = QtWidgets.QFileDialog.getSaveFileName()[0]
        
        # Apply settings
        self.figure.set_figheight(self.doubleSpinBox.value())
        self.figure.set_figwidth(self.doubleSpinBox_2.value())

        self.figure.suptitle(self.lineEdit.text())
        self.figure.axes[-1].set_xlabel(self.lineEdit_2.text())
        self.figure.axes[-1].set_ylabel(self.lineEdit_3.text())
        
        self.figure.subplots_adjust(left=self.doubleSpinBox_3.value(), bottom=self.doubleSpinBox_6.value(),
                                    right=self.doubleSpinBox_4.value(), top=self.doubleSpinBox_5.value())
        
        format_ = self.comboBox.currentText()         
        self.figure.savefig(output_path + format_, dpi=self.spinBox.value(), format=format_[1:])

class CrossSectionDialog(QtWidgets.QDialog, UiReceiverFunctionsCrossSection):
    def __init__(self, x, y, z, start, end):
        super(CrossSectionDialog, self).__init__()
        self.setupUi(self)
        
        self.start = start
        self.end = end
        self.x = x
        self.y = y
        self.z = z
        
        self.mplwidget.figure.subplots(1)
        im = self.mplwidget.figure.axes[0].pcolormesh(x, y, z, vmin=-1., vmax=1, cmap="RdBu_r")
        
        divider = make_axes_locatable(self.mplwidget.figure.axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        self.mplwidget.figure.axes[0].set_xlabel("Distance (km)")
        self.mplwidget.figure.axes[0].set_ylabel("Depth (km)")
        
        self.pushButton.clicked.connect(self.save_cross_section)
        self.pushButton_2.clicked.connect(self.save_figure)
        self.pushButton_3.clicked.connect(self.close)
    
    def save_cross_section(self):
        fname = QtWidgets.QFileDialog.getSaveFileName()[0]
        
        css_dict = {"start":self.start,
                    "end":self.end,
                    "distance":self.x,
                    "depth":self.y,
                    "cross_section":self.z}

        if fname:
            pickle.dump(css_dict, open(fname, "wb"))
    
    def save_figure(self):
        buffer = io.BytesIO()
        pickle.dump(self.mplwidget.figure, buffer)
        # Point to the first byte of the buffer and read it
        buffer.seek(0)
        fig_copy = pickle.load(buffer)
        
        #We also need a new canvas manager
        newfig = plt.figure()
        newmanager = newfig.canvas.manager
        newmanager.canvas.figure = fig_copy
        fig_copy.set_canvas(newmanager.canvas)        
        
        """dialog = SaveFigureDialog(fig_copy, preferred_size, preferred_margins, preferred_title,
                 preferred_xlabel, preferred_ylabel, preferred_fname)
        dialog.exec_()"""
    
    def close(self):
        self.done(0)

class AboutDialog(QtWidgets.QDialog, UiReceiverFunctionsAbout):
    def __init__(self):
        super(AboutDialog, self).__init__()
        self.setupUi(self)