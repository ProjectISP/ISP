# -*- coding: utf-8 -*-
"""
This file is part of Rfun, a toolbox for the analysis of teleseismic receiver
functions.

Copyright (C) 2020-2025 Andrés Olivar-Castaño

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

import os
from PyQt5 import uic, QtWidgets
import obspy
import pickle
from pathlib import Path
from functools import partial
import isp.receiverfunctions.rfun_dialogs_utils as du
import isp.receiverfunctions.rfun_main_window_utils as mwu

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from isp.Gui.Frames import UiReceiverFunctionsCut, \
                           UiReceiverFunctionsCrossSection, UiReceiverFunctionsAbout, \
                           UiReceiverFunctionsShowEarthquake, UiReceiverFunctionsPreferences

from isp.receiverfunctions.definitions import ROOT_DIR, CONFIG_PATH

class ShowEarthquakeDialog(QtWidgets.QDialog, UiReceiverFunctionsShowEarthquake):
    def __init__(self, hdf5_file, stnm, event_id, settings, rotation,
                 rf_freq_filter_params):
        super(ShowEarthquakeDialog, self).__init__()
        self.setupUi(self)

        arr = hdf5_file[stnm][event_id][:]
        z = arr[np.where(hdf5_file[stnm][event_id].attrs["data_structure"] == "Z")[0][0]]
        e = arr[np.where(hdf5_file[stnm][event_id].attrs["data_structure"] == "E")[0][0]]
        n = arr[np.where(hdf5_file[stnm][event_id].attrs["data_structure"] == "N")[0][0]]
        time = arr[np.where(hdf5_file[stnm][event_id].attrs["data_structure"] == "time")[0][0]]
        baz = hdf5_file[stnm][event_id].attrs["baz"]
        snr = hdf5_file[stnm][event_id].attrs["SNR"]
        inc = hdf5_file[stnm][event_id].attrs["incident_angle"]

        if rotation == "LQT":
            l, q, t = obspy.signal.rotate.rotate_zne_lqt(z, n, e, baz, inc)
            components = [l, q, t]
            labels = ["L", "Q", "T"]
        elif rotation == "ZRT":
            r, t = obspy.signal.rotate.rotate_ne_rt(n, e, baz)
            components = [z, r, t]
            labels = ["Z", "R", "T"]
               
        # Compute the receiver functions
        rfs = []
        for dcmp in ["Z", "R", "T"]:
            rf = mwu.compute_rfs(stnm, hdf5_file,
                                 settings['rfs']['computation_method']['method'],
                                 settings['rfs']['computation_method']['method_settings'],
                                 normalize=False,
                                 w0=settings['rfs']['general_settings']['w0'],
                                 time_shift=settings['rfs']['general_settings']['time_shift'],
                                 rotation=rotation,
                                 component=dcmp,
                                 event_ids=[event_id])
            rfs.append(rf[0][0])
        rftime = rf[0][1]
        
        rfs = np.array(rfs)
        components = np.array(components)
        
        if rf_freq_filter_params is not None:
            filter_, args, df, order = rf_freq_filter_params
            for i in range(len(rfs)):
                rfs[i] = filter_(rfs[i], *args, df, corners=order, zerophase=True)
                components[i] = filter_(components[i], *args, df, corners=order, zerophase=True)
       
        for rf in rfs:
            rf /= np.max(np.abs(rfs[1]))
        
        norm_val = max([np.max(np.abs(x)) for x in components])
        # Button connections
        self.pushButton_2.clicked.connect(self.close)
        
        # Set up figure
        self.mplwidget.figure.subplots(nrows=3, ncols=2)
        
        j = 0
        k = 1
        for i, tr in enumerate(components):
            self.mplwidget.figure.axes[i+j].text(0.01, 0.80, labels[i], weight="bold", transform=self.mplwidget.figure.axes[i+j].transAxes)
            self.mplwidget.figure.axes[i+j].plot(time, tr, color="black")
            self.mplwidget.figure.axes[i+j].set_xlim(np.min(time), np.max(time))
            j += 1
            
            self.mplwidget.figure.axes[i+k].text(0.01, 0.80, labels[i], weight="bold", transform=self.mplwidget.figure.axes[i+k].transAxes)
            self.mplwidget.figure.axes[i+k].fill_between(rftime, rfs[i], color=settings['rfs']['appearance']['line_color'],
                                                       linewidth=settings['rfs']['appearance']['line_width'])            
            self.mplwidget.figure.axes[i+k].fill_between(rftime, np.zeros(len(rftime)),
                                                        rfs[i], where=(rfs[i] > 0), color=settings['rfs']['appearance']['positive_fill_color'])
            self.mplwidget.figure.axes[i+k].fill_between(rftime, np.zeros(len(rftime)),
                                                        rfs[i], where=(rfs[i] < 0), color=settings['rfs']['appearance']['negative_fill_color'])
            self.mplwidget.figure.axes[i+k].set_xlim(np.min(rftime), np.max(rftime))
            k += 1
            
        # self.mplwidget.figure.title()
        self.mplwidget.figure.axes[3].set_ylim(-1, 1)
        self.mplwidget.figure.axes[5].set_ylim(-1, 1)
        
        self.mplwidget.figure.axes[0].set_title("Waveforms")
        self.mplwidget.figure.axes[1].set_title("Receiver functions")
        self.mplwidget.figure.axes[2].set_ylabel("Normalized amplitude")
        self.mplwidget.figure.axes[3].set_ylabel("Normalized amplitude to R/Q")
        self.mplwidget.figure.axes[4].set_xlabel("Time (s)")
        self.mplwidget.figure.axes[5].set_xlabel("Time (s)")
    
    def close(self):
        self.done(0)

class CutEarthquakesDialog(QtWidgets.QDialog, UiReceiverFunctionsCut):
    def __init__(self):
        super(CutEarthquakesDialog, self).__init__()
        self.setupUi(self)

        # connections
        self.pushButton_2.clicked.connect(partial(self.get_path, 2))
        self.pushButton_3.clicked.connect(partial(self.get_path, 3))
        self.pushButton_5.clicked.connect(partial(self.get_path, 5))        
        self.pushButton_9.clicked.connect(self.cut_earthquakes)
        self.pushButton_10.clicked.connect(self.close)
        
        # Add the custom model files to the corresponding combobox
        tvel_files = Path(os.path.join(ROOT_DIR, "earth_models")).rglob("*.tvel")
        self.custom_earth_models = []
        for tvel in tvel_files:
            model_name = str(tvel).split(os.sep)[-1].strip(".tvel")
            self.comboBox_4.addItem(model_name)
            self.custom_earth_models.append(model_name)
    
    def get_path(self, pushButton):
        if pushButton == 2:
            path = QtWidgets.QFileDialog.getExistingDirectory()
            if path:
                self.lineEdit.setText(path)
        elif pushButton == 3:
            path = QtWidgets.QFileDialog.getOpenFileName()[0]
            if path:
                self.lineEdit_3.setText(path)
        elif pushButton == 5:
            path = QtWidgets.QFileDialog.getSaveFileName()[0]
            if path:
                self.lineEdit_4.setText(path)

    def cut_earthquakes(self):
        data_path = self.lineEdit.text()
        format_ = self.comboBox_4.currentText()
        stationxml = self.lineEdit_3.text()
        output_dir = self.lineEdit_4.text()
        starttime = self.dateTimeEdit.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z")
        endtime = self.dateTimeEdit_2.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z")
        min_mag = self.doubleSpinBox_2.value()
        min_snr = self.doubleSpinBox.value()
        min_dist = self.doubleSpinBox_3.value()
        max_dist = self.doubleSpinBox_4.value()
        client = self.comboBox.currentText()
        model = self.comboBox_4.currentText()
        min_depth = self.doubleSpinBox_7.value()
        max_depth = self.doubleSpinBox_8.value()
        phase = self.comboBox_5.currentText()
        noise_wlen = self.doubleSpinBox_10.value()
        noise_start = self.doubleSpinBox_9.value()
        
        time_before = self.doubleSpinBox_5.value()
        time_after = self.doubleSpinBox_6.value()
        
        catalog = du.get_catalog(starttime, endtime, client=client, min_magnitude=min_mag)
        data_map = du.map_data(data_path, format_=format_)
        du.cut_earthquakes(data_map, catalog, time_before, time_after, min_snr,
                            min_mag, min_dist, max_dist, min_depth, max_depth,
                            stationxml, output_dir, model, self.custom_earth_models,
                            noise_wlen=noise_wlen, noise_before_P=noise_start,
                            pre_filt=[1/200, 1/100, 45, 50],
                            phase=phase)
        
        
        
# class SaveFigureDialog(QtWidgets.QDialog):
#     def __init__(self, figure, preferred_size, preferred_margins, preferred_title,
#                  preferred_xlabel, preferred_ylabel, preferred_fname):
#         super(SaveFigureDialog, self).__init__()
#         uic.loadUi(os.path.join(ROOT_DIR, 'ui/RfunDialogsSaveFigure.ui'), self)
        
#         self.figure = figure
        
#         self.pushButton_2.clicked.connect(self.save_figure)
        
#         self.doubleSpinBox.setValue(preferred_size[0])
#         self.doubleSpinBox_2.setValue(preferred_size[1])
#         self.doubleSpinBox_3.setValue(preferred_margins[0])
#         self.doubleSpinBox_4.setValue(preferred_margins[1])
#         self.doubleSpinBox_6.setValue(preferred_margins[2])
#         self.doubleSpinBox_5.setValue(preferred_margins[3])
#         self.lineEdit.setText(preferred_title)
#         self.lineEdit_2.setText(preferred_xlabel)
#         self.lineEdit_3.setText(preferred_ylabel)
        
#         self.preferred_fname = preferred_fname
    
#     def save_figure(self):
#         output_path = QtWidgets.QFileDialog.getSaveFileName(directory=self.preferred_fname)[0]
        
#         if output_path:
        
#             # Apply settings
#             self.figure.set_size_inches(self.doubleSpinBox_2.value(), self.doubleSpinBox.value())
#             self.figure.suptitle(self.lineEdit.text())
#             self.figure.axes[-1].set_xlabel(self.lineEdit_2.text())
#             self.figure.axes[-1].set_ylabel(self.lineEdit_3.text())
#             self.figure.subplots_adjust(left=self.doubleSpinBox_3.value(), bottom=self.doubleSpinBox_6.value(),
#                                         right=self.doubleSpinBox_4.value(), top=self.doubleSpinBox_5.value())
            
#             format_ = self.comboBox.currentText()         
#             self.figure.savefig(output_path + format_, dpi=self.spinBox.value(), format=format_[1:])

import matplotlib.colors as colors
class CrossSectionDialog(QtWidgets.QDialog, UiReceiverFunctionsCrossSection):
    def __init__(self, x, y, z, start, end):
        super(CrossSectionDialog, self).__init__()
        self.setupUi(self)
        
        self.start = start
        self.end = end
        self.x = x
        self.y = y
        self.z = z
        
        z[np.isnan(z)] == 0
        
        # norm = colors.TwoSlopeNorm(vmin=np.min(z), vcenter=0, vmax=np.max(z))
        # # define the colormap
        # cmap = plt.get_cmap('seismic')
        
        # # extract all colors from the .jet map
        # cmaplist = [cmap(i) for i in range(cmap.N)]
        # # create the new map
        # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        
        # # define the bins and normalize and forcing 0 to be part of the colorbar!
        # bounds = np.linspace(np.min(z),np.max(z),100)
        # idx=np.searchsorted(bounds,0)
        # bounds=np.insert(bounds,idx,0)
        # norm = BoundaryNorm(bounds, cmap.N)
        
        self.mplwidget.figure.subplots(1)
        im = self.mplwidget.figure.axes[0].pcolormesh(x, y, z[:len(y)-1,:][:,:len(x)-1], norm=colors.TwoSlopeNorm(vcenter=0), cmap="seismic")
        # self.mplwidget.figure.axes[0].axhline(-35)
        
        divider = make_axes_locatable(self.mplwidget.figure.axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.mplwidget.figure.colorbar(im, cax=cax)
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
        fname = QtWidgets.QFileDialog.getSaveFileName(None, "Save figure", "", "PNG (*.png)")[0]
        if fname:
            self.mplwidget.figure.savefig(fname, dpi=600)
            
    def close(self):
        self.done(0)

class PreferencesDialog(QtWidgets.QDialog, UiReceiverFunctionsPreferences):
    def __init__(self):
        super(PreferencesDialog, self).__init__()
        self.setupUi(self)
        
        self.pushButton_2.clicked.connect(self.save_settings)
        self.pushButton_3.clicked.connect(self.close)
        self.pushButton.clicked.connect(self.reset_to_defaults)
        
        self.settings = mwu.read_preferences()
        self.read_settings()
    
    def read_settings(self):
        
        # CCP stack settings
        # Appearance
        self.checkBox_6.setChecked(self.settings['ccp']['appearance']['include_stations'])
        self.comboBox_6.setCurrentIndex(self.comboBox_6.findText(self.settings['ccp']['appearance']['plotting_method']))
        self.comboBox_5.setCurrentIndex(self.comboBox_5.findText(self.settings['ccp']['appearance']['colormap']))
        self.lineEdit_14.setText(self.settings['ccp']['appearance']['station_marker'])
        self.lineEdit_13.setText(self.settings['ccp']['appearance']['station_marker_color'])
        # Shapefiles
        self.checkBox_5.setChecked(self.settings['ccp']['shapefiles']['include'])
        self.lineEdit_12.setText(self.settings['ccp']['shapefiles']['path'])
        # Computation
        self.comboBox_8.setCurrentIndex(self.comboBox_8.findText(self.settings['ccp']['computation']['stacking_method']))
        
        # RFS settings
        # Appearance
        self.lineEdit.setText(self.settings['rfs']['appearance']['line_color'])
        self.doubleSpinBox.setValue(self.settings['rfs']['appearance']['line_width'])
        self.lineEdit_3.setText(self.settings['rfs']['appearance']['positive_fill_color'])
        self.lineEdit_2.setText(self.settings['rfs']['appearance']['negative_fill_color'])
        # General Settings
        self.checkBox.setChecked(self.settings['rfs']['general_settings']['normalize'])
        self.doubleSpinBox_2.setValue(self.settings['rfs']['general_settings']['w0'])
        self.doubleSpinBox_3.setValue(self.settings['rfs']['general_settings']['time_shift'])
        # Computation Method
        if self.settings['rfs']['computation_method']['method'] == 'Waterlevel':
            self.radioButton.setChecked(True)
        elif self.settings['rfs']['computation_method']['method'] == 'Time-domain':
            self.radioButton_2.setChecked(True)
            self.spinBox_3.setValue(self.settings['rfs']['computation_method']['method_settings']['max_iters'])
            self.doubleSpinBox_4.setValue(self.settings['rfs']['computation_method']['method_settings']['min_deltaE'])
        # Stacking
        self.doubleSpinBox_8.setValue(self.settings['rfs']['stacking']['ref_slowness'])
        
        # HK settings
        # Appearance
        self.comboBox.setCurrentIndex(self.comboBox.findText(self.settings['hk']['appearance']['plotting_method']))
        self.comboBox_2.setCurrentIndex(self.comboBox_2.findText(self.settings['hk']['appearance']['colormap']))
        self.lineEdit_7.setText(self.settings['hk']['appearance']['line_color'])
        self.lineEdit_8.setText(self.settings['hk']['appearance']['ser_color'])
        # Computation
        self.checkBox_2.setChecked(self.settings['hk']['computation']['semblance_weighting'])
        self.spinBox.setValue(self.settings['hk']['computation']['H_points'])
        self.spinBox_2.setValue(self.settings['hk']['computation']['k_points'])
        self.doubleSpinBox_5.setValue(self.settings['hk']['computation']['avg_vp'])
        # Theoretical arrival times
        self.doubleSpinBox_6.setValue(self.settings['hk']['theoretical_atimes']['ref_slowness'])
        self.doubleSpinBox_7.setValue(self.settings['hk']['theoretical_atimes']['avg_vp'])

        # Crustal thickness map
        # Appearance
        self.checkBox_3.setChecked(self.settings['map']['appearance']['include_stations'])
        self.comboBox_3.setCurrentIndex(self.comboBox_3.findText(self.settings['map']['appearance']['plotting_method']))
        self.comboBox_4.setCurrentIndex(self.comboBox_4.findText(self.settings['map']['appearance']['colormap']))
        self.lineEdit_10.setText(self.settings['map']['appearance']['station_marker'])
        self.lineEdit_9.setText(self.settings['map']['appearance']['station_marker_color'])
        # Shapefiles
        self.checkBox_4.setChecked(self.settings['map']['shapefiles']['include'])
        self.lineEdit_11.setText(self.settings['map']['shapefiles']['path'])

    def close(self):
        self.done(0)
        
    def save_settings(self):
        
        # Check which RF computation method is selected
        method_radiobuttons = [elem for elem in self.groupBox_16.children() if isinstance(elem, QtWidgets.QRadioButton)]
        for radiobutton in method_radiobuttons:
            if radiobutton.isChecked():
                method = radiobutton.text()
                break
        
        if method == 'Waterlevel':
            method_settings = {}        
        elif method == "Time-domain":
            method_settings = {'max_iters': self.spinBox_3.value(),
                               'min_deltaE': self.doubleSpinBox_4.value()}

        settings = {'ccp':{'appearance':{'include_stations':self.checkBox_6.isChecked(),
                                         'plotting_method':self.comboBox_6.currentText(),
                                         'colormap':self.comboBox_5.currentText(),
                                         'station_marker':self.lineEdit_14.text(),
                                         'station_marker_color':self.lineEdit_13.text()},
                           'shapefiles':{'include':self.checkBox_5.isChecked(),
                                         'path':self.lineEdit_12.text()},
                           'computation':{#'earth_model':self.comboBox_7.currentText(), currently not in use
                                          'stacking_method':self.comboBox_8.currentText()}},
                    'rfs':{'appearance':{'line_color':self.lineEdit.text(),
                                         'line_width':self.doubleSpinBox.value(),
                                         'positive_fill_color':self.lineEdit_3.text(),
                                         'negative_fill_color':self.lineEdit_2.text()},
                           'general_settings':{'normalize':self.checkBox.isChecked(),
                                            'w0':self.doubleSpinBox_2.value(),
                                            'time_shift':self.doubleSpinBox_3.value()},
                           'computation_method':{'method':method,
                                                 'method_settings':method_settings},
                           'stacking':{'ref_slowness':self.doubleSpinBox_8.value()}},
                    'hk':{'appearance':{'plotting_method':self.comboBox.currentText(),
                                        'colormap':self.comboBox_2.currentText(),
                                        'line_color':self.lineEdit_7.text(),
                                        'ser_color':self.lineEdit_8.text()},
                          'computation':{'semblance_weighting':self.checkBox_2.isChecked(),
                                         'H_points':self.spinBox.value(),
                                         'k_points':self.spinBox_2.value(),
                                         'avg_vp':self.doubleSpinBox_5.value()},
                          'theoretical_atimes':{'ref_slowness':self.doubleSpinBox_6.value(),
                                                'avg_vp':self.doubleSpinBox_7.value()}},
                    'map':{'appearance':{'include_stations':self.checkBox_3.isChecked(),
                                         'plotting_method':self.comboBox_3.currentText(),
                                         'colormap':self.comboBox_4.currentText(),
                                         'station_marker':self.lineEdit_10.text(),
                                         'station_marker_color':self.lineEdit_9.text()},
                           'shapefiles':{'include':self.checkBox_4.isChecked(),
                                         'path':self.lineEdit_11.text()}}}
        
        pickle.dump(settings, open(CONFIG_PATH, 'wb'))
    
    def reset_to_defaults(self):
        qm = QtWidgets.QMessageBox
        ret = qm.question(self,'', "Are you sure to reset all the values?", qm.Yes | qm.No)
        if ret == qm.Yes:
            self.settings = mwu.read_preferences(return_defaults=True)
            self.read_settings()

class AboutDialog(QtWidgets.QDialog, UiReceiverFunctionsAbout):
    def __init__(self):
        super(AboutDialog, self).__init__()
        self.setupUi(self)