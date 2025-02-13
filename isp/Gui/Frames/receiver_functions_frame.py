# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:45:54 2020

@author: olivar

Rfun, a toolbox for the analysis of teleseismic receiver functions
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
# isp imports
from isp.Gui.Frames import BaseFrame
import isp.receiverfunctions.rfun_dialogs as dialogs
import isp.receiverfunctions.rfun_main_window_utils as mwu
from isp.Gui.Frames.uis_frames import UiReceiverFunctions
# other imports
import os
import math
import pickle
import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.backend_bases
import matplotlib.patches as mpatches
from PyQt5 import QtWidgets
import obspy.signal.filter
import copy
import scipy.interpolate as scint
from pathlib import Path
from isp.receiverfunctions.definitions import ROOT_DIR, CONFIG_PATH

class RecfFrame(BaseFrame, UiReceiverFunctions):
    
    def __init__(self):
        super(RecfFrame, self).__init__()
        self.setupUi(self)
        
        self.settings = mwu.read_preferences()
        
        # This should be maybe changed in the UI directly
        self.actionRead_waveforms.setEnabled(True)
        self.actionClose_HDF5_file.setEnabled(False)
        
        # Set up RF analysis UI based on the chosen computation method
        # in self.settings
        self.choose_rf_analysis_widget(self.settings['rfs']['computation_method']['method'])
        
        # RF analysis-related attributes
        self.hdf5_waveforms = None
        self.rfs = []
        self.rf_current_page = 1
        self.rf_pages = 1
    
        self.first_rf_stack_plot = True # To be removed
        self.first_hk_stack_plot = True # To be removed
        self.hk_result = {}
    
        # RF computation progress bar and label
        self.RFs_are_computed = False
        p2 = self.progressBar_2.sizePolicy()
        p2.setRetainSizeWhenHidden(True)
        self.progressBar_2.setSizePolicy(p2)
        self.progressBar_2.setVisible(False)
    
        # p3 = self.label_49.sizePolicy()
        # p3.setRetainSizeWhenHidden(True)
        # self.label_49.setSizePolicy(p3)
        # self.label_49.setVisible(False)
        self.label_49.setText("")
        
        # Theoretical arrival times obtained from the maximum of the H-k
        self.Ps = 0
        self.PpPs = 0
        self.PpSs_PsPs = 0
        
        # CCP stacking-related attributes
        self.ccp_stack = None
        self.ccp_x = None
        self.ccp_y = None
        self.ccp_z = None
        
        # Linear RF stacking-related attributes
        self.linear_RF_stack = None
        
        self.ccp_grid = {'x0':None, 'y0':None, 'x1':None, 'y1':None}
        self.ccp_cross_section = {'x0':None, 'y0':None, 'x1':None, 'y1':None}
        self.ccp_stack_map_plot_mode = None
        self.button_pressed = False
        self.ccp_grid_mpl_line = None
        self.ccp_cross_section_mpl_line = None
        self.rfs_hdf5 = None
        self.istack = None
        self.depth_array = None
        self.first_ccp_stack_plot = True
        self.mplwidget_5_basemap = None
        self.mplwidget_5_gridlines = None
        self.mplwidget_5_ccp_pcolormesh = None
        
        p = self.progressBar.sizePolicy()
        p.setRetainSizeWhenHidden(True)
        self.progressBar.setSizePolicy(p)
        self.progressBar.setVisible(False)
        
        # Connect GUI elements
        self.connect_rf_analysis_gui_elements()
        self.connect_ccp_stack_gui_elements()
        self.connect_hk_map_gui_elements()
        
        # Menu actions
        self.actionAbout.triggered.connect(self.about_dialog)
        
        # RF Analysis toolbox
        self.analysis_tools_groupbox_iscollapsed = True
        self.rf_freq_filter_isapplied = False
        self.rf_freq_filter_params = None
       
    def choose_rf_analysis_widget(self, widget):
        if widget == "Waterlevel":
            self.groupBox_4.setTitle("Waterlevel deconvolution")
            self.label_4.setVisible(True)
            self.doubleSpinBox_2.setVisible(True)
        elif widget == "Time-domain":
            self.groupBox_4.setTitle("Iterative time-domain deconvolution")
            self.label_4.setVisible(False)
            self.doubleSpinBox_2.setVisible(False)
    
    def populate_component_combobox(self):
        self.comboBox_10.clear()
        if self.comboBox_9.currentText() == "LQT":
            for cmpn in ["Q", "T", "L"]:
                self.comboBox_10.addItem(cmpn)
        elif self.comboBox_9.currentText() == "ZRT":
            for cmpn in ["R", "T", "Z"]:
                self.comboBox_10.addItem(cmpn)
    
    def connect_rf_analysis_gui_elements(self):
        # Menu actions
        self.actionRead_waveforms.triggered.connect(self.read_waveforms)
        self.actionClose_HDF5_file.triggered.connect(self.close_waveforms)
        self.actionCut_earthquakes_from_raw_data.triggered.connect(self.cut_earthquakes_dialog)
        self.actionPreferences.triggered.connect(self.preferences_dialog)
        
        # Pushbuttons
        self.pushButton.clicked.connect(self.compute_rfs)
        self.pushButton_2.clicked.connect(self.save_rfs)
        self.pushButton_7.clicked.connect(self.previous_rfs_page)
        self.pushButton_8.clicked.connect(self.next_rfs_page)
        self.pushButton_3.clicked.connect(self.plot_rf_stack)
        self.pushButton_5.clicked.connect(self.plot_hk_stack)
        self.pushButton_9.clicked.connect(self.plot_map)
        self.pushButton_4.clicked.connect(partial(self.save_figure, self.mplwidget_2.figure))
        self.pushButton_15.clicked.connect(partial(self.save_figure, self.mplwidget_3.figure))
        self.pushButton_10.clicked.connect(partial(self.save_figure, self.mplwidget_4.figure))
        self.pushButton_6.clicked.connect(self.save_hk_result)
        self.pushButton_19.clicked.connect(self.save_linear_rf_stack)
        
        self.pushButton_17.clicked.connect(self.auto_reject_rfs)
        self.pushButton_16.clicked.connect(self.remove_rejections)
        # Combobox changes
        self.comboBox.currentTextChanged.connect(self.reset_rf_analysis_gui)
        self.comboBox_3.currentTextChanged.connect(self.plot_rfs)
        self.spinBox_3.valueChanged.connect(self.number_of_rfs_per_page_changed)
        # self.spinBox_7.valueChanged.connect(self.rfs_xlim_changed)
        # self.spinBox_8.valueChanged.connect(self.rfs_xlim_changed)
        # mplwidgets
        self.mplwidget.figure.canvas.mpl_connect('button_press_event', self.recf_plot_clicked)
        
        # Analysis tools
        # self.groupBox_10.setVisible(False)
        # self.toolButton.clicked.connect(self.collapse_analysis_options)
        # self.toolButton.setIcon(QtGui.QIcon("rfun/resources/arrow_right.png"))
        self.pushButton_21.clicked.connect(self.filter_receiver_functions)
        self.pushButton_22.clicked.connect(self.filter_receiver_functions)
        self.pushButton_21.setEnabled(False)
        self.comboBox_9.currentTextChanged.connect(self.populate_component_combobox)
    
    def connect_hk_map_gui_elements(self):
        self.actionRead_H_k_results_file.triggered.connect(self.read_hk_results)
        self.pushButton_18.clicked.connect(self.plot_hk_results)
        
        for doubleSpinBox in [self.doubleSpinBox_33, self.doubleSpinBox_34,
                              self.doubleSpinBox_35, self.doubleSpinBox_36]:
            
            doubleSpinBox.valueChanged.connect(self.update_hk_map)
        
        self.comboBox_7.currentTextChanged.connect(self.hk_basemap)
        self.doubleSpinBox_38.valueChanged.connect(self.hk_gridlines)
        self.pushButton_20.clicked.connect(partial(self.save_figure, self.mplwidget_6.figure))
        
    def connect_ccp_stack_gui_elements(self):
        # Menu actions
        self.actionRead_RFs.triggered.connect(self.ccp_stack_read_rfs)
        self.actionRead_CCP_Stack.triggered.connect(self.read_ccp_stack)
        # mplwidget
        self.mplwidget_5.figure.canvas.mpl_connect('button_press_event', self.ccp_stack_map_event_handler)
        self.mplwidget_5.figure.canvas.mpl_connect('button_release_event', self.ccp_stack_map_event_handler)
        self.mplwidget_5.figure.canvas.mpl_connect('motion_notify_event', self.ccp_stack_map_event_handler)
        # Pushbuttons
        self.pushButton_11.clicked.connect(partial(self.ccp_stack_map_toggle_plot_mode, "grid"))
        self.pushButton_25.clicked.connect(partial(self.ccp_stack_map_toggle_plot_mode, "cross_section"))
        self.pushButton_12.clicked.connect(self.compute_ccp_stack)
        self.pushButton_14.clicked.connect(self.save_ccp_stack)
        self.pushButton_13.clicked.connect(partial(self.save_figure, self.mplwidget_5.figure))
        self.pushButton_26.clicked.connect(self.cross_section_dialog)
    
        self.doubleSpinBox_21.valueChanged.connect(self.update_ccp_map)
        self.doubleSpinBox_22.valueChanged.connect(self.update_ccp_map)
        self.doubleSpinBox_23.valueChanged.connect(self.update_ccp_map)
        self.doubleSpinBox_24.valueChanged.connect(self.update_ccp_map)
        self.doubleSpinBox_25.valueChanged.connect(self.plot_ccp_stack)
        
        self.doubleSpinBox_26.valueChanged.connect(self.ccp_gridlines)
        
        self.comboBox_6.currentTextChanged.connect(self.ccp_basemap)
        self.spinBox_5.valueChanged.connect(self.plot_ccp_stack)
    
        self.spinBox_6.valueChanged.connect(self.ccp_basemap)
        
        # Add the custom model files to the CCP Earth model combobox
        tvel_files = Path(os.path.join(ROOT_DIR, "earth_models")).rglob("*.tvel")
        for tvel in tvel_files:
            model_name = str(tvel).split(os.sep)[-1].strip(".tvel")
            self.comboBox_11.addItem(model_name)
    
    def preferences_dialog(self):
        dialog = dialogs.PreferencesDialog()
        dialog.exec_()
        
        self.settings = mwu.read_preferences()
        self.choose_rf_analysis_widget(self.settings['rfs']['computation_method']['method'])
    
    def read_waveforms(self):
        """Map the mseed files inside the given directory, return a dict
        
        """
        self.comboBox.clear()
        
        dir_ = QtWidgets.QFileDialog.getOpenFileName()[0]
        
        if dir_:
            self.hdf5_waveforms = mwu.read_hdf5(dir_)
            
            # Populate the station combobox
            for stnm in sorted(list(self.hdf5_waveforms.keys())):
                if len(self.hdf5_waveforms[stnm].keys()) > 0:
                    self.comboBox.addItem(stnm)
        
            self.actionClose_HDF5_file.setEnabled(True)
    
    def close_waveforms(self):
        self.comboBox.clear() # This also resets the GUI as it's connected to the reset_rf_analysis_gui function
        self.hdf5_waveforms.close()
        self.actionClose_HDF5_file.setEnabled(False)
    
    def compute_rfs(self):
        """Compute and plot selected receiver functions
        
        """
        self.rf_current_page = 1
        stnm = self.comboBox.currentText()
        a = self.doubleSpinBox.value()
        self.settings['rfs']['computation_method']['method_settings']['gaussian_filter_width'] = a
        if self.settings['rfs']['computation_method']['method'] == 'Waterlevel':
            c = self.doubleSpinBox_2.value()
            self.settings['rfs']['computation_method']['method_settings']['waterlevel_parameter'] = c
        
        # Preserve rejections of RFs through changes in the processing parameters
        rejections = [(x[5], x[6]) for x in self.rfs] # pos 5 is accept/reject, pos 6 is event ID
        
              
        # Compute new rfs
        self.label_49.setText("Calculating RFs:")
        self.progressBar_2.setVisible(True)
        
        self.rfs = mwu.compute_rfs(stnm, self.hdf5_waveforms,
                                     self.settings['rfs']['computation_method']['method'],
                                     self.settings['rfs']['computation_method']['method_settings'],
                                     normalize=self.settings['rfs']['general_settings']['normalize'],
                                     w0=self.settings['rfs']['general_settings']['w0'],
                                     time_shift=self.settings['rfs']['general_settings']['time_shift'],
                                     pbar=self.progressBar_2,
                                     rotation=self.comboBox_9.currentText(),
                                     component=self.comboBox_10.currentText())
        
        self.label_49.setText("")
        self.progressBar_2.setVisible(False)
        
        # Restore the previous accepted/rejected state
        sorted_rej = sorted(rejections, key=lambda x: x[1])
        if len(rejections) == len(self.rfs):
            for i, rf in enumerate(sorted(self.rfs, key=lambda x: x[6])):
                if sorted_rej[i][0] == 0:
                    rf[5] = 0
    
        self.rf_pages = int(math.ceil(len(self.rfs)/self.spinBox_3.value()))
        self.label_14.setText("{}/{}".format(self.rf_current_page, self.rf_pages))
        # Reset filter menu - these kind of things should be turned into functions
        self.rf_freq_filter_isapplied = False
        self.rf_freq_filter_params = None
        self.pushButton_21.setEnabled(False)
        self.pushButton_22.setEnabled(True)
        self.RFs_are_computed = True
        self.plot_rfs()
    
    def filter_receiver_functions(self):
         # Post-computation filtering
        pfilter_type = self.comboBox_8.currentText()
        order = self.spinBox_4.value()
        fmin = self.doubleSpinBox_37.value()
        fmax = self.doubleSpinBox_39.value()
        
        # This funcion should be moved to utils
        if pfilter_type == "Bandpass":
            filter_ = obspy.signal.filter.bandpass
            args = [fmin, fmax]
        elif pfilter_type == "Lowpass":
            filter_ = obspy.signal.filter.lowpass
            args = [fmax]
        elif pfilter_type == "Highpass":
            filter_ = obspy.signal.filter.highpass
            args = [fmin]
        
        if pfilter_type == "Bandpass" and fmin >= fmax:
            return
        
        if not self.rf_freq_filter_isapplied:
            self.unfiltered_rfs = copy.deepcopy(self.rfs)
            self.rfs = []
            for rf in self.unfiltered_rfs:
                rf = copy.deepcopy(rf)
                df = 1/(rf[1][1] - rf[1][0])
                filtered_rf = filter_(rf[0], *args, df, corners=order, zerophase=True)
                rf[0] = filtered_rf
                self.rfs.append(rf)
            self.pushButton_21.setEnabled(True)
            self.pushButton_22.setEnabled(False)
            self.plot_rfs()
            self.rf_freq_filter_isapplied = True
            self.rf_freq_filter_params = [filter_, args, df, order]
        elif self.rf_freq_filter_isapplied:
            self.rfs = copy.deepcopy(self.unfiltered_rfs)
            self.unfiltered_rfs = []
            self.plot_rfs()
            self.pushButton_21.setEnabled(False)
            self.pushButton_22.setEnabled(True)
            self.rf_freq_filter_isapplied = False
            self.rf_freq_filter_params = None
    
    def reset_rf_analysis_gui(self):
        self.rfs = []
        self.rf_current_page = 1
        self.rf_pages = 1       
        self.RFs_are_computed = False
        
        # Reset filter menu
        self.rf_freq_filter_isapplied = False
        self.rf_freq_filter_params = None
        self.pushButton_21.setEnabled(False)
        self.pushButton_22.setEnabled(True)
        
        self.linear_RF_stack = None
    
        self.first_rf_stack_plot = True # To be removed
        self.first_hk_stack_plot = True # To be removed
        self.hk_result = {}
        
        # Theoretical arrival times obtained from the maximum of the H-k
        self.Ps = 0
        self.PpPs = 0
        self.PpSs_PsPs = 0        
        
        # Clear all matplotlib axes
        for mplwidget in [self.mplwidget, self.mplwidget_2, self.mplwidget_3, self.mplwidget_4]:
            mplwidget.figure.clf()
            mplwidget.figure.canvas.draw()
        
        # Reset Hk measurements
        self.label_3.setText("H: n/a")
        self.label_22.setText("k: n/a")
    
    def setup_rf_axes(self):
        """Prepare receiver function axes for plotting
        
        """
        self.mplwidget.figure.clf()
        gs = gridspec.GridSpec(self.spinBox_3.value(), 1)
        gs.update(left=0.10, right=0.95, top=0.98, bottom=0.085, hspace=0.35)
        for i in range(self.spinBox_3.value()):
            self.mplwidget.figure.add_subplot(gs[i])
            
        for i in range(self.spinBox_3.value() - 1):
            self.mplwidget.figure.axes[i].set_xticklabels([])
            self.mplwidget.figure.axes[self.spinBox_3.value()-1].set_xlabel("Time in seconds")
    
    def number_of_rfs_per_page_changed(self):
        self.rf_current_page = 1
        self.rf_pages = int(math.ceil(len(self.rfs)/self.spinBox_3.value()))
        self.label_14.setText("{}/{}".format(self.rf_current_page, self.rf_pages))
        self.plot_rfs()   
    
    def plot_rfs(self):
        """ Sort and plot receiver functions according to user selection
        
        """
        if not self.RFs_are_computed:
            return        
        
        if self.comboBox_3.currentText() == "Back az.":
            sort_index = 2
        elif self.comboBox_3.currentText() == "Distance":
            sort_index = 3
        
        self.rfs = sorted(self.rfs, key=lambda x: x[sort_index])
        
        self.setup_rf_axes()
        
        xmin = min([self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value()])
        xmax = max([self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value()])
            
        rf_index = 0 + (self.rf_current_page - 1)*self.spinBox_3.value()
        j = np.minimum(self.spinBox_3.value(), len(self.rfs[rf_index:]))
        for i in range(j):
            rf = self.rfs[rf_index+i][0]
            t = self.rfs[rf_index+i][1]
            text = np.round(self.rfs[rf_index+i][sort_index], decimals=0)
            evid = self.rfs[rf_index+i][6]
    
            self.mplwidget.figure.axes[i].fill_between(t, rf, color=self.settings['rfs']['appearance']['line_color'],
                                                       linewidth=self.settings['rfs']['appearance']['line_width'])            
            self.mplwidget.figure.axes[i].fill_between(t, np.zeros(len(t)),
                                                       rf, where=(rf > 0), color=self.settings['rfs']['appearance']['positive_fill_color'])
            self.mplwidget.figure.axes[i].fill_between(t, np.zeros(len(t)),
                                                       rf, where=(rf < 0), color=self.settings['rfs']['appearance']['negative_fill_color'])
            
            if self.rfs[rf_index+i][5]:
                plt.setp(self.mplwidget.figure.axes[i].spines.values(), color='darkblue', linewidth=0.4)
            else:
                plt.setp(self.mplwidget.figure.axes[i].spines.values(), color='red', linewidth=2)
            
            self.mplwidget.figure.axes[i].set_xlim(xmin, xmax)
            self.mplwidget.figure.axes[i].text(1, 1.035, "{}".format(text)+r"$\degree$"+" - EQ{}".format(evid),
                                               transform = self.mplwidget.figure.axes[i].transAxes,
                                               ha='right', fontweight="bold")
            
        # Update text with number of discarded rfs
        self.label_24.setText("Total {} receiver functions ({} discarded)".format(len(self.rfs),
                                                                                  len([x for x in self.rfs if x[5] == 0])))
        self.mplwidget.figure.canvas.draw()
    
    def next_rfs_page(self):
        """Show the next page in the receiver function panel
        
        """
        if self.rf_current_page < self.rf_pages:
            self.rf_current_page += 1
            self.label_14.setText("{}/{}".format(self.rf_current_page, self.rf_pages))
            self.plot_rfs()
    
    def previous_rfs_page(self):
        """Show the previous page in the receiver function panel
        
        """
        if self.rf_current_page > 1:
            self.rf_current_page -= 1
            self.label_14.setText("{}/{}".format(self.rf_current_page, self.rf_pages))
            self.plot_rfs()
    
    def recf_plot_clicked(self, event):
        """Register click events in the receiver function panel
        
        """
        axind = self.mplwidget.figure.axes.index(event.inaxes)
        rf_index = (self.rf_current_page - 1)*self.spinBox_3.value() + axind
        
        # Read the gaussian filter width and waverlevel, if necessary
        a = self.doubleSpinBox.value()
        self.settings['rfs']['computation_method']['method_settings']['gaussian_filter_width'] = a
        if self.settings['rfs']['computation_method']['method'] == 'Waterlevel':
            c = self.doubleSpinBox_2.value()
            self.settings['rfs']['computation_method']['method_settings']['waterlevel_parameter'] = c
        
        if event.button == matplotlib.backend_bases.MouseButton(1):  # Left click discards rf
            if self.rfs[rf_index][5]:
                self.rfs[rf_index][5] = 0
            else:
                self.rfs[rf_index][5] = 1
            self.plot_rfs()
        elif event.button == matplotlib.backend_bases.MouseButton(3): # Right click shows earthquake
            eq_id = self.rfs[rf_index][6]
            dialog = dialogs.ShowEarthquakeDialog(self.hdf5_waveforms, self.comboBox.currentText(), eq_id, self.settings, self.comboBox_9.currentText(), self.rf_freq_filter_params)
            dialog.exec_()
    
    def hk_plot_clicked(self, event):
        """Register click events in the hk panel
        
        """
        pass
    
    
    def auto_reject_rfs(self):
        baz_start = self.spinBox_7.value()
        baz_end = self.spinBox_10.value()
        for rf in self.rfs:
            rf[5] = 1
            if self.checkBox_2.isChecked() and rf[7] < self.doubleSpinBox_29.value():
                rf[5] = 0
            
            if self.checkBox_3.isChecked() and rf[8] < self.doubleSpinBox_40.value():
                rf[5] = 0
            
            if self.checkBox_4.isChecked():
                baz = rf[2]
                if baz_start < baz_end:
                    inside = (baz_start <= baz <= baz_end)
                else:
                    inside = (baz >= baz_start or baz <= baz_end)
                
                if not inside:
                    rf[5] = 0
                
        self.plot_rfs()
    
    def remove_rejections(self):
        for rf in self.rfs:
            if rf[5] == 0:
                rf[5] = 1
        
        self.plot_rfs()
    
    def save_rfs(self):
        """Save the current receiver functions to disk
        
        """
        if len(self.rfs) > 0:
            outfile = QtWidgets.QFileDialog.getSaveFileName()[0]
            if outfile:
                stnm = self.comboBox.currentText()            
                mwu.save_rfs_hdf5(stnm, self.hdf5_waveforms[stnm].attrs["stla"], self.hdf5_waveforms[stnm].attrs["stlo"],
                                  self.hdf5_waveforms[stnm].attrs["stel"],  self.settings['rfs']['computation_method']['method'],
                                  self.settings['rfs']['computation_method']['method_settings'], self.rfs, outfile)
        else:
            qm = QtWidgets.QMessageBox
            ret = qm.warning(self,'', "No receiver functions to save! First compute some.")
    
    def setup_rf_stack_axes(self):
        """Prepare receiver function axes for plotting
        
        """
        if self.first_rf_stack_plot:
            self.mplwidget_2.figure.subplots(2, gridspec_kw={'height_ratios': [1, 7]})
            self.mplwidget_2.figure.subplots_adjust(left=0.125, bottom=0.075,
                                                    right=0.95, top=0.98, hspace=0.1)
            self.first_rf_stack_plot = False
        else:
            for i in range(2):
                self.mplwidget_2.figure.axes[i].clear()
        
        self.mplwidget_2.figure.axes[1].set_xlabel("Time in seconds")
    
    def plot_rf_stack(self):
        """Plot RF stack according to user preferences
        
        """
        if self.spinBox.value() < self.spinBox_2.value():
            qm = QtWidgets.QMessageBox
            ret = qm.warning(self,'', "The overlap can not be larger than the bin size!")
            return
        
        stack, bin_stacks, bins = mwu.compute_stack(self.rfs, bin_size=self.spinBox.value(),
                                                                  overlap=self.spinBox_2.value(),
                                                                  stack_by=self.comboBox_2.currentText(),
                                                                  moveout_phase=self.comboBox_4.currentText(),
                                                                  ref_slowness=self.settings['rfs']['stacking']['ref_slowness'])
        self.setup_rf_stack_axes()
        self.linear_RF_stack = stack
        
        t = self.rfs[0][1]
        
        self.mplwidget_2.figure.axes[0].plot(t, stack, color=self.settings['rfs']['appearance']['line_color'],
                                             linewidth=self.settings['rfs']['appearance']['line_width'])
        self.mplwidget_2.figure.axes[0].fill_between(t, 0, stack,
                                                     where=(stack > 0),
                                                     color=self.settings['rfs']['appearance']['positive_fill_color'])
        self.mplwidget_2.figure.axes[0].fill_between(t, 0, stack,
                                                     where=(stack < 0),
                                                     color=self.settings['rfs']['appearance']['negative_fill_color'])
        zorder = len(bins)
        
        if self.comboBox_2.currentText() == "Distance":
            factor = 2
        else:
            factor = 10
        
        for i, b in enumerate(bins):
            bstack = bin_stacks[i]*factor+b
            height = np.zeros(len(t)) + b
            self.mplwidget_2.figure.axes[1].plot(t, bstack, color=self.settings['rfs']['appearance']['line_color'],
                                                 linewidth=self.settings['rfs']['appearance']['line_width'], zorder=zorder)
            self.mplwidget_2.figure.axes[1].fill_between(t, height, bstack,
                                                          where=(bstack > b),
                                                          color=self.settings['rfs']['appearance']['positive_fill_color'], zorder=zorder)
            self.mplwidget_2.figure.axes[1].fill_between(t, height, bstack,
                                                          where=(bstack < b),
                                                          color=self.settings['rfs']['appearance']['negative_fill_color'], zorder=zorder)
        
            zorder -= 1
        
        xmin = min([self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value()])
        xmax = max([self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value()])
    
        self.mplwidget_2.figure.axes[0].set_xlim(xmin, xmax)
        self.mplwidget_2.figure.axes[1].set_xlim(xmin, xmax)
        
        if self.comboBox_2.currentText() == "Back az.":
            self.mplwidget_2.figure.axes[1].set_ylabel('Back azimuth '+r'$(\degree)$')
        else:
            self.mplwidget_2.figure.axes[1].set_ylabel('Distance '+r'$(\degree)$')
        
        if self.checkBox.isChecked() and self.Ps > 0:
            # Plot the theoretical traveltimes
            self.mplwidget_2.figure.axes[0].axvline(self.Ps, color='gray', linewidth=0.75)
            self.mplwidget_2.figure.axes[0].axvline(self.PpPs, color='gray', linewidth=0.75)
            self.mplwidget_2.figure.axes[0].axvline(self.PpSs_PsPs, color='gray', linewidth=0.75)
            self.mplwidget_2.figure.axes[0].text(self.Ps+0.5, 0.9, "Ps", color='gray')
            self.mplwidget_2.figure.axes[0].text(self.PpPs+0.5, 0.9, "PpPs", color='gray')
            self.mplwidget_2.figure.axes[0].text(self.PpSs_PsPs+0.5, 0.9, "PpSs+PsPs", color='gray')
    
    
            self.mplwidget_2.figure.axes[1].axvline(self.Ps, color='gray', linewidth=0.5)
            self.mplwidget_2.figure.axes[1].axvline(self.PpPs, color='gray', linewidth=0.5)
            self.mplwidget_2.figure.axes[1].axvline(self.PpSs_PsPs, color='gray', linewidth=0.5)
        
        self.mplwidget_2.figure.canvas.draw()
    
    def setup_hk_stack_axes(self):
        """Prepare H-k stack axes for plotting
        
        """
        if self.first_hk_stack_plot:
            self.mplwidget_3.figure.subplots(1)
            self.mplwidget_3.figure.subplots_adjust(left=0.175, right=0.95, top=0.95, bottom=0.15)
            self.first_hk_stack_plot = False
        else:
            self.mplwidget_3.figure.axes[0].clear()
        
        self.mplwidget_3.figure.axes[0].set_xlabel("H (km)")
        self.mplwidget_3.figure.axes[0].set_ylabel("k")
    
    def plot_hk_stack(self):
        """Plot RF stack according to user preferences
        
        """
        minH = min([self.doubleSpinBox_5.value(), self.doubleSpinBox_6.value()])
        maxH = max([self.doubleSpinBox_5.value(), self.doubleSpinBox_6.value()])
        Hvalues = self.settings['hk']['computation']['H_points']
        
        mink = min([self.doubleSpinBox_3.value(), self.doubleSpinBox_4.value()])
        maxk = max([self.doubleSpinBox_3.value(), self.doubleSpinBox_4.value()])
        kvalues = self.settings['hk']['computation']['k_points']
        
        w1 = self.doubleSpinBox_30.value()
        w2 = self.doubleSpinBox_31.value()
        w3 = self.doubleSpinBox_32.value()
        
        H_arr, k_arr, matrix, H, k, events = mwu.compute_hk_stack(self.rfs, H_range=(minH, maxH), H_values=Hvalues,
                                                                  k_range=(mink, maxk), k_values=kvalues,
                                                                  w1=w1, w2=w2, w3=w3,
                                                                  avg_vp=self.settings['hk']['computation']['avg_vp'],
                                                                  semblance_weighting=self.settings['hk']['computation']['semblance_weighting'])
        
        self.Ps, self.PpPs, self.PpSs_PsPs = mwu.compute_theoretical_arrival_times(H, k,
                                                                                   ref_slowness=self.settings['hk']['theoretical_atimes']['ref_slowness'],
                                                                                   avg_vp=self.settings['hk']['theoretical_atimes']['avg_vp'])
    
        N_stacked = len([rf for rf in self.rfs if rf[5]]) # NOT VERY EFFICIENT! JUST TESTING
        a, error_area, k_95, H_95, error_contour_level = mwu.determine_error_region(matrix, H_arr, k_arr, N_stacked)     
    
        self.hk_result = {"H_arr":H_arr,
                          "k_arr":k_arr,
                          "events":events,
                          "Hk_stack":matrix,
                          "H":H,
                          "H_95":H_95,
                          "k":k,
                          "k_95":k_95}
    
        self.setup_hk_stack_axes()
        if self.settings['hk']['appearance']['plotting_method'] == 'colored grid':
            self.mplwidget_3.figure.axes[0].pcolormesh(H_arr, k_arr, matrix, cmap=self.settings['hk']['appearance']['colormap'], vmin=0)
        elif self.settings['hk']['appearance']['plotting_method'] == 'filled contour map':
            self.mplwidget_3.figure.axes[0].contourf(H_arr, k_arr, matrix, levels=10, cmap=self.settings['hk']['appearance']['colormap'], vmin=0)
        elif self.settings['hk']['appearance']['plotting_method'] == 'contour map':
            self.mplwidget_3.figure.axes[0].contour(H_arr, k_arr, matrix, levels=10, cmap=self.settings['hk']['appearance']['colormap'], vmin=0)
            
        self.mplwidget_3.figure.axes[0].contour(H_arr, k_arr, matrix, levels=[np.max(matrix) - error_contour_level],
                                                colors=[self.settings['hk']['appearance']['ser_color']])
        self.mplwidget_3.figure.axes[0].set_xlim(minH, maxH)
        self.mplwidget_3.figure.axes[0].set_ylim(mink, maxk)
        if error_area != None:
            self.mplwidget_3.figure.axes[0].axvline(H, color=self.settings['hk']['appearance']['line_color'])
            self.mplwidget_3.figure.axes[0].axhline(k, color=self.settings['hk']['appearance']['line_color'])
            self.label_3.setText("H: {:.2f} ({:.2f} - {:.2f} @ 95%) km".format(H, H_95[0], H_95[1]))
            self.label_22.setText("k: {:.2f} ({:.2f} - {:.2f} @ 95%)".format(k, k_95[0], k_95[1]))
        else:
            self.label_3.setText("H: {:.2f} (n/a @ 95%) km".format(H))
            self.label_22.setText("k: {:.2f} (n/a @ 95%)".format(k))
        self.mplwidget_3.figure.canvas.draw()
    
    def save_hk_result(self):
        """Saves the H-k stack. Writes the station name, location, and H and
        k values in a chosen txt file (creates one if it doesn't exist). Saves
        the H-k stack as a .pickle in the same location.
    
        """
        stnm = self.comboBox.currentText()
        
        txt_fname = QtWidgets.QFileDialog.getSaveFileName()[0]
        if not txt_fname:
            return
        
        if not os.path.exists(txt_fname):
            with open(txt_fname, "w", newline='\n') as f:
                f.write("STATION,LONG,LAT,EVENTS,H,MIN_H95,MAX_H95,k,MIN_k95,MAX_k95,METHOD,PARAMETERS" + '\n')
    
        stla = self.hdf5_waveforms[stnm].attrs["stla"]
        stlo = self.hdf5_waveforms[stnm].attrs["stlo"]
        
        try:
            line = stnm + "," + str(stlo) + "," + str(stla) + "," + str(self.hk_result['events']) + "," + str(self.hk_result['H']) + "," + str(self.hk_result['H_95'][0]) + "," + str(self.hk_result['H_95'][1]) \
                   + "," + str(self.hk_result['k']) + "," + str(self.hk_result['k_95'][0]) + "," + str(self.hk_result['k_95'][1])
                   
            # Add information regarding the method and parameters
            line += "," + self.settings["rfs"]["computation_method"]["method"]# + "," + "f1={}".format(self.doubleSpinBox_27.value()) + ";" + "f2={}".format(self.doubleSpinBox_28.value())
            for key in self.settings["rfs"]["computation_method"]["method_settings"]:
                line += ";" + "{}={}".format(key, self.settings["rfs"]["computation_method"]["method_settings"][key])
                   
            with open(txt_fname, "a", newline='\n') as f:
                f.write(line + '\n')
        except TypeError:
            qm = QtWidgets.QMessageBox
            ret = qm.warning(self,'', "No maximum found in the H-k stack. Please adjust the parameters and try again.")
        
        pickle_output_path = os.sep.join(txt_fname.split(os.sep)[:-1])
        pickle_fname = stnm + "_Hk" + ".pickle"
        pickle.dump(self.hk_result, open(os.path.join(pickle_output_path, pickle_fname), "wb"))
    
    def setup_map_axes(self):
        """Prepare map axes for plotting
        
        """
        self.mplwidget_4.figure.clf()
        
        stnm = self.comboBox.currentText()
        lat = self.hdf5_waveforms[stnm].attrs["stla"]
        lon = self.hdf5_waveforms[stnm].attrs["stlo"]
        proj = ccrs.Stereographic(central_longitude=lon, central_latitude=lat)
        
        self.mplwidget_4.figure.subplots(1, subplot_kw=dict(projection=proj))
        self.mplwidget_4.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Aqui tiene que haber un for con los circulitos que elija el usuario
        r1 = mwu.compute_radius(proj, lat, lon, 30)
        r2 = mwu.compute_radius(proj, lat, lon, 90)        
        self.mplwidget_4.figure.axes[0].text(r1, -r1, r"$30\degree$", fontsize=9)
        self.mplwidget_4.figure.axes[0].text(0.8*r2, -r2*0.8, r"$90\degree$", fontsize=9)
        self.mplwidget_4.figure.axes[0].add_patch(mpatches.Circle(xy=[lon, lat], radius=r1, fill=False, edgecolor='darkblue', alpha=1, transform=proj, zorder=30))
        self.mplwidget_4.figure.axes[0].add_patch(mpatches.Circle(xy=[lon, lat], radius=r2, fill=False, edgecolor='darkblue', alpha=1, transform=proj, zorder=30))
        
        self.mplwidget_4.figure.axes[0].stock_img()
        self.mplwidget_4.figure.axes[0].coastlines()
    
    def plot_map(self):
        """Plot event map
        
        """
        
        self.setup_map_axes()
        self.mplwidget_4.figure.canvas.draw()
        stnm = self.comboBox.currentText()
        
        for rf in self.rfs:
            if rf[5]:
                event_id = rf[6]
                lat = self.hdf5_waveforms[stnm][event_id].attrs["evla"]
                lon = self.hdf5_waveforms[stnm][event_id].attrs["evlo"]
                self.mplwidget_4.figure.axes[0].plot(lon, lat, marker='o', color='red',
                                                     transform=ccrs.Geodetic(), markersize=1.5)
        
        self.mplwidget_4.figure.canvas.draw()
    
    def read_hk_results(self):
        fname = QtWidgets.QFileDialog.getOpenFileName()[0]
        if fname:
            self.hk_results = mwu.read_hk_results_file(fname)
            self.pushButton_18.setEnabled(True)
    
    def plot_hk_results(self):
    
        try:
            self.mplwidget_6.figure.clf()
        except IndexError:
            pass
    
        
        self.mplwidget_6.figure.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
        self.mplwidget_6.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
    
        values = []
        points = []
        for stnm in self.hk_results.keys():
            values.append(self.hk_results[stnm][self.comboBox_5.currentText()])
            points.append(self.hk_results[stnm]["loc"])
        
        lats = [x[1] for x in points]
        lons = [x[0] for x in points]
        
        
        if len(self.hk_results.keys()) < 4:
            qm = QtWidgets.QMessageBox
            qm.warning(self,'', "Not enough points for performing the interpolation.")
        else:
            grid_x, grid_y = np.mgrid[min(lons):max(lons):100j, min(lats):max(lats):100j]
            grid_z = scint.griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.NaN)
            
            if self.settings['map']['appearance']['plotting_method'] == 'colored grid':
                CS = self.mplwidget_6.figure.axes[0].pcolormesh(grid_x, grid_y, grid_z, cmap=self.settings['map']['appearance']['colormap'])
            elif self.settings['map']['appearance']['plotting_method'] == 'filled contour map':
                CS = self.mplwidget_6.figure.axes[0].contourf(grid_x, grid_y, grid_z, cmap=self.settings['map']['appearance']['colormap'])
            elif self.settings['map']['appearance']['plotting_method'] == 'contour map':
                CS = self.mplwidget_6.figure.axes[0].contour(grid_x, grid_y, grid_z, cmap=self.settings['map']['appearance']['colormap'])
        
            cbar = self.mplwidget_6.figure.add_axes([0.25, 0.1, 0.50, 0.01])
            self.mplwidget_6.figure.colorbar(CS, cax=cbar, orientation='horizontal')
            if self.comboBox_5.currentText() == 'H':
                cbar.set_xlabel('Crustal thickness (km)')
            elif self.comboBox_5.currentText() == 'k':
                cbar.set_xlabel(r'$V_P/V_S$')
        
        if self.settings['map']['appearance']['include_stations']:
            self.mplwidget_6.figure.axes[0].scatter(lons, lats, marker=self.settings['map']['appearance']['station_marker'],
                                                    s=40, c=self.settings['map']['appearance']['station_marker_color'])
        
        if self.settings['map']['shapefiles']['include'] and self.settings['map']['shapefiles']['path']:
            sfs = mwu.read_shapefiles(self.settings['map']['shapefiles']['path'])
            for sf in sfs:
                for shape in sf.shapeRecords():
                    x = [i[0] for i in shape.shape.points[:]]
                    y = [i[1] for i in shape.shape.points[:]]
                    self.mplwidget_6.figure.axes[0].plot(x,y, color='black', alpha=0.4)
        
        self.mplwidget_6.figure.axes[0].set_extent([min(lons)-0.1, max(lons)+0.1,
                                                    min(lats)-0.1, max(lats)+0.1], crs=ccrs.PlateCarree())
        
        # Update GUI elements
        for doubleSpinBox in [self.doubleSpinBox_33, self.doubleSpinBox_34,
                              self.doubleSpinBox_35, self.doubleSpinBox_36]:
            
            doubleSpinBox.disconnect()
            
        self.doubleSpinBox_34.setValue(min(lats) - 0.1)
        self.doubleSpinBox_33.setValue(max(lats) + 0.1)
        self.doubleSpinBox_35.setValue(min(lons) - 0.1)
        self.doubleSpinBox_36.setValue(max(lons) + 0.1)
        
        for doubleSpinBox in [self.doubleSpinBox_33, self.doubleSpinBox_34,
                              self.doubleSpinBox_35, self.doubleSpinBox_36]:
            
            doubleSpinBox.valueChanged.connect(self.update_hk_map)
        
        self.mplwidget_6.figure.axes[0].set_zorder(1)
    
        self.hk_basemap()
        self.hk_gridlines()
    
    
    def update_hk_map(self):
        lons = (self.doubleSpinBox_35.value(),self.doubleSpinBox_36.value())
        lats = (self.doubleSpinBox_34.value(),self.doubleSpinBox_33.value())
        try: # Try except due to persistence of interface changes in ISP
            self.mplwidget_6.figure.axes[0].set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
            try:
                self.mplwidget_6_basemap_ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
            except AttributeError:
                pass
            try:
                self.mplwidget_6_gridlines.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
            except AttributeError:
                pass
            
            self.mplwidget_6.figure.canvas.draw()
        except:
            pass
    
    def hk_basemap(self):
        
        try:
            del self.mplwidget_6_basemap_ax
        except AttributeError:
            pass
    
        try:
            self.mplwidget_6.figure.axes[0].patch.set_facecolor((1, 1, 1, 0))
            self.mplwidget_6_basemap_ax = self.mplwidget_6.figure.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
            self.mplwidget_6_basemap_ax.set_zorder(0)
            self.mplwidget_6.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
            self.mplwidget_6_basemap_ax.set_extent(self.mplwidget_6.figure.axes[0].get_extent())
            
            if self.comboBox_7.currentText() == 'Stamen':
                self.mplwidget_6_basemap_ax.add_image(Stamen('terrain-background'), self.spinBox_8.value())
        
            self.mplwidget_6.figure.canvas.draw()
        except IndexError:
            return
    
    def hk_gridlines(self):
    
        try:
            self.mplwidget_6_gridlines.remove()
            del self.mplwidget_6_gridlines
        except (AttributeError, KeyError, ValueError) as e:
            pass
    
        self.mplwidget_6_gridlines = self.mplwidget_6.figure.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
        self.mplwidget_6.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
        self.mplwidget_6_gridlines.set_extent(self.mplwidget_6.figure.axes[0].get_extent())
        self.mplwidget_6_gridlines.patch.set_facecolor((1, 1, 1, 0))
        self.mplwidget_6_gridlines.set_zorder(2)
        
        if self.doubleSpinBox_38.value() > 0.01:
            spacing = self.doubleSpinBox_38.value()
            gl = self.mplwidget_6_gridlines.gridlines(draw_labels=True)
            extent = self.mplwidget_6.figure.axes[0].get_extent()
            xticks = np.arange(extent[0], extent[1] + spacing, spacing)
            yticks = np.arange(extent[2], extent[3] + spacing, spacing)
            gl.xlocator = mticker.FixedLocator(xticks)
            gl.ylocator = mticker.FixedLocator(yticks)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        
        self.mplwidget_6.figure.canvas.draw()
    
    def save_linear_rf_stack(self):
        fname = QtWidgets.QFileDialog.getSaveFileName(None, "Save RF stack", "", "csv (*.csv)")[0]
        if fname:
            time = self.rfs[0][1]
            with open(fname, "w") as f:
                for t, a in zip(time, self.linear_RF_stack):
                    f.write("{},{}\n".format(t,a))
        
    def save_figure(self, figure):
        fname = QtWidgets.QFileDialog.getSaveFileName(None, "Save figure", "", "PNG (*.png)")[0]
        if fname:
            figure.savefig(fname, dpi=600)
    
    def ccp_stack_map_toggle_plot_mode(self, mode):
        """Enable or disable user ability to manually plot the grid limits or
        a cross-section
        
        """        
        if mode == "grid":
            self.pushButton_25.setChecked(False)
        elif mode == "cross_section":
            self.pushButton_11.setChecked(False)
    
    def ccp_stack_map_event_handler(self, event):
        """Pick event handler for the CCP stack map
        
        """        
    
        if self.pushButton_11.isChecked():
            mode = "grid"
            dict_ = self.ccp_grid
        elif self.pushButton_25.isChecked():
            mode = "cross_section"
            dict_ = self.ccp_cross_section
    
        else:
            return
    
        if event.name == "button_press_event":
            dict_['x0'] = event.xdata
            dict_['y0'] = event.ydata
            if mode == "grid":
                try:
                    self.ccp_grid_mpl_line.remove()
                    self.mplwidget_5.figure.canvas.restore_region(self.ccp_events_background)
                    self.mplwidget_5.figure.canvas.blit(self.mplwidget_5.figure.axes[0].bbox)
                    self.mplwidget_5.figure.canvas.flush_events()
                except AttributeError:
                    pass                    
                # Cache the background
                self.ccp_events_background = self.mplwidget_5.figure.canvas.copy_from_bbox(self.mplwidget_5.figure.bbox)
                self.ccp_grid_mpl_line, = self.mplwidget_5.figure.axes[0].plot(event.xdata, event.ydata, color="red", animated=True)
            elif mode == "cross_section":
                try:
                    self.ccp_cross_section_mpl_line.remove()
                    self.mplwidget_5.figure.canvas.restore_region(self.ccp_events_background)
                    self.mplwidget_5.figure.canvas.blit(self.mplwidget_5.figure.axes[0].bbox)
                    self.mplwidget_5.figure.canvas.flush_events()
                except AttributeError:
                    pass                    
                # Cache the background
                self.ccp_events_background = self.mplwidget_5.figure.canvas.copy_from_bbox(self.mplwidget_5.figure.bbox)
                self.ccp_cross_section_mpl_line, = self.mplwidget_5.figure.axes[0].plot(event.xdata, event.ydata, color="red", animated=True)                
            self.button_pressed = True
        elif event.name == "button_release_event":
            self.button_pressed = False
        elif event.name == "motion_notify_event":
            if self.button_pressed:
                dict_['x1'] = event.xdata
                dict_['y1'] = event.ydata
        
                if mode == "grid":
                    x = [dict_['x0'], dict_['x1'],
                         dict_['x1'], dict_['x0'],
                         dict_['x0']]
                    y = [dict_['y0'], dict_['y0'],
                         dict_['y1'], dict_['y1'],
                         dict_['y0']]
    
                    self.ccp_grid_mpl_line.set_xdata(x)
                    self.ccp_grid_mpl_line.set_ydata(y)
                    self.mplwidget_5.figure.canvas.restore_region(self.ccp_events_background)
                    self.mplwidget_5.figure.axes[0].draw_artist(self.ccp_grid_mpl_line)
                    self.mplwidget_5.figure.axes[0].patch.set_alpha(0)
                    self.set_map_zorder()                    
                    self.mplwidget_5.figure.canvas.blit(self.mplwidget_5.figure.axes[0].bbox)
                    self.mplwidget_5.figure.canvas.flush_events()
                    
                elif mode == "cross_section":
                    x = [dict_['x0'], dict_['x1']]
                    y = [dict_['y0'], dict_['y1']]
                    
                    self.mplwidget_5.figure.canvas.restore_region(self.ccp_events_background)
                    self.ccp_cross_section_mpl_line.set_xdata(x)
                    self.ccp_cross_section_mpl_line.set_ydata(y)
                    self.mplwidget_5.figure.axes[0].draw_artist(self.ccp_cross_section_mpl_line)
                    self.mplwidget_5.figure.canvas.blit(self.mplwidget_5.figure.bbox)
                    self.mplwidget_5.figure.axes[0].patch.set_alpha(0)
                    self.set_map_zorder()
                    self.mplwidget_5.figure.canvas.flush_events()
                
                self.ccp_stack_set_comboboxes()
    
    def ccp_stack_set_comboboxes(self):
        """Update the comboboxes in the CCP stack options with the correct grid
        and cross-section coordinates
        
        """
        if self.pushButton_11.isChecked():
            self.doubleSpinBox_9.setValue(self.ccp_grid['y0'])
            self.doubleSpinBox_10.setValue(self.ccp_grid['y1'])
            self.doubleSpinBox_11.setValue(self.ccp_grid['x0'])
            self.doubleSpinBox_12.setValue(self.ccp_grid['x1'])
        elif self.pushButton_25.isChecked():
            start = (self.ccp_cross_section['y0'], self.ccp_cross_section['x0'])
            end = (self.ccp_cross_section['y1'], self.ccp_cross_section['x1'])
            self.doubleSpinBox_17.setValue(start[0])
            self.doubleSpinBox_20.setValue(start[1])
            self.doubleSpinBox_18.setValue(end[1])
            self.doubleSpinBox_19.setValue(end[0])
            
    
    def ccp_stack_read_rfs(self):
        """Read receiver functions and plot stations on the map
        
        """
        # dir_ = QtWidgets.QFileDialog.getExistingDirectory()
        # self.rfs_dicts = mwu.map_rfs(rfs_dir=dir_)
        file = QtWidgets.QFileDialog.getOpenFileName()[0]
        if file:
            self.rfs_hdf5 = mwu.read_hdf5(file)
            self.plot_ccp_map()
            self.mplwidget_5.figure.canvas.draw()
    
    def compute_ccp_stack(self):
        """Compute and draw CCP stack
        
        """
        dz = self.doubleSpinBox_15.value()
        max_depth = self.doubleSpinBox_16.value()
        dlat = self.doubleSpinBox_14.value()
        dlon = self.doubleSpinBox_13.value()
        earth_model = self.comboBox_11.currentText()
        
        self.progressBar.setVisible(True)
        self.ccp_stack, self.ccp_x, self.ccp_y, self.ccp_z = mwu.ccp_stack(self.rfs_hdf5,
                                                                           min([self.doubleSpinBox_11.value(), self.doubleSpinBox_12.value()]),
                                                                           max([self.doubleSpinBox_11.value(), self.doubleSpinBox_12.value()]),
                                                                           min([self.doubleSpinBox_9.value(), self.doubleSpinBox_10.value()]),
                                                                           max([self.doubleSpinBox_9.value(), self.doubleSpinBox_10.value()]),
                                                                           dlon,
                                                                           dlat,
                                                                           dz,
                                                                           max_depth,
                                                                           model=earth_model,
                                                                           stacking_method=self.settings['ccp']['computation']['stacking_method'],
                                                                           pbar=self.progressBar)      
        self.progressBar.setVisible(False)
        
        self.istack = None
        self.plot_ccp_stack()
        
        self.doubleSpinBox_25.setMaximum(max_depth-dz)
        self.doubleSpinBox_25.setSingleStep(dz)
    
    def plot_ccp_map(self):
    
        self.mplwidget_5.figure.clf()
        self.mplwidget_5.figure.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
        self.mplwidget_5.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
        
        lats, lons = [], []
        for stnm in self.rfs_hdf5:
            stla = self.rfs_hdf5[stnm].attrs["stla"]
            stlo = self.rfs_hdf5[stnm].attrs["stlo"]
            # Zstel = self.rfs_hdf5[stnm].attrs["stel"]
            lats.append(stla)
            lons.append(stlo)
    
            if self.settings['ccp']['appearance']['include_stations']:
                self.mplwidget_5.figure.axes[0].plot(stlo, stla, marker=self.settings['ccp']['appearance']['station_marker'],
                                                     transform=ccrs.Geodetic(), color=self.settings['ccp']['appearance']['station_marker_color'])
    
        if self.settings['ccp']['shapefiles']['include'] and self.settings['ccp']['shapefiles']['path']:
            sfs = mwu.read_shapefiles(self.settings['ccp']['shapefiles']['path'])
            for sf in sfs:
                for shape in sf.shapeRecords():
                    x = [i[0] for i in shape.shape.points[:]]
                    y = [i[1] for i in shape.shape.points[:]]
                    self.mplwidget_5.figure.axes[0].plot(x,y, color='black', alpha=0.4)
        
        self.mplwidget_5.figure.axes[0].set_extent([min(lons)-1, max(lons)+1,
                                                    min(lats)-1, max(lats)+1], crs=ccrs.PlateCarree())
    
        # Update GUI elements
        for doubleSpinBox in [self.doubleSpinBox_21, self.doubleSpinBox_22,
                              self.doubleSpinBox_23, self.doubleSpinBox_24]:
            
            doubleSpinBox.disconnect()
            
        self.doubleSpinBox_21.setValue(min(lats) - 1)
        self.doubleSpinBox_22.setValue(max(lats) + 1)
        self.doubleSpinBox_23.setValue(min(lons) - 1)
        self.doubleSpinBox_24.setValue(max(lons) + 1)
        
        for doubleSpinBox in [self.doubleSpinBox_21, self.doubleSpinBox_22,
                              self.doubleSpinBox_23, self.doubleSpinBox_24]:
            
            doubleSpinBox.valueChanged.connect(self.update_ccp_map)
        
        self.mplwidget_5.figure.axes[0].patch.set_alpha(0)
        self.mplwidget_5.figure.axes[0].set_zorder(1)
    
        self.ccp_basemap(draw=False)
        self.ccp_gridlines(draw=False)
        self.set_map_zorder()
        self.mplwidget_5.figure.canvas.draw()
        
        self.ccp_events_background = self.mplwidget_5.figure.canvas.copy_from_bbox(self.mplwidget_5.figure.bbox)
    
    def set_map_zorder(self):
        self.mplwidget_5.figure.axes[0].set_zorder(1)
        self.mplwidget_5_basemap_ax.set_zorder(0)
        self.mplwidget_5_gridlines.set_zorder(2) 
        
    def update_ccp_map(self):
        lons = (self.doubleSpinBox_23.value(),self.doubleSpinBox_24.value())
        lats = (self.doubleSpinBox_21.value(),self.doubleSpinBox_22.value())
        for ax in self.mplwidget_5.figure.axes:
            ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
    
        self.mplwidget_5.figure.canvas.draw()
        self.ccp_events_background = self.mplwidget_5.figure.canvas.copy_from_bbox(self.mplwidget_5.figure.bbox)
    
    def ccp_basemap(self, draw=True):
        
        try:
            del self.mplwidget_5_basemap_ax
        except AttributeError:
            pass
    
        self.mplwidget_5_basemap_ax = self.mplwidget_5.figure.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
        self.mplwidget_5_basemap_ax.set_zorder(0)
        self.mplwidget_5.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
        self.mplwidget_5_basemap_ax.set_extent(self.mplwidget_5.figure.axes[0].get_extent())
        
        if self.comboBox_6.currentText() == 'Stamen':
            self.mplwidget_5_basemap_ax.add_image(Stamen('terrain-background'), self.spinBox_6.value())
    
        if draw:
            self.set_map_zorder()
            self.mplwidget_5.figure.canvas.draw()
            self.ccp_events_background = self.mplwidget_5.figure.canvas.copy_from_bbox(self.mplwidget_5.figure.bbox)
    
    def ccp_gridlines(self, draw=True):
    
        try:
            self.mplwidget_5_gridlines.remove()
            del self.mplwidget_5_gridlines
        except (AttributeError, KeyError) as e:
            pass
        
    
        spacing = self.doubleSpinBox_26.value()
    
        self.mplwidget_5_gridlines = self.mplwidget_5.figure.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
        self.mplwidget_5.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
        self.mplwidget_5_gridlines.set_extent(self.mplwidget_5.figure.axes[0].get_extent())
        self.mplwidget_5_gridlines.patch.set_alpha(0)
        
        if spacing > 0.01:
            gl = self.mplwidget_5_gridlines.gridlines(draw_labels=True)
            extent = self.mplwidget_5.figure.axes[0].get_extent()
            xticks = np.arange(extent[0], extent[1] + spacing, spacing)
            yticks = np.arange(extent[2], extent[3] + spacing, spacing)
            gl.xlocator = mticker.FixedLocator(xticks)
            gl.ylocator = mticker.FixedLocator(yticks)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        
        if draw:
            self.set_map_zorder()
            self.mplwidget_5.figure.canvas.draw()
            self.ccp_events_background = self.mplwidget_5.figure.canvas.copy_from_bbox(self.mplwidget_5.figure.bbox)
            
    
    def plot_ccp_stack(self):
        
        try:
            self.mplwidget_5_ccp_pcolormesh.remove()
        except AttributeError:
            pass
    
        alpha = self.spinBox_5.value()/100
        hslice = np.where(np.abs(self.ccp_z - self.doubleSpinBox_25.value()) == np.min(np.abs(self.ccp_z - self.doubleSpinBox_25.value())))[0][0]
        
        if self.settings['ccp']['appearance']['plotting_method'] == 'colored grid':
            self.mplwidget_5_ccp_pcolormesh = self.mplwidget_5.figure.axes[0].pcolormesh(self.ccp_x, self.ccp_y, self.ccp_stack[hslice,:,:].T,
                                                                                         alpha=alpha, transform=ccrs.PlateCarree(), cmap=self.settings['ccp']['appearance']['colormap'])
        elif self.settings['ccp']['appearance']['plotting_method'] == 'filled contour map':
            self.mplwidget_5_ccp_pcolormesh = self.mplwidget_5.figure.axes[0].contourf(self.ccp_x, self.ccp_y, self.ccp_stack[hslice,:,:].T,
                                                                                       alpha=alpha, transform=ccrs.PlateCarree(), cmap=self.settings['ccp']['appearance']['colormap'])
        elif self.settings['ccp']['appearance']['plotting_method'] == 'contour map':
            self.mplwidget_5_ccp_pcolormesh = self.mplwidget_5.figure.axes[0].contour(self.ccp_x, self.ccp_y, self.ccp_stack[hslice,:,:].T,
                                                                                      alpha=alpha, transform=ccrs.PlateCarree(), cmap=self.settings['ccp']['appearance']['colormap'])
    
        self.mplwidget_5.figure.canvas.draw()  
    
        self.ccp_events_background = self.mplwidget_5.figure.canvas.copy_from_bbox(self.mplwidget_5.figure.axes[0].bbox)
    
    def save_ccp_stack(self):
        if type(self.ccp_stack) == type(None):
            return
        
        fname = QtWidgets.QFileDialog.getSaveFileName()[0]
        
        if fname:
            ccp_dict = {"ccp_stack":self.ccp_stack,
                        "x_array":self.ccp_x,
                        "y_array":self.ccp_y,
                        "z_array":self.ccp_z}
    
            
            pickle.dump(ccp_dict, open(fname, "wb"))
    
    def read_ccp_stack(self):
        fname = QtWidgets.QFileDialog.getOpenFileName()[0]
        
        if fname:
            ccp_dict = pickle.load(open(fname, 'rb'))
            self.ccp_stack = ccp_dict["ccp_stack"]
            self.ccp_x = ccp_dict["x_array"]
            self.ccp_y = ccp_dict["y_array"]
            self.ccp_z = ccp_dict["z_array"]
            
            self.doubleSpinBox_25.setMaximum(np.max(self.ccp_z))
            self.doubleSpinBox_25.setSingleStep(np.diff(self.ccp_z)[0])
            self.plot_ccp_stack()
    
    def cross_section_dialog(self):
        
        if self.istack == None:
            self.istack = mwu.interpolate_ccp_stack(self.ccp_x, self.ccp_y, self.ccp_stack)
        
        start = (self.doubleSpinBox_17.value(), self.doubleSpinBox_20.value())
        end = (self.doubleSpinBox_19.value(), self.doubleSpinBox_18.value())
        newlats, newlons, dist_arr = mwu.compute_intermediate_points(start, end, 100)
    
        matrix = []
        for i, stack in enumerate(self.istack):
            row = []
            for lat, lon in zip(newlats, newlons):
                row.append(stack(lat, lon)[0])
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        dialog = dialogs.CrossSectionDialog(dist_arr, -self.ccp_z, matrix[:, 1:-1], start, end)
        dialog.exec_()
        
    def cut_earthquakes_dialog(self):
        """Display the Cut earthquakes from local data dialog
        
        """
        dialog = dialogs.CutEarthquakesDialog()
        dialog.exec_()
    
    def about_dialog(self):
        dialog = dialogs.AboutDialog()
        dialog.exec_()
        