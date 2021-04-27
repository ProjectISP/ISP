# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:45:54 2020

@author: olivar

Rfun, a toolbox for the analysis of teleseismic receiver functions
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

# isp imports
from isp.Gui.Frames import BaseFrame
import isp.receiverfunctions.rf_dialogs as dialogs
import isp.receiverfunctions.rf_main_window_utils as mwu
from isp.Gui.Frames.uis_frames import UiReceiverFunctions
from isp.Gui.Frames.help_frame import HelpDoc
# other imports
import io
import os
import math
import pickle
import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
import numpy as np
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.backend_bases
import matplotlib.patches as mpatches
from owslib.wms import WebMapService
from PyQt5 import QtWidgets

class RecfFrame(BaseFrame, UiReceiverFunctions):
    
    def __init__(self):
        super(RecfFrame, self).__init__()
        self.setupUi(self)
        
        # RF analysis-related attributes
        self.data_map = {}
        self.arrivals = {}
        self.srcfs = {}
        self.rfs = []
        self.rf_current_page = 1
        self.rf_pages = 1

        self.first_rf_stack_plot = True # To be removed
        self.first_hk_stack_plot = True # To be removed
        self.hk_result = {}
        
        # Theoretical arrival times obtained from the maximum of the H-k
        self.Ps = 0
        self.PpPs = 0
        self.PpSs_PsPs = 0
        
        # CCP stacking-related attributes
        self.ccp_grid = {'x0':None, 'y0':None, 'x1':None, 'y1':None}
        self.ccp_cross_section = {'x0':None, 'y0':None, 'x1':None, 'y1':None}
        self.ccp_stack_map_plot_mode = None
        self.button_pressed = False
        self.ccp_grid_mpl_line = None
        self.ccp_cross_section_mpl_line = None
        self.rfs_dicts = None
        self.stack = None
        self.istack = None
        self.depth_array = None
        self.first_ccp_stack_plot = True
        self.mplwidget_5_basemap = None
        self.mplwidget_5_gridlines = None
        self.mplwidget_5_ccp_pcolormesh = None
        
        # Connect GUI elements
        self.connect_rf_analysis_gui_elements()
        self.connect_ccp_stack_gui_elements()
    
    def connect_rf_analysis_gui_elements(self):
        # Menu actions
        self.actionRead_waveforms.triggered.connect(self.read_waveforms)
        self.actionRead_metadata.triggered.connect(self.read_metadata)
        self.actionCompute_source_functions.triggered.connect(self.compute_srcfs)
        self.actionCut_earthquakes_from_raw_data.triggered.connect(self.cut_earthquakes_dialog)
        self.actionAbout_2.triggered.connect(self.about_dialog)
        # Pushbuttons
        self.pushButton.clicked.connect(self.compute_rfs)
        self.pushButton_2.clicked.connect(self.save_rfs)
        self.pushButton_7.clicked.connect(self.previous_rfs_page)
        self.pushButton_8.clicked.connect(self.next_rfs_page)
        self.pushButton_3.clicked.connect(self.plot_rf_stack)
        self.pushButton_5.clicked.connect(self.plot_hk_stack)
        self.pushButton_9.clicked.connect(self.plot_map)
        self.pushButton_4.clicked.connect(partial(self.save_figure_dialog, "stack_figure"))
        self.pushButton_15.clicked.connect(partial(self.save_figure_dialog, "hk_stack_figure"))
        self.pushButton_10.clicked.connect(partial(self.save_figure_dialog, "earthquakes_map_figure"))
        self.pushButton_6.clicked.connect(self.save_hk_result)
        
        self.pushButton_17.clicked.connect(self.filter_by_magnitude)
        self.pushButton_16.clicked.connect(self.remove_rejections)
        # Combobox changes
        self.comboBox_3.currentTextChanged.connect(self.plot_rfs)
        # mplwidgets
        self.mplwidget.figure.canvas.mpl_connect('button_press_event', self.recf_plot_clicked)
        
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
        self.pushButton_13.clicked.connect(partial(self.save_figure_dialog, "ccp_stack_figure"))
        self.pushButton_26.clicked.connect(self.cross_section_dialog)

        self.doubleSpinBox_21.valueChanged.connect(self.change_ccp_map_extent)
        self.doubleSpinBox_22.valueChanged.connect(self.change_ccp_map_extent)
        self.doubleSpinBox_23.valueChanged.connect(self.change_ccp_map_extent)
        self.doubleSpinBox_24.valueChanged.connect(self.change_ccp_map_extent)
        self.doubleSpinBox_25.valueChanged.connect(self.plot_ccp_stack)
        
        #♠self.doubleSpinBox_26.valueChanged.connect(self.plot_ccp_stack)
        
        self.comboBox_6.currentTextChanged.connect(self.plot_ccp_basemap)
        self.spinBox_5.valueChanged.connect(self.plot_ccp_stack)
        self.doubleSpinBox_26.valueChanged.connect(self.ccp_map_axes_gridlines)
        self.spinBox_6.valueChanged.connect(self.update_ccp_tile_map)
        
    
    def read_waveforms(self):
        """Map the mseed files inside the given directory, return a dict
        
        """
        dir_ = QtWidgets.QFileDialog.getExistingDirectory()
        
        if dir_:
            self.data_map = mwu.map_earthquakes(eq_dir=dir_)
            
            # Populate the station combobox
            for stnm in sorted(list(self.data_map.keys())):
                self.comboBox.addItem(stnm)
    
    def read_metadata(self):
        """Read the event data file
        
        """
        dir_ = QtWidgets.QFileDialog.getOpenFileName()[0]
        
        if dir_:
            self.arrivals = pickle.load(open(dir_, 'rb'))
            
            # Enable the other menu actions
            self.actionRead_waveforms.setEnabled(True)
            self.actionRead_RFs.setEnabled(True)
            self.actionRead_CCP_Stack.setEnabled(True)
    
    def compute_srcfs(self):
        """Compute source functions by averaging the L (or Z) component of all
        stations in the array
        
        """
        self.srcfs = mwu.compute_source_functions(self.data_map,
                                                  corner_freqs=(self.doubleSpinBox_27.value(),
                                                                self.doubleSpinBox_28.value())
                                                  )
    
    def compute_rfs(self):
        """Compute and plot selected receiver functions
        
        """
        self.rf_current_page = 1
        stnm = self.comboBox.currentText()
        a = self.doubleSpinBox.value()
        c = self.doubleSpinBox_2.value()
        
        # Preserve rejections of RFs through changes in the processing parameters
        rejections = [(x[5], x[6]) for x in self.rfs] # pos 5 is accept/reject, pos 6 is event ID
        
        # Compute new rfs
        self.rfs = mwu.compute_rfs(stnm, self.data_map, self.arrivals,
                                     srfs=self.srcfs, a=a, c=c,
                                     corner_freqs=(self.doubleSpinBox_27.value(),
                                                   self.doubleSpinBox_28.value()))
        
        # Restore the previous accepted/rejected state
        sorted_rej = sorted(rejections, key=lambda x: x[1])
        if len(rejections) == len(self.rfs):
            for i, rf in enumerate(sorted(self.rfs, key=lambda x: x[6])):
                if sorted_rej[i][0] == 0:
                    rf[5] = 0

        self.rf_pages = int(math.ceil(len(self.rfs)/7))
        self.label_14.setText("{}/{}".format(self.rf_current_page, self.rf_pages))
        self.plot_rfs()
    
    def setup_rf_axes(self):
        """Prepare receiver function axes for plotting
        
        """
        self.mplwidget.figure.clf()
        gs = gridspec.GridSpec(7, 1)
        gs.update(left=0.10, right=0.95, top=0.98, bottom=0.085, hspace=0.35)
        for i in range(7):
            self.mplwidget.figure.add_subplot(gs[i])
            
        for i in range(6):
            self.mplwidget.figure.axes[i].set_xticklabels([])
            self.mplwidget.figure.axes[6].set_xlabel("Time in seconds")

    def plot_rfs(self):
        """ Sort and plot receiver functions according to user selection
        
        """
        if self.comboBox_3.currentText() == "Back az.":
            sort_index = 2
        elif self.comboBox_3.currentText() == "Distance":
            sort_index = 3
        
        self.rfs = sorted(self.rfs, key=lambda x: x[sort_index])
        
        self.setup_rf_axes()
        
        xmin = min([self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value()])
        xmax = max([self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value()])
            
        rf_index = 0 + (self.rf_current_page - 1)*7
        j = np.minimum(7, len(self.rfs[rf_index:]))
        for i in range(j):
            rf = self.rfs[rf_index+i][0]
            t = self.rfs[rf_index+i][1]
            text = np.round(self.rfs[rf_index+i][sort_index], decimals=0)
            evid = self.rfs[rf_index+i][6]
            
            self.mplwidget.figure.axes[i].fill_between(t, rf, color='black',
                                                       linewidth=0.5)            
            self.mplwidget.figure.axes[i].fill_between(t, np.zeros(len(t)),
                                                       rf, where=(rf > 0), color='black')
            self.mplwidget.figure.axes[i].fill_between(t, np.zeros(len(t)),
                                                       rf, where=(rf < 0), color='red')
            
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
        rf_index = (self.rf_current_page - 1)*7 + axind
        
        if event.button == matplotlib.backend_bases.MouseButton(1):  # Left click discards rf
            if self.rfs[rf_index][5]:
                self.rfs[rf_index][5] = 0
            else:
                self.rfs[rf_index][5] = 1
            self.plot_rfs()
        elif event.button == matplotlib.backend_bases.MouseButton(3): # Right click shows earthquake
            rf = self.rfs[rf_index]
            file = rf[8]
            bandpass = (0.03, 1.50)
            dialog = dialogs.ShowEarthquakeDialog(file, bandpass)
            dialog.exec_()

    def hk_plot_clicked(self, event):
        """Register click events in the hk panel
        
        """
        pass

    
    def filter_by_magnitude(self):
        for rf in self.rfs:
            if rf[7] < self.doubleSpinBox_29.value():
                rf[5] = 0
            else:
                rf[5] = 1
        
        self.plot_rfs()
    
    def remove_rejections(self):
        for rf in self.rfs:
            if rf[5] == 0:
                rf[5] = 1
        
        self.plot_rfs()
    
    def save_rfs(self):
        """Save the current receiver functions to disk
        
        """
        outdir = QtWidgets.QFileDialog.getExistingDirectory()
        stnm = self.comboBox.currentText()
        a = self.doubleSpinBox.value()
        c = self.doubleSpinBox_2.value()
        mwu.save_rfs(stnm, a, c, self.rfs, outdir=outdir)

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
        stack, bin_stacks, bins, ymin, ymax = mwu.compute_stack(self.rfs, bin_size=self.spinBox.value(),
                                                                  overlap=self.spinBox_2.value(),
                                                                  stack_by=self.comboBox_2.currentText(),
                                                                  moveout_phase=self.comboBox_4.currentText())
        self.setup_rf_stack_axes()
        
        t = self.rfs[0][1]
        
        self.mplwidget_2.figure.axes[0].plot(t, stack, color='black', linewidth=0.5)
        self.mplwidget_2.figure.axes[0].fill_between(t, 0, stack,
                                                     where=(stack > 0),
                                                     color='black')
        self.mplwidget_2.figure.axes[0].fill_between(t, 0, stack,
                                                     where=(stack < 0),
                                                     color='red')
        zorder = len(bins)
        for i, b in enumerate(bins):
            bstack = bin_stacks[:,i]+b
            height = np.zeros(len(t)) + b
            self.mplwidget_2.figure.axes[1].plot(t, bstack, color='black', linewidth=0.5, zorder=zorder)
            self.mplwidget_2.figure.axes[1].fill_between(t, height, bstack,
                                                         where=(bstack > b),
                                                         color='black', zorder=zorder)
            self.mplwidget_2.figure.axes[1].fill_between(t, height, bstack,
                                                         where=(bstack < b),
                                                         color='red', zorder=zorder)
        
            zorder -= 1
        
        xmin = min([self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value()])
        xmax = max([self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value()])

        self.mplwidget_2.figure.axes[0].set_xlim(xmin, xmax)

        self.mplwidget_2.figure.axes[1].set_ylim(ymin, ymax)
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
        Hvalues = self.spinBox_3.value()
        
        mink = min([self.doubleSpinBox_3.value(), self.doubleSpinBox_4.value()])
        maxk = max([self.doubleSpinBox_3.value(), self.doubleSpinBox_4.value()])
        kvalues = self.spinBox_4.value()
        
        w1 = self.doubleSpinBox_30.value()
        w2 = self.doubleSpinBox_31.value()
        w3 = self.doubleSpinBox_32.value()
        
        H_arr, k_arr, matrix, H, k, events = mwu.compute_hk_stack(self.rfs, H_range=(minH, maxH), H_values=Hvalues,
                                                                  k_range=(mink, maxk), k_values=kvalues,
                                                                  w1=w1, w2=w2, w3=w3)
        
        self.Ps, self.PpPs, self.PpSs_PsPs = mwu.compute_theoretical_arrival_times(H, k)

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
        self.mplwidget_3.figure.axes[0].contourf(H_arr, k_arr,  matrix/np.max(np.abs(matrix)), levels=100, vmin=0, vmax=np.max(matrix/np.max(np.abs(matrix))), cmap='viridis')
        self.mplwidget_3.figure.axes[0].contour(H_arr, k_arr, matrix, levels=[np.max(matrix) - error_contour_level], colors=["green"])
        self.mplwidget_3.figure.axes[0].set_xlim(minH, maxH)
        self.mplwidget_3.figure.axes[0].set_ylim(mink, maxk)
        if error_area != None:
            self.mplwidget_3.figure.axes[0].axvline(H, color="white")
            self.mplwidget_3.figure.axes[0].axhline(k, color="white")
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
        if not os.path.exists(txt_fname):
            with open(txt_fname, "w", newline='\n') as f:
                f.write("STATION,LONG,LAT,EVENTS,H,MIN_H95,MAX_H95,k,MIN_k95,MAX_k95" + '\n')

        stla = self.arrivals['stations'][stnm]["lat"]
        stlo = self.arrivals['stations'][stnm]["lon"]
        
        line = stnm + "," + str(stlo) + "," + str(stla) + "," + str(self.hk_result['events']) + "," + str(self.hk_result['H']) + "," + str(self.hk_result['H_95'][0]) + "," + str(self.hk_result['H_95'][1]) \
               + "," + str(self.hk_result['k']) + "," + str(self.hk_result['k_95'][0]) + "," + str(self.hk_result['k_95'][1])
        with open(txt_fname, "a", newline='\n') as f:
            f.write(line + '\n')
        
        pickle_output_path = os.sep.join(txt_fname.split(os.sep)[:-1])
        pickle_fname = stnm + "_Hk" + ".pickle"
        pickle.dump(self.hk_result, open(os.path.join(pickle_output_path, pickle_fname), "wb"))

    def setup_map_axes(self):
        """Prepare map axes for plotting
        
        """
        self.mplwidget_4.figure.clf()
        
        stnm = self.comboBox.currentText()
        lat = self.arrivals['stations'][stnm]["lat"]
        lon = self.arrivals['stations'][stnm]["lon"]
        proj = ccrs.Stereographic(central_longitude=lon, central_latitude=lat)
        

        self.mplwidget_4.figure.subplots(1, subplot_kw=dict(projection=proj))
        self.mplwidget_4.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        r1 = mwu.compute_radius(proj, lat, lon, 30)
        r2 = mwu.compute_radius(proj, lat, lon, 90)
        pad_radius = mwu.compute_radius(proj, lat, lon, 90 + 5)
        self.mplwidget_4.figure.axes[0].set_xlim([-pad_radius, pad_radius])
        self.mplwidget_4.figure.axes[0].set_ylim([-pad_radius, pad_radius])
        
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
        
        for rf in self.rfs:
            if rf[5]:
                event_id = rf[6]
                lat = self.arrivals['events'][event_id]['event_info']['latitude']
                lon = self.arrivals['events'][event_id]['event_info']['longitude']
                self.mplwidget_4.figure.axes[0].plot(lon, lat, marker='o', color='red',
                                                     transform=ccrs.Geodetic(), markersize=1.5)
        
        self.mplwidget_4.figure.canvas.draw()
    
    def setup_ccp_map_axes(self):
        """Prepare ccp map axes for plotting
        
        """
        if self.first_ccp_stack_plot:
            self.mplwidget_5.figure.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
            self.mplwidget_5.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            self.ccp_map_axes_gridlines()
            self.first_ccp_stack_plot = False
        else:
            self.mplwidget_5.figure.axes[0].clear()
            self.ccp_grid_mpl_line = None
            self.ccp_cross_section_mpl_line = None
    
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
            line = self.ccp_grid_mpl_line
        elif self.pushButton_25.isChecked():
            mode = "cross_section"
            dict_ = self.ccp_cross_section
            line = self.ccp_cross_section_mpl_line
        else:
            return

        if event.name == "button_press_event":
            dict_['x0'] = event.xdata
            dict_['y0'] = event.ydata
            if mode == "grid":
                if self.ccp_grid_mpl_line != None:
                    self.mplwidget_5.figure.axes[0].lines.remove(self.ccp_grid_mpl_line)
                self.ccp_grid_mpl_line, = self.mplwidget_5.figure.axes[0].plot(event.xdata, event.ydata, color="red")
            elif mode == "cross_section":
                if self.ccp_cross_section_mpl_line != None:
                    self.mplwidget_5.figure.axes[0].lines.remove(self.ccp_cross_section_mpl_line)
                self.ccp_cross_section_mpl_line, = self.mplwidget_5.figure.axes[0].plot(event.xdata, event.ydata, color="red")                
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
                    
                elif mode == "cross_section":
                    x = [dict_['x0'], dict_['x1']]
                    y = [dict_['y0'], dict_['y1']]
                    
                    self.ccp_cross_section_mpl_line.set_xdata(x)
                    self.ccp_cross_section_mpl_line.set_ydata(y)
                
                self.mplwidget_5.figure.canvas.draw()
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
            self.doubleSpinBox_19.setValue(end[0])
            self.doubleSpinBox_18.setValue(end[1])
    
    def ccp_stack_read_rfs(self):
        """Read receiver functions and plot stations on the map
        
        """
        dir_ = QtWidgets.QFileDialog.getExistingDirectory()
        self.rfs_dicts = mwu.map_rfs(rfs_dir=dir_)
        
        self.setup_ccp_map_axes()
        self.plot_ccp_stations()
        self.change_ccp_map_extent()
        
        self.mplwidget_5.figure.canvas.draw()

    def compute_ccp_stack(self):
        """Compute and draw CCP stack
        
        """
        dz = self.doubleSpinBox_15.value()
        max_depth = self.doubleSpinBox_16.value()
        dlat = self.doubleSpinBox_14.value()
        dlon = self.doubleSpinBox_13.value()
        
        self.stack, self.depth_array = mwu.ccp_stack(self.rfs_dicts, self.arrivals,
                                                     min([self.doubleSpinBox_11.value(), self.doubleSpinBox_12.value()]),
                                                     max([self.doubleSpinBox_11.value(), self.doubleSpinBox_12.value()]),
                                                     min([self.doubleSpinBox_9.value(), self.doubleSpinBox_10.value()]),
                                                     max([self.doubleSpinBox_9.value(), self.doubleSpinBox_10.value()]),
                                                     dlon,
                                                     dlat,
                                                     dz,
                                                     max_depth)
        
        self.istack = None
        
        self.stack_x = np.arange(min([self.doubleSpinBox_11.value(), self.doubleSpinBox_12.value()]),
                      max([self.doubleSpinBox_11.value(), self.doubleSpinBox_12.value()]), dlon)
        self.stack_y = np.arange(min([self.doubleSpinBox_9.value(), self.doubleSpinBox_10.value()]),
                      max([self.doubleSpinBox_9.value(), self.doubleSpinBox_10.value()]), dlat)
        
        self.plot_ccp_stack()
        
        self.doubleSpinBox_25.setMaximum(max_depth-dz)
        self.doubleSpinBox_25.setSingleStep(dz)        
        
    def plot_ccp_stations(self):
        lats, lons = [], []
        for stnm in self.rfs_dicts.keys():
            stla = self.arrivals['stations'][stnm]["lat"]
            stlo = self.arrivals['stations'][stnm]["lon"]
            lats.append(stla)
            lons.append(stlo)
            self.mplwidget_5.figure.axes[0].plot(stlo, stla, marker="^",
                                                 transform=ccrs.Geodetic(), color="green")        
        self.doubleSpinBox_21.setValue(min(lats) - 0.5)
        self.doubleSpinBox_22.setValue(max(lats) + 0.5)
        self.doubleSpinBox_23.setValue(min(lons) - 0.5)
        self.doubleSpinBox_24.setValue(max(lons) + 0.5)
    
    def plot_ccp_basemap(self):
        # Updates to cartopy objects are never shown on already displayed
        # axes, so we create a new axis for the basemaps, to be placed behind
        # the data axes
        if self.first_ccp_stack_plot:
                    return

        if self.mplwidget_5_basemap != None:
            #Remove previous basemap if necessary
            self.mplwidget_5_basemap.remove()

        if self.comboBox_6.currentText() == "None":
            self.mplwidget_5_basemap = None
            self.mplwidget_5.figure.axes[0].background_patch.set_facecolor((1, 1, 1, 1))
        else:
            self.mplwidget_5_basemap = self.mplwidget_5.figure.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
            self.mplwidget_5.figure.axes[0].set_zorder(1)
            self.mplwidget_5.figure.axes[0].background_patch.set_facecolor((1, 1, 1, 0))
            self.mplwidget_5_basemap.set_zorder(0)
            self.mplwidget_5_basemap.set_extent(self.mplwidget_5.figure.axes[0].get_extent())
        
        if self.comboBox_6.currentText() == "GEBCO":
            MAP_SERVICE_URL = 'https://www.gebco.net/data_and_products/gebco_web_services/2019/mapserv?'
            wms = WebMapService(MAP_SERVICE_URL)
            layer = 'GEBCO_2019_Grid'
            self.mplwidget_5_basemap.add_wms(wms, layer)
        elif self.comboBox_6.currentText() == "Stamen":
            self.mplwidget_5_basemap.add_image(Stamen('terrain-background'), self.spinBox_6.value())
        
        self.mplwidget_5.figure.canvas.draw()
    
    def update_ccp_tile_map(self):
        if self.comboBox_6.currentText() == "Stamen":
            self.plot_ccp_basemap()

    def ccp_map_axes_gridlines(self):
        """Plots grid lines and labels according to the user specified
        spacing (value of doubleSpinBox_26). If spacing = 0, no grid lines are
        drawn.

        """
        if self.first_ccp_stack_plot:
            return
        
        spacing = self.doubleSpinBox_26.value()
        
        if self.mplwidget_5_gridlines != None:
            # Remove previous gridlines if necessary
            self.mplwidget_5_gridlines.remove()
            
        if spacing == 0:
            return
        
        # Updates to cartopy objects are never shown on already displayed
        # axes, so we create a new axis for the basemaps, to be placed on top
        # the data axes
        self.mplwidget_5_gridlines = self.mplwidget_5.figure.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
        self.mplwidget_5_gridlines.set_zorder(2)
        self.mplwidget_5_gridlines.background_patch.set_facecolor((1, 1, 1, 0))
        extent = self.mplwidget_5.figure.axes[0].get_extent()
        self.mplwidget_5_gridlines.set_extent(extent)
        self.ccp_map_gridlines = self.mplwidget_5_gridlines.gridlines(draw_labels=True)
        xticks = np.arange(extent[0], extent[1] + spacing, spacing)
        yticks = np.arange(extent[2], extent[3] + spacing, spacing)
        self.ccp_map_gridlines.xlocator = mticker.FixedLocator(xticks)
        self.ccp_map_gridlines.ylocator = mticker.FixedLocator(yticks)
        
        self.mplwidget_5.figure.canvas.draw()
            

    def plot_ccp_stack(self):
        if self.first_ccp_stack_plot:
            return
        
        try:
            self.mplwidget_5_ccp_pcolormesh.remove()
        except AttributeError:
            pass

        alpha = self.spinBox_5.value()/100
        hslice = np.where(self.depth_array == self.doubleSpinBox_25.value())[0][0]
        self.mplwidget_5_ccp_pcolormesh = self.mplwidget_5.figure.axes[0].pcolormesh(self.stack_x, self.stack_y, self.stack[:,:,hslice].T,
                                                                                     alpha=alpha, transform=ccrs.PlateCarree(), cmap="RdBu")
        self.first_ccp_stack_plot = False
        self.mplwidget_5.figure.canvas.draw()
    
    def save_ccp_stack(self):
        if type(self.stack) == type(None):
            return
        
        fname = QtWidgets.QFileDialog.getSaveFileName()[0]
        
        if fname:
            ccp_dict = {"ccp_stack":self.stack,
                        "x_array":self.stack_x,
                        "y_array":self.stack_y,
                        "z_array":self.depth_array}
    
            
            pickle.dump(ccp_dict, open(fname, "wb"))
    
    def read_ccp_stack(self):
        fname = QtWidgets.QFileDialog.getOpenFileName()[0]
        
        if fname:
            ccp_dict = pickle.load(open(fname, 'rb'))
            self.stack = ccp_dict["ccp_stack"]
            self.stack_x = ccp_dict["x_array"]
            self.stack_y = ccp_dict["y_array"]
            self.depth_array = ccp_dict["z_array"]
            
            self.doubleSpinBox_25.setMaximum(np.max(self.depth_array))
            self.doubleSpinBox_25.setSingleStep(np.diff(self.depth_array)[0])
            self.plot_ccp_stack()
    
    def change_ccp_map_extent(self):
        if not self.first_ccp_stack_plot:
            miny = min([self.doubleSpinBox_21.value(), self.doubleSpinBox_22.value()])
            maxy = max([self.doubleSpinBox_21.value(), self.doubleSpinBox_22.value()])
            minx = min([self.doubleSpinBox_23.value(), self.doubleSpinBox_24.value()])
            maxx = max([self.doubleSpinBox_23.value(), self.doubleSpinBox_24.value()])
            self.mplwidget_5.figure.axes[0].set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
            
            # A secondary axis containing a tile map may exist, if so, we need to adjust its extent
            # too:
            if len(self.mplwidget_5.figure.axes) > 1:
                self.mplwidget_5.figure.axes[1].set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

            self.ccp_map_axes_gridlines()

    def about_dialog(self):
        dialog = dialogs.AboutDialog()
        dialog.exec_()

    def cross_section_dialog(self):
        
        if self.istack == None:
            self.istack = mwu.interpolate_ccp_stack(self.stack_x, self.stack_y, self.stack)
        
        start = (self.doubleSpinBox_20.value(), self.doubleSpinBox_17.value())
        end = (self.doubleSpinBox_18.value(), self.doubleSpinBox_19.value())
        newlats, newlons, dist_arr = mwu.compute_intermediate_points(start, end, self.spinBox_9.value())

        matrix = []
        for i, stack in enumerate(self.istack):
            row = []
            for lat, lon in zip(newlats, newlons):
                row.append(stack(lat, lon)[0])
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        dialog = dialogs.CrossSectionDialog(dist_arr, -self.depth_array, matrix[:, 1:-1], start, end)
        dialog.exec_()
        
    def cut_earthquakes_dialog(self):
        """Display the Cut earthquakes from local data dialog
        
        """
        dialog = dialogs.CutEarthquakesDialog()
        dialog.exec_()

    def save_figure_dialog(self, figure):
        """Display the save figure dialog
        
        """

        if figure == "stack_figure":
            fig = self.mplwidget_2.figure
            preferred_size = (8, 5.5)
            preferred_margins = (0.10, 0.95, 0.07, 0.92)
            preferred_xlabel = "Time (s)"
            if self.comboBox_2.currentText() == "Back az.":
                preferred_ylabel = "Back azimuth (" + r"$\degree$" + ")"
            elif self.comboBox_2.currentText() == "Distance":
                preferred_ylabel = "Distance (" + r"$\degree$" + ")"
            preferred_title = "RF stack for {} (bin size {}, overlap {})".format(self.comboBox.currentText(),
                                                                                 self.spinBox.value(),
                                                                                 self.spinBox_2.value())
            preferred_fname = "{}_stack".format(self.comboBox.currentText())
        elif figure == "hk_stack_figure":
            fig = self.mplwidget_3.figure
            preferred_size = (4, 5.5)
            preferred_margins = (0.10, 0.95, 0.10, 0.92)
            preferred_xlabel = "H (km)"
            preferred_ylabel = "k"
            preferred_title = "H-k stack for station {} ({} events)".format(self.comboBox.currentText(),self.hk_result['events'])
            preferred_fname = "{}_hk".format(self.comboBox.currentText())
        elif figure == "earthquakes_map_figure":
            fig = self.mplwidget_4.figure
            fig.savefig("{}_events.png".format(self.comboBox.currentText()), dpi=600)
            # Figure must be recreated with the desired size because otherwise the dots and circles are displaced            
            stnm = self.comboBox.currentText()
            lat = self.arrivals['stations'][stnm]["lat"]
            lon = self.arrivals['stations'][stnm]["lon"]
            proj = ccrs.Stereographic(central_longitude=lon, central_latitude=lat)

            fig = plt.figure(figsize=(4, 5.5))
            ax = fig.add_subplot(111, projection=proj)
            r1 = mwu.compute_radius(proj, lat, lon, 30)
            r2 = mwu.compute_radius(proj, lat, lon, 90)
            pad_radius = mwu.compute_radius(proj, lat, lon, 90 + 5)
            ax.set_xlim([-pad_radius, pad_radius])
            ax.set_ylim([-pad_radius, pad_radius])
            
            ax.text(r1, -r1, r"$30\degree$", fontsize=9)
            ax.text(0.8*r2, -r2*0.8, r"$90\degree$", fontsize=9)
    
            ax.add_patch(mpatches.Circle(xy=[lon, lat], radius=r1, fill=False, edgecolor='darkblue', alpha=1, transform=proj, zorder=30))
            ax.add_patch(mpatches.Circle(xy=[lon, lat], radius=r2, fill=False, edgecolor='darkblue', alpha=1, transform=proj, zorder=30))
            ax.coastlines()
        
            
            for rf in self.rfs:
                if rf[5]:
                    event_id = rf[6]
                    lat = self.arrivals['events'][event_id]['event_info']['latitude']
                    lon = self.arrivals['events'][event_id]['event_info']['longitude']
                    ax.plot(lon, lat, marker='o', color='red',
                                                         transform=ccrs.Geodetic(), markersize=1.5)
            
            fig.savefig("{}_events.png".format(self.comboBox.currentText()), dpi=600)
            fig.close()
            
        elif figure == "ccp_stack_figure":
            # ccp stack figure must be recreated as the patched pcolormesh
            # used by cartopy cannot be pickled
            fig = plt.figure()
            fig.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
            fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            
            lats, lons = [], []
            for stnm in self.rfs_dicts.keys():
                stla = self.arrivals['stations'][stnm]["lat"]
                stlo = self.arrivals['stations'][stnm]["lon"]
                lats.append(stla)
                lons.append(stlo)
                fig.axes[0].plot(stlo, stla, marker="^", transform=ccrs.Geodetic(), color="green")
            
            if self.stack != None:

                hslice = np.where(self.depth_array == self.doubleSpinBox_25.value())[0][0]
                fig.axes[0].pcolormesh(self.stack_x, self.stack_y, self.stack[:,:,hslice].T, transform=ccrs.PlateCarree(), cmap="RdBu",
                                       alpha=self.spinBox_5.value()/100)
                fig.axes[0].plot([self.ccp_cross_section['x0'], self.ccp_cross_section['x1']],
                                 [self.ccp_cross_section['y0'], self.ccp_cross_section['y1']], color="red")

            if self.comboBox_6.currentText() == "GEBCO":
                MAP_SERVICE_URL = 'https://www.gebco.net/data_and_products/gebco_web_services/2019/mapserv?'
                wms = WebMapService(MAP_SERVICE_URL)
                layer = 'GEBCO_2019_Grid'
                fig.axes[0].add_wms(wms, layer)
            elif self.comboBox_6.currentText() == "Stamen":
                fig.axes[0].add_image(Stamen('terrain-background'), self.spinBox_6.value())

            spacing = self.doubleSpinBox_26.value()
            if spacing > 0:
                extent = self.mplwidget_5.figure.axes[0].get_extent()
                fig.axes[0].set_extent(extent)
                fig.axes[0].gridlines(draw_labels=True)
                xticks = np.arange(extent[0], extent[1] + spacing, spacing)
                yticks = np.arange(extent[2], extent[3] + spacing, spacing)
                fig.axes[0].xlocator = mticker.FixedLocator(xticks)
                fig.axes[0].ylocator = mticker.FixedLocator(yticks)

            # Default figure size, margins, and labels
            preferred_size = (4, 5.5)
            preferred_margins = (0.10, 0.95, 0.10, 0.92)
            preferred_xlabel = "H (km)"
            preferred_ylabel = "k"
            preferred_title = "H-k stack for station {} ({} events)".format(self.comboBox.currentText(),self.hk_result['events'])
            
            dialog = dialogs.SaveFigureDialog(fig, preferred_size, preferred_margins, preferred_title,
                                              preferred_xlabel, preferred_ylabel)
            dialog.exec_()
            return
        
        # We do not want to modify the original figure on the main frame of
        # the program. But figure objects cannot be copied, not even with
        # copy.deepcopy(). Therefore we pickle the original figure and load it
        # again as a new one
        
        buffer = io.BytesIO()
        pickle.dump(fig, buffer)
        # Point to the first byte of the buffer and read it
        buffer.seek(0)
        fig_copy = pickle.load(buffer)
        
        #We also need a new canvas manager
        newfig = plt.figure()
        newmanager = newfig.canvas.manager
        newmanager.canvas.figure = fig_copy
        fig_copy.set_canvas(newmanager.canvas)        
        
        dialog = dialogs.SaveFigureDialog(fig_copy, preferred_size, preferred_margins, preferred_title,
                                          preferred_xlabel, preferred_ylabel, preferred_fname)
        dialog.exec_()
        return

    def open_help(self):
        self.help.show()