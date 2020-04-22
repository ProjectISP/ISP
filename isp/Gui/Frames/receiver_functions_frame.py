# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:45:54 2020

@author: olivar
"""
from isp.Gui.Frames import BaseFrame
from isp.Gui.Frames.uis_frames import UiReceiverFunctions
from isp.Gui import pyqt, pqg, pw, pyc, qt
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
import numpy as np
import pickle
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from functools import partial
import isp.receiverfunctions.dialogs as dialogs
import isp.receiverfunctions.main_window_utils as utils


class RecfFrame(BaseFrame, UiReceiverFunctions):
    
    def __init__(self):
        super(RecfFrame, self).__init__()
        self.setupUi(self)
        
        # Global variables ############ RF ANALYSIS ##########################
        self.data_map = {}
        self.arrivals = {}
        self.srcfs = {}
        self.rfs = []
        self.rf_current_page = 1
        self.rf_pages = 1
        self.first_rf_plot = True
        self.first_rf_stack_plot = True
        self.first_hk_stack_plot = True
        self.first_map_plot = True
        
        # Global variables ############ CCP STACKING #########################
        self.ccp_grid = {'x0':None, 'y0':None, 'x1':None, 'y1':None}
        self.ccp_cross_section = {'x0':None, 'y0':None, 'x1':None, 'y1':None}
        self.ccp_stack_map_plot_mode = None
        self.button_pressed = False
        self.ccp_grid_mpl_line = None
        self.ccp_cross_section_mpl_line = None
        self.rfs_dicts = None
        self.stack = None
        self.istack = None
        
        # Connect GUI elements to functions ############ RF ANALYSIS
        
        # Menu actions
        self.actionRead_waveforms.triggered.connect(self.read_waveforms)
        self.actionRead_metadata.triggered.connect(self.read_metadata)
        self.actionCompute_source_functions.triggered.connect(self.compute_srcfs)
        self.actionCut_earthquakes_from_raw_data.triggered.connect(self.cut_earthquakes_dialog)
        
        # Pushbuttons
        self.pushButton.clicked.connect(self.compute_rfs)
        self.pushButton_2.clicked.connect(self.save_rfs)
        
        self.pushButton_7.clicked.connect(self.previous_rfs_page)
        self.pushButton_8.clicked.connect(self.next_rfs_page)
        self.pushButton_3.clicked.connect(self.plot_rf_stack)
        self.pushButton_5.clicked.connect(self.plot_hk_stack)
        self.pushButton_9.clicked.connect(self.plot_map)
        self.pushButton_26.clicked.connect(self.cross_section)
        
        # Combobox changes
        self.comboBox_3.currentTextChanged.connect(self.plot_rfs)
        
        # mplwidgets
        self.mplwidget.figure.canvas.mpl_connect('button_press_event', self.recf_plot_clicked)
        
        # Connect GUI elements to functions ############ CCP STACKING
        self.actionRead_RFs.triggered.connect(self.ccp_stack_read_rfs)
        
        
        self.mplwidget_5.figure.canvas.mpl_connect('button_press_event', self.ccp_stack_map_event_handler)
        self.mplwidget_5.figure.canvas.mpl_connect('button_release_event', self.ccp_stack_map_event_handler)
        self.mplwidget_5.figure.canvas.mpl_connect('motion_notify_event', self.ccp_stack_map_event_handler)
        
        self.pushButton_11.clicked.connect(partial(self.ccp_stack_map_toggle_plot_mode, "grid"))
        self.pushButton_25.clicked.connect(partial(self.ccp_stack_map_toggle_plot_mode, "cross_section"))
        self.pushButton_12.clicked.connect(self.compute_ccp_stack)
    
    def read_waveforms(self):
        # Map the mseed files inside the given directory, return a dict
        dir_ = pw.QFileDialog.getExistingDirectory()
        self.data_map = utils.map_earthquakes(eq_dir=dir_)
        
        # Populate the station combobox
        for stnm in sorted(list(self.data_map.keys())):
            self.comboBox.addItem(stnm)
    
    def read_metadata(self):
        dir_ = pw.QFileDialog.getOpenFileName()[0]
        self.arrivals = pickle.load(open(dir_, 'rb'))
    
    def compute_srcfs(self):
        self.srcfs = utils.compute_source_functions(self.data_map)
    
    def compute_rfs(self):
        self.rf_current_page = 1
        stnm = self.comboBox.currentText()
        a = self.doubleSpinBox.value()
        c = self.doubleSpinBox_2.value()
        self.rfs = utils.compute_rfs(stnm, self.data_map, self.arrivals,
                                     srfs=self.srcfs, a=a, c=c)
        self.rf_pages = int(math.ceil(len(self.rfs)/7))
        self.label_14.setText("{}/{}".format(self.rf_current_page, self.rf_pages))
        self.plot_rfs()
    
    def setup_rf_axes(self):
        if self.first_rf_plot:
            gs = gridspec.GridSpec(7, 1)
            gs.update(left=0.10, right=0.95, top=0.95, bottom=0.075, hspace=0.35)
            for i in range(7):
                self.mplwidget.figure.add_subplot(gs[i])
            self.first_rf_plot = False
        else:
            for i in range(7):
                self.mplwidget.figure.axes[i].clear()
            
        for i in range(6):
            self.mplwidget.figure.axes[i].set_xticklabels([])
            self.mplwidget.figure.axes[6].set_xlabel("Time in seconds")
            self.mplwidget.figure.suptitle("Receiver functions", y=0.985)
            
    
    def plot_rfs(self):
        if self.comboBox_3.currentText() == "Back az.":
            sort_index = 2
        elif self.comboBox_3.currentText() == "Distance":
            sort_index = 3
        elif self.comboBox_3.currentText() == "Slowness":
            sort_index = 4
        
        self.rfs = sorted(self.rfs, key=lambda x: x[sort_index])
        
        self.setup_rf_axes()
        xmin = min(self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value())
        xmax = max(self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value())
            
        rf_index = 0 + (self.rf_current_page - 1)*7
        j = np.minimum(7, len(self.rfs[rf_index:]))
        for i in range(j):
            rf = self.rfs[rf_index+i][0]
            t = self.rfs[rf_index+i][1]
            self.mplwidget.figure.axes[i].fill_between(t, np.zeros(len(t)),
                                                       rf, where=(rf > 0), color='black')
            self.mplwidget.figure.axes[i].fill_between(t, np.zeros(len(t)),
                                                       rf, where=(rf < 0), color='red')
            
            if self.rfs[rf_index+i][5]:
                plt.setp(self.mplwidget.figure.axes[i].spines.values(), color='black', linewidth=0.4)
            else:
                plt.setp(self.mplwidget.figure.axes[i].spines.values(), color='red', linewidth=2)
            
            self.mplwidget.figure.axes[i].set_xlim(xmin, xmax)

        self.mplwidget.figure.canvas.draw()
    
    def next_rfs_page(self):
        if self.rf_current_page < self.rf_pages:
            self.rf_current_page += 1
            self.label_14.setText("{}/{}".format(self.rf_current_page, self.rf_pages))
            self.plot_rfs()
    
    def previous_rfs_page(self):
        if self.rf_current_page > 1:
            self.rf_current_page -= 1
            self.label_14.setText("{}/{}".format(self.rf_current_page, self.rf_pages))
            self.plot_rfs()
    
    def recf_plot_clicked(self, event):
        axind = self.mplwidget.figure.axes.index(event.inaxes)
        rf_index = (self.rf_current_page - 1)*7 + axind
        if self.rfs[rf_index][5]:
            self.rfs[rf_index][5] = 0
        else:
            self.rfs[rf_index][5] = 1
        self.plot_rfs()
    
    def save_rfs(self):
        stnm = self.comboBox.currentText()
        a = self.doubleSpinBox.value()
        c = self.doubleSpinBox_2.value()
        utils.save_rfs(stnm, a, c, self.rfs)

    def setup_rf_stack_axes(self):
        if self.first_rf_stack_plot:
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 7])
            gs.update(left=0.125, right=0.95, top=0.98, bottom=0.075, hspace=0.1)
            self.mplwidget_2.figure.add_subplot(gs[0])
            self.mplwidget_2.figure.add_subplot(gs[1])
            self.first_rf_stack_plot = False
        else:
            for i in range(2):
                self.mplwidget_2.figure.axes[i].clear()
        
        self.mplwidget_2.figure.axes[1].set_xlabel("Time in seconds")

    def plot_rf_stack(self):
        stack, bin_stacks, bins, ymin, ymax = utils.compute_stack(self.rfs, bin_size=self.spinBox.value(),
                                                                  overlap=self.spinBox_2.value(),
                                                                  stack_by=self.comboBox_2.currentText(),
                                                                  moveout_correction=self.comboBox_4.currentText())
        self.setup_rf_stack_axes()
        
        t = self.rfs[0][1]
        
        self.mplwidget_2.figure.axes[0].fill_between(t, 0, stack,
                                                     where=(stack > 0),
                                                     color='black')
        self.mplwidget_2.figure.axes[0].fill_between(t, 0, stack,
                                                     where=(stack < 0),
                                                     color='red')
        
        for i, b in enumerate(bins):
            bstack = bin_stacks[:,i]+b
            height = np.zeros(len(t)) + b
            self.mplwidget_2.figure.axes[1].fill_between(t, height, bstack,
                                                         where=(bstack > b),
                                                         color='black')
            self.mplwidget_2.figure.axes[1].fill_between(t, height, bstack,
                                                         where=(bstack < b),
                                                         color='red')
        
        xmin = min(self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value())
        xmax = max(self.doubleSpinBox_7.value(), self.doubleSpinBox_8.value())

        self.mplwidget_2.figure.axes[0].set_xlim(xmin, xmax)

        self.mplwidget_2.figure.axes[1].set_ylim(ymin, ymax)
        self.mplwidget_2.figure.axes[1].set_xlim(xmin, xmax)
        
        self.mplwidget_2.figure.canvas.draw()
    
    def setup_hk_stack_axes(self):
        if self.first_hk_stack_plot:
            gs = gridspec.GridSpec(1, 1)
            gs.update(left=0.175, right=0.95, top=0.95, bottom=0.15)
            self.mplwidget_3.figure.add_subplot(gs[0])
            self.first_hk_stack_plot = False
        else:
            self.mplwidget_3.figure.axes[0].clear()
        
        self.mplwidget_3.figure.axes[0].set_xlabel("H (km)")
        self.mplwidget_3.figure.axes[0].set_ylabel("k")
    
    def plot_hk_stack(self):
        minH = min(self.doubleSpinBox_5.value(), self.doubleSpinBox_6.value())
        maxH = max(self.doubleSpinBox_5.value(), self.doubleSpinBox_6.value())
        Hvalues = self.spinBox_3.value()
        
        mink = min(self.doubleSpinBox_3.value(), self.doubleSpinBox_4.value())
        maxk = max(self.doubleSpinBox_3.value(), self.doubleSpinBox_4.value())
        kvalues = self.spinBox_4.value()
        
        H_arr, k_arr, matrix = utils.compute_hk_stack(self.rfs, H_range=(minH, maxH), H_values=Hvalues,
                                        k_range=(mink, maxk), k_values=kvalues)

        self.setup_hk_stack_axes()        
        self.mplwidget_3.figure.axes[0].pcolormesh(H_arr, k_arr, matrix, cmap="inferno")
        self.mplwidget_3.figure.axes[0].set_xlim(minH, maxH)
        self.mplwidget_3.figure.axes[0].set_ylim(mink, maxk)
        self.mplwidget_3.figure.canvas.draw()
    
    def setup_map_axes(self):
        if self.first_map_plot:
            gs = gridspec.GridSpec(1, 1)
            gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05)
            self.mplwidget_4.figure.add_subplot(gs[0], projection=ccrs.Miller())
            self.first_map_plot = False
        else:
            self.mplwidget_4.figure.axes[0].clear()
        
        self.mplwidget_4.figure.axes[0].coastlines()
        self.mplwidget_4.figure.axes[0].gridlines()
    
    def plot_map(self):
        
        self.setup_map_axes()
        self.mplwidget_4.figure.canvas.draw()
        
        for rf in self.rfs:
            if rf[5]:
                event_id = rf[6]
                lat = self.arrivals['events'][event_id]['event_info']['latitude']
                lon = self.arrivals['events'][event_id]['event_info']['longitude']
                self.mplwidget_4.figure.axes[0].plot(lon, lat, marker='o', color='red',
                                                     transform=ccrs.Geodetic(), markersize=1.5)
        
        self.mplwidget_4.figure.axes[0].set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
        
        self.mplwidget_4.figure.canvas.draw()
    
    def plot_ccp_stack_map(self):
        self.mplwidget_5.figure.add_subplot(111, projection=ccrs.PlateCarree())
    
    def ccp_stack_map_toggle_plot_mode(self, mode):
        
        if mode == "grid":
            self.pushButton_25.setChecked(False)
        elif mode == "cross_section":
            self.pushButton_11.setChecked(False)
    
    def ccp_stack_map_event_handler(self, event):

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
        if self.pushButton_11.isChecked():
            self.doubleSpinBox_9.setValue(self.ccp_grid['y0'])
            self.doubleSpinBox_10.setValue(self.ccp_grid['y1'])
            self.doubleSpinBox_11.setValue(self.ccp_grid['x0'])
            self.doubleSpinBox_12.setValue(self.ccp_grid['x1'])
        elif self.pushButton_25.isChecked():
            start = (self.ccp_cross_section['y0'], self.ccp_cross_section['x0'])
            end = (self.ccp_cross_section['y1'], self.ccp_cross_section['x1'])
            self.doubleSpinBox_17.setValue(start[0])
            self.doubleSpinBox_18.setValue(start[1])
            self.doubleSpinBox_19.setValue(end[0])
            self.doubleSpinBox_20.setValue(end[1])
    
    def ccp_stack_read_rfs(self):
        self.plot_ccp_stack_map()
        dir_ = pw.QFileDialog.getExistingDirectory()
        self.rfs_dicts = utils.map_rfs(rfs_dir=dir_)
        lats, lons = [], []
        for stnm in self.rfs_dicts.keys():
            stla = self.arrivals['stations'][stnm]["lat"]
            stlo = self.arrivals['stations'][stnm]["lon"]
            lats.append(stla)
            lons.append(stlo)
            self.mplwidget_5.figure.axes[0].plot(stlo, stla, marker="^",
                                                 transform=ccrs.Geodetic(), color="green")
            self.mplwidget_5.figure.axes[0].coastlines()

        self.mplwidget_5.figure.axes[0].set_extent([min(lons) - 0.5,
                                                    max(lons) + 0.5,
                                                    min(lats) - 0.5,
                                                    min(lats) + 0.5], ccrs.PlateCarree())

        self.mplwidget_5.figure.canvas.draw()

    def compute_ccp_stack(self):
        self.stack = utils.ccp_stack(self.rfs_dicts, self.arrivals,
                                     min(self.ccp_grid['x0'], self.ccp_grid['x1']),
                                     max(self.ccp_grid['x0'], self.ccp_grid['x1']),
                                     min(self.ccp_grid['y0'], self.ccp_grid['y1']),
                                     max(self.ccp_grid['y0'], self.ccp_grid['y1']))
        
        self.stack_x = np.arange(min(self.ccp_grid['x0'], self.ccp_grid['x1']),
                      max(self.ccp_grid['x0'], self.ccp_grid['x1']), 0.01)
        self.stack_y = np.arange(min(self.ccp_grid['y0'], self.ccp_grid['y1']),
                      max(self.ccp_grid['y0'], self.ccp_grid['y1']), 0.01)
        self.mplwidget_5.figure.axes[0].pcolormesh(self.stack_x, self.stack_y, self.stack[:,:,300].T, transform=ccrs.PlateCarree())
        self.mplwidget_5.figure.canvas.draw()
    
    def cross_section(self):
        if self.istack == None:
            self.istack = utils.interpolate_ccp_stack(self.stack_x, self.stack_y, self.stack)
        
        start = (self.doubleSpinBox_18.value(), self.doubleSpinBox_17.value())
        end = (self.doubleSpinBox_20.value(), self.doubleSpinBox_19.value())
        newlats, newlons = utils.compute_intermediate_points(start, end, 100)
        
        matrix = []
        for lat, lon in zip(newlats, newlons):
            column = []
            for i, stack in enumerate(self.istack):
                cs_val = stack(lon, lat)
                column.append(cs_val)
            matrix.append(column)
        
        matrix = np.array(matrix)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(matrix.size)
        ax.pcolormesh(matrix)
        
    def cut_earthquakes_dialog(self):
        dialog = dialogs.CutEarthquakesDialog()
        dialog.show()
        #dialog.exec_()

#if __name__ == "__main__":
#    app = pw.QApplication([])
#    window = RecfFrame()
#    app.exec_()