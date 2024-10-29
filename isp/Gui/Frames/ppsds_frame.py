from isp.Gui.Frames import BaseFrame, UiPPSDs
from isp.Gui.Frames.ppsds_db_frame import PPSDsGeneratorDialog
import isp.receiverfunctions.rf_dialogs as dialogs # using save_figure dialog

from PyQt5 import uic, QtGui, QtCore, QtWidgets
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import io
import pickle
import numpy as np
import obspy.imaging.cm
import obspy.signal.spectral_estimation as osse
from functools import partial
import copy
from obspy.signal.spectral_estimation import earthquake_models
from matplotlib.patheffects import withStroke
from isp.Gui.Frames import MessageDialog

class PPSDFrame(BaseFrame, UiPPSDs):

    def __init__(self):
        super(PPSDFrame, self).__init__()
        self.setupUi(self)
        
        self.ppsds_dialog = None
        self.ppsds_db = None
        
        self.page = None
        self.pages = None
        
        self.ppsds_dialog = PPSDsGeneratorDialog(self)
        self.ppsds_dialog.show()
        self.canvas_is_empty = True
        
        # Connect signals w/slots
        self.actionGenerate_synthetics.triggered.connect(self.run_ppsds)
        self.ppsds_dialog.finished.connect(self.populate_list_widget)
        
        self.plotBtn.clicked.connect(self.plot_ppsds)
        self.comboBox_2.currentIndexChanged.connect(self.plot_ppsds)
        self.pushButton.clicked.connect(partial(self.change_page_index, "decrease_index"))
        self.pushButton_2.clicked.connect(partial(self.change_page_index, "increase_index"))
        self.pushButton_3.clicked.connect(self.save_plot)
        #self.comboBox_2.currentTextChanged.connect(self.plot_ppsds)

    def run_ppsds(self):
        self.ppsds_dialog.show()

    
    def populate_list_widget(self):
        try:
             self.ppsd_db = self.ppsds_dialog.db
             for network in self.ppsd_db['nets'].keys():
                 for station in self.ppsd_db['nets'][network].keys():
                     for channel in self.ppsd_db['nets'][network][station].keys():
                         self.tableWidget.insertRow(self.tableWidget.rowCount())
                         self.tableWidget.setItem(self.tableWidget.rowCount() - 1,0,QtWidgets.QTableWidgetItem(network))
                         self.tableWidget.setItem(self.tableWidget.rowCount() - 1,1,QtWidgets.QTableWidgetItem(station))
                         self.tableWidget.setItem(self.tableWidget.rowCount() - 1,2,QtWidgets.QTableWidgetItem(channel))
        except:
            pass

    def change_page_index(self, change):
        
        if change == "decrease_index":
            new_index = self.comboBox_2.currentIndex() - 1
            if new_index >= 0:        
                self.comboBox_2.setCurrentIndex(new_index)
        elif change == "increase_index":
            new_index = self.comboBox_2.currentIndex() + 1
            if new_index < self.comboBox_2.count():
                self.comboBox_2.setCurrentIndex(new_index)
    
    def plot_ppsds(self, stations_per_page):
        
        starttime = obspy.UTCDateTime(self.dateTimeEdit_4.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
        endtime = obspy.UTCDateTime(self.dateTimeEdit_5.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
        
        # Check which radiobutton is checked: PDFs or Variation
        if self.radioButton.isChecked():
            plot_mode = "pdf"
        elif self.radioButton_2.isChecked():
            plot_mode = "variation"

        stations_per_page = self.spinBox.value()
        #mode = self.comboBox.currentText() # NOT YET IMPLEMENTED
        
        self.mplwidget.figure.clf()  

        # Retrieve selected stations from the tableWidget
        selected_ppsds = self.tableWidget.selectedItems()
        
        # Loop over the items and create a dictionary
        # THIS WILL BREAK IF TWO STATIONS HAVE THE SAME NAME AS OF THIS VERSION
        item_type = 0
        db_query = {}
        for item in selected_ppsds:
            if item_type == 0:
                ntwk = item.text()
            elif item_type == 1:
                stnm = item.text()
                db_query.setdefault(stnm, {})
                db_query[stnm].setdefault('channels', [])
                db_query[stnm]['network'] = ntwk
            else:
                chnm = item.text()
                db_query[stnm]['channels'].append(chnm)           
            
            item_type += 1
            
            if item_type == 3:
                item_type = 0
        
        selected_stations = sorted(list(db_query.keys()))

        # Number of stations per page can never be greater than the number of
        # selected stations
        if len(selected_stations) < stations_per_page:
            stations_per_page = len(selected_stations)

        # Initialize the plot with the necessary number of axes
        gs = gridspec.GridSpec(stations_per_page, 3)
        gs.update(left=0.10, right=0.95, top=0.95, bottom=0.075, hspace=0.35, wspace=0.35)
        for i in range(stations_per_page*3):
            self.mplwidget.figure.add_subplot(gs[i])                    

        # If necessary populate the page combobox
        try:
            pages = round(len(db_query.keys())/stations_per_page)
        except ZeroDivisionError: # This means no stations are selected
            md = MessageDialog(self)
            md.set_warning_message("No stations selected!!!")
            return
        
        if self.pages != pages:
            self.pages = pages
            self.comboBox_2.clear()
            for i in range(pages):
                self.comboBox_2.addItem(str(i+1))

        # Plot the corresponding stations
        try:
            page = int(self.comboBox_2.currentText()) - 1
        except ValueError:
            page = 0
        
        st1 = stations_per_page * page
        st2 = min(st1 + stations_per_page, len(selected_stations))
        
        j = 0 # j: axis index for first channel of a station
        
        for i in range(st1, st2):
            stnm = selected_stations[i]
            ntwk = db_query[stnm]['network']
            
            c = 0 # c index of channel in current station
            for chnm in sorted(db_query[stnm]['channels']):
                # THIS WILL BE MOVED TO AN EXTERNAL METHOD IN A FUTURE REVISION
                ppsd = self.ppsd_db['nets'][ntwk][stnm][chnm][1]
                
                if starttime == endtime:
                    ppsd.calculate_histogram()
                else:
                    ppsd.calculate_histogram(starttime=starttime, endtime=endtime)
                
                if plot_mode == "pdf":
                    try:
                        zdata = (ppsd.current_histogram * 100 / (ppsd.current_histogram_count or 1))
                    except:  # This means no stations are selected
                        md = MessageDialog(self)
                        md.set_error_message("Some data channel no valid.")
                        return

                    xedges = ppsd.period_xedges
                    yedges = ppsd.db_bin_edges
                    meshgrid = np.meshgrid(xedges, yedges)
                    mode = ppsd.db_bin_centers[ppsd._current_hist_stack.argmax(axis=1)]

                    self.mplwidget.figure.axes[j + c].pcolormesh(meshgrid[0], meshgrid[1], zdata.T, cmap=obspy.imaging.cm.pqlx)
                    self.mplwidget.figure.axes[j + c].set_xscale("log")

                    #self.mplwidget.figure.axes[j + c].plot(ppsd.period_bin_centers, mode, color='black', linewidth=2, linestyle='--', label="Mode")
                    
                    self.mplwidget.figure.axes[j + c].set_xscale("log")
                    
                    self.mplwidget.figure.axes[j + c].set_xlabel("Period (s)")
                    self.mplwidget.figure.axes[j + c].set_ylabel("Amplitude (dB)")
                    
                    self.plot_statistics(self.mplwidget.figure.axes[j + c], ppsd)
                    self.mplwidget.figure.axes[j + c].set_xlim(0.02, 120)
                
                elif plot_mode == "variation":
                    variation = self.comboBox.currentText()
                    # THIS WILL BE MOVED TO AN EXTERNAL METHOD IN A FUTURE REVISION
                    if variation == "Diurnal":
                        hist_dict = {}
                        num_period_bins = len(ppsd.period_bin_centers)
                        num_db_bins = len(ppsd.db_bin_centers)
                        for i in range(24):
                            hist_dict.setdefault(i, np.zeros((num_period_bins, num_db_bins), dtype=np.uint64))
    
                        for i, time in enumerate(ppsd.times_processed):
                            if starttime != time:
                                if not starttime < time < endtime:
                                    continue
                            year = time.year
                            jday = time.julday
                            hour = time.hour
                            # Check here if time is inside starttime-endtime <---- TO DO
                            inds = ppsd._binned_psds[i]
                            inds = ppsd.db_bin_edges.searchsorted(inds, side="left") - 1
                            inds[inds == -1] = 0
                            inds[inds == num_db_bins] -= 1
                            for i, inds_ in enumerate(inds):
                                # count how often each bin has been hit for this period bin,
                                # set the current 2D histogram column accordingly
                                hist_dict[hour][i, inds_] += 1
    
                        # Finally compute statistical mode for each hour:
                        modes = []
                        for i in sorted(list(hist_dict.keys())):
                            current_hist = hist_dict[i]
                            mode = ppsd.db_bin_centers[current_hist.argmax(axis=1)]
                            modes.append(mode)
                        
                        x = ppsd.period_bin_centers
                        y = np.arange(1, 25, 1)
    
                        self.mplwidget.figure.axes[j + c].contourf(y, x, np.array(modes).T, cmap=obspy.imaging.cm.pqlx, levels=200)
                        self.mplwidget.figure.axes[j + c].set_xlabel("GMT Hour")
                        self.mplwidget.figure.axes[j + c].set_ylabel("Period (s)")
                        self.mplwidget.figure.axes[j + c].set_ylim(0.02, 120)
                    elif variation == "Seasonal":
                        # Create blank 2D histogram for each hour
                        hist_dict = {}
                        num_period_bins = len(ppsd.period_bin_centers)
                        num_db_bins = len(ppsd.db_bin_centers)
                        for i in range(12):
                            hist_dict.setdefault(i, np.zeros((num_period_bins, num_db_bins), dtype=np.uint64))
                        for i, time in enumerate(ppsd.times_processed):
                            if starttime != time:
                                if not starttime < time < endtime:
                                    continue
                            year = time.year
                            jday = time.julday
                            hour = time.hour
                            month = time.month
                            inds = ppsd._binned_psds[i]
                            inds = ppsd.db_bin_edges.searchsorted(inds, side="left") - 1
                            inds[inds == -1] = 0
                            inds[inds == num_db_bins] -= 1
                            for i, inds_ in enumerate(inds):
                                # count how often each bin has been hit for this period bin,
                                # set the current 2D histogram column accordingly
                                hist_dict[month - 1][i, inds_] += 1
                    
                        # Finally compute statistical mode for each month:
                        modes = []
                        for i in sorted(list(hist_dict.keys())):
                            current_hist = hist_dict[i]
                            mode = ppsd.db_bin_centers[current_hist.argmax(axis=1)]
                            modes.append(mode)
                        
                        x = ppsd.period_bin_centers
                        y = np.arange(1, 13, 1)

                        self.mplwidget.figure.axes[j + c].contourf(y, x, np.array(modes).T, cmap=obspy.imaging.cm.pqlx, levels=200)
                        self.mplwidget.figure.axes[j + c].set_xlabel("Month")
                        self.mplwidget.figure.axes[j + c].set_ylabel("Period (s)")
                        self.mplwidget.figure.axes[j + c].set_ylim(0.02, 120)
                        self.mplwidget.figure.axes[j + c].set_xlim(1, 12)

                self.mplwidget.figure.axes[j + c].set_title(chnm, fontsize=9, fontweight="medium")

                if c == 0:
                    self.mplwidget.figure.axes[j].set_title(stnm, loc="left", fontsize=11, fontweight="bold")
                
                c = c + 1
            
            if c < 3:
                for l in range(c, 3):
                    self.mplwidget.figure.axes[j + l].set_axis_off()
                
                c = 3
            
            j += c
        
        # set axis off for unused axis in the plot
        if j < stations_per_page*3:
            for m in range(j, stations_per_page*3):
                self.mplwidget.figure.axes[m].set_axis_off()
                
                
            
        self.mplwidget.figure.canvas.draw()
    
    def plot_statistics(self, axis, ppsd):
        if self.checkBox.isChecked():
            mean = ppsd.get_mean()
            axis.plot(mean[0], mean[1], color='black', linewidth=1, linestyle='--', label="Mean")
        
        if self.checkBox_2.isChecked():
            mode = ppsd.get_mode()       
            axis.plot(mode[0], mode[1], color='green', linewidth=1, linestyle='--', label="Mode")

        if self.checkBox_3.isChecked():
            nhnm = osse.get_nhnm()    
            axis.plot(nhnm[0], nhnm[1], color='gray', linewidth=2, linestyle='-', label="NHNM (Peterson et al., 2003)")

        if self.checkBox_4.isChecked():
            nlnm = osse.get_nlnm()      
            axis.plot(nlnm[0], nlnm[1], color='gray', linewidth=2, linestyle='-', label="NLNM (Peterson et al., 2003)")

        if self.groupBox_2.isChecked():

            min_mag, max_mag, min_dist, max_dist = (self.doubleSpinBox.value(), self.doubleSpinBox_2.value(),
                                                    self.doubleSpinBox_3.value(), self.doubleSpinBox_4.value())
    
            for key, data in earthquake_models.items():
                magnitude, distance = key
                frequencies, accelerations = data
                accelerations = np.array(accelerations)
                frequencies = np.array(frequencies)
                periods = 1.0 / frequencies
                # Eq.1 from Clinton and Cauzzi (2013) converts
                # power to density
                ydata = accelerations / (periods ** (-.5))
                ydata = 20 * np.log10(ydata / 2)
                if not (min_mag <= magnitude <= max_mag and
                        min_dist <= distance <= max_dist and
                        min(ydata) < ppsd.db_bin_edges[-1]):
                    continue
                xdata = periods
                axis.plot(xdata, ydata, linewidth=2, color="black")
                leftpoint = np.argsort(xdata)[0]
                if not ydata[leftpoint] < ppsd.db_bin_edges[-1]:
                    continue
                axis.text(xdata[leftpoint],
                        ydata[leftpoint],
                        'M%.1f\n%dkm' % (magnitude, distance),
                        ha='right', va='top',
                        color='w', weight='bold', fontsize='x-small',
                        path_effects=[withStroke(linewidth=3,
                                                 foreground='0.4')])

    def save_plot(self):
        fig = self.mplwidget.figure
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
        
        dialog = dialogs.SaveFigureDialog(fig_copy)
        dialog.exec_()
        return