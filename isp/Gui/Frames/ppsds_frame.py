from isp.Gui.Frames import BaseFrame, UiPPSDs
from isp.Gui.Frames.ppsds_db_frame import PPSDsGeneratorDialog

from PyQt5 import uic, QtGui, QtCore, QtWidgets
import matplotlib.gridspec as gridspec
import numpy as np
import obspy.imaging.cm
from functools import partial
import copy

class PPSDFrame(BaseFrame, UiPPSDs):

    def __init__(self):
        super(PPSDFrame, self).__init__()
        self.setupUi(self)
        
        self.ppsds_dialog = None
        self.ppsds_db = None
        
        self.page = None
        self.pages = None
        
        self.ppsds_dialog = PPSDsGeneratorDialog(self) 
        
        self.canvas_is_empty = True
        
        # Connect signals w/slots
        self.actionGenerate_synthetics.triggered.connect(self.run_ppsds)
        self.ppsds_dialog.finished.connect(self.populate_list_widget)
        
        self.plotBtn.clicked.connect(self.plot_ppsds)
        self.comboBox_2.currentIndexChanged.connect(self.plot_ppsds)
        self.pushButton.clicked.connect(partial(self.change_page_index, "decrease_index"))
        self.pushButton_2.clicked.connect(partial(self.change_page_index, "increase_index"))
        #self.comboBox_2.currentTextChanged.connect(self.plot_ppsds)

    def run_ppsds(self):
        self.ppsds_dialog.show()

    
    def populate_list_widget(self):
         self.ppsd_db = self.ppsds_dialog.db
         for network in self.ppsd_db['nets'].keys():
             for station in self.ppsd_db['nets'][network].keys():
                 for channel in self.ppsd_db['nets'][network][station].keys():
                     self.tableWidget.insertRow(self.tableWidget.rowCount())
                     self.tableWidget.setItem(self.tableWidget.rowCount() - 1,0,QtWidgets.QTableWidgetItem(network))
                     self.tableWidget.setItem(self.tableWidget.rowCount() - 1,1,QtWidgets.QTableWidgetItem(station))
                     self.tableWidget.setItem(self.tableWidget.rowCount() - 1,2,QtWidgets.QTableWidgetItem(channel))
    
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
        pages = round(len(db_query.keys())/stations_per_page)
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
            
                ppsd = self.ppsd_db['nets'][ntwk][stnm][chnm][1]
                ppsd.calculate_histogram()
                zdata = (ppsd.current_histogram * 100 / (ppsd.current_histogram_count or 1))
                xedges = ppsd.period_xedges
                yedges = ppsd.db_bin_edges
                meshgrid = np.meshgrid(xedges, yedges)                    
                mode = ppsd.db_bin_centers[ppsd._current_hist_stack.argmax(axis=1)]            

                self.mplwidget.figure.axes[j + c].pcolormesh(meshgrid[0], meshgrid[1], zdata.T, cmap=obspy.imaging.cm.pqlx)
                self.mplwidget.figure.axes[j + c].set_xscale("log")
                self.mplwidget.figure.axes[j + c].plot(ppsd.period_bin_centers, mode, color='black', linewidth=2, linestyle='--', label="Mode")
                
                self.mplwidget.figure.axes[j + c].set_xscale("log")
                
                self.mplwidget.figure.axes[j + c].set_title(chnm, fontsize=9, fontweight="medium")
                self.mplwidget.figure.axes[j + c].set_xlabel("Period (s)")
                self.mplwidget.figure.axes[j + c].set_ylabel("Amplitude (dB)")
                
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