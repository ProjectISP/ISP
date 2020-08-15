from isp.Gui.Frames import BaseFrame, UiPPSDs
from isp.Gui.Frames.ppsds_db_frame import PPSDsGeneratorDialog

from PyQt5 import uic, QtGui, QtCore, QtWidgets
import matplotlib.gridspec as gridspec


class PPSDFrame(BaseFrame, UiPPSDs):

    def __init__(self):
        super(PPSDFrame, self).__init__()
        self.setupUi(self)
        
        self.ppsds_dialog = None
        self.ppsds_db = None
        
        self.ppsds_dialog = PPSDsGeneratorDialog(self)        
        
        # Connect signals w/slots
        self.actionGenerate_synthetics.triggered.connect(self.run_ppsds)
        
        self.plotBtn.clicked.connect(self.plot_ppsds)
        
        self.ppsds_dialog.finished.connect(self.populate_list_widget)
        
        # mpl tests
        gs = gridspec.GridSpec(3, 3)
        #gs.update(left=0.10, right=0.95, top=0.95, bottom=0.075, hspace=0.35)
        #for i in range(7*3):
        #    self.mplwidget.figure.add_subplot(gs[i])
        self.mplwidget.figure.set_size_inches(h=40, w=16)
        for i in range(3*3):
            self.mplwidget.figure.add_subplot(gs[i])
        self.mplwidget.figure.canvas.draw()

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
    
    def plot_ppsds(self, mode="TEST"):
        
        # Retrieve selected stations from the tableWidget
        selected_ppsds = self.tableWidget.selectedItems()
        
        # Loop over the items and create a dictionary
        item_type = 0
        db_query = {}
        for item in selected_ppsds:
            if item_type == 0:
                ntwk = item.text()
                db_query.setdefault(ntwk, {})
            elif item_type == 1:
                stnm = item.text()
                db_query[ntwk].setdefault(stnm, [])
            else:
                db_query[ntwk][stnm].append(item.text())
            
            item_type += 1
            
            if item_type == 3:
                item_type = 0
        
        # Query the DB for the obspy pssd instances
        for ntwk in db_query.keys():
            for stnm in db_query[ntwk].keys():
                for chnm in db_query[ntwk][stnm]:
                    
                    ppsd = self.ppsd_db['nets'][ntwk][stnm][chnm][1]
                    print(self.ppsd_db['nets'][ntwk][stnm][chnm][1])
                    
                    xdata = ppsd.psd_periods
                    mode = ppsd.db_bin_centers[ppsd._current_hist_stack.argmax(axis=1)]
                    self.mplwidget.figure.axes[0].plot(ppsd.period_bin_centers, mode, color='black', linewidth=2, linestyle='--', label="Mode")
                    self.mplwidget.figure.canvas.draw()
        