from isp.Gui.Frames import BaseFrame, UiPPSDs
from isp.Gui.Frames.ppsds_db_frame import PPSDsGeneratorDialog

from PyQt5 import uic, QtGui, QtCore, QtWidgets


class PPSDFrame(BaseFrame, UiPPSDs):

    def __init__(self):
        super(PPSDFrame, self).__init__()
        self.setupUi(self)
        
        self.ppsds_dialog = None
        self.ppsds_db = None
        
        self.ppsds_dialog = PPSDsGeneratorDialog(self)        
        
        # Connect signals w/slots
        self.actionGenerate_synthetics.triggered.connect(self.run_ppsds)
        self.ppsds_dialog.finished.connect(self.populate_list_widget)

    def run_ppsds(self):
        self.ppsds_dialog.show()
    
    def populate_list_widget(self):
        
        print("Test")
        
        self.ppsd_db = self.ppsds_dialog.db
        
        for network in self.ppsd_db['nets'].keys():
            for station in self.ppsd_db['nets'][network].keys():
                for channel in self.ppsd_db['nets'][network][station].keys():
                    self.tableWidget.insertRow(self.tableWidget.rowCount())
                    self.tableWidget.setItem(self.tableWidget.rowCount() - 1,0,QtWidgets.QTableWidgetItem(network))
                    self.tableWidget.setItem(self.tableWidget.rowCount() - 1,1,QtWidgets.QTableWidgetItem(station))
                    self.tableWidget.setItem(self.tableWidget.rowCount() - 1,2,QtWidgets.QTableWidgetItem(channel))