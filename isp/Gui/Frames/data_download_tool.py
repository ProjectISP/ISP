# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 00:39:39 2020

@author: olivar
"""

# isp imports
from isp.Gui.Frames import BaseFrame
from isp.Gui import pyqt, pqg, pw, pyc, qt
import isp.receiverfunctions.rf_dialogs as dialogs
import isp.receiverfunctions.rf_main_window_utils as mwu
from isp.Gui.Frames.uis_frames import UiDataDownloadFrame


from PyQt5 import uic, QtGui, QtCore, QtWidgets

import os

import obspy
import obspy.clients.fdsn
import obspy.taup

from functools import partial

class DataDownloadFrame(BaseFrame, UiDataDownloadFrame):
    
    def __init__(self, parent=None):
        super(DataDownloadFrame, self).__init__()
        self.setupUi(self)
        
        self.pushButton.clicked.connect(self.get_catalog)
        self.pushButton_2.clicked.connect(self.download_events)
        self.pushButton_3.clicked.connect(partial(self.download_stationxml, "events"))
        
        self.pushButton_8.clicked.connect(partial(self.download_stationxml, "time_series"))
    
    def get_catalog(self):
        service = self.comboBox.currentText()
        client = obspy.clients.fdsn.Client(service)
        starttime = obspy.UTCDateTime(self.dateTimeEdit.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
        endtime = obspy.UTCDateTime(self.dateTimeEdit_2.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
        minmagnitude = self.doubleSpinBox.value()
        catalog = client.get_events(starttime=starttime, endtime=endtime, minmagnitude=minmagnitude)

        for event in catalog:
            otime = event.origins[0].time
            lat = event.origins[0].latitude
            lon = event.origins[0].longitude
            depth = event.origins[0].depth
            magnitude = event.magnitudes[0].mag
            magnitude_type = event.magnitudes[0].magnitude_type
            
            self.tableWidget.insertRow(self.tableWidget.rowCount())
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1,0,QtWidgets.QTableWidgetItem(str(otime)))
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1,1,QtWidgets.QTableWidgetItem(str(lat)))
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1,2,QtWidgets.QTableWidgetItem(str(lon)))
            try:
                self.tableWidget.setItem(self.tableWidget.rowCount() - 1,3,QtWidgets.QTableWidgetItem(str(depth/1000)))
            except TypeError:
                self.tableWidget.setItem(self.tableWidget.rowCount() - 1,3,QtWidgets.QTableWidgetItem("N/A"))
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1,4,QtWidgets.QTableWidgetItem(str(magnitude)))
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1,5,QtWidgets.QTableWidgetItem(str(magnitude_type)))
        
        selection_range = QtWidgets.QTableWidgetSelectionRange(0,0,self.tableWidget.rowCount() - 1, self.tableWidget.columnCount() - 1)
        print(selection_range.bottomRow())
        
        self.tableWidget.setRangeSelected(selection_range, True)
        print(selection_range)

        #obspy.clients.fdsn.client.Client
    
    def download_events(self):
        outdir = QtWidgets.QFileDialog.getExistingDirectory()
        selected_items = self.tableWidget.selectedItems()
        event_dict = {}
        
        row = 0
        column = 0
        for i, item in enumerate(selected_items):
            event_dict.setdefault(row, {})
            header = self.tableWidget.horizontalHeaderItem(column).text()
            event_dict[row][header] = item.text()
            column += 1
            if i % 5 == 0 and i > 0:
                row += 1
                column = 0
        
        starttime = obspy.UTCDateTime(self.dateTimeEdit.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
        endtime = obspy.UTCDateTime(self.dateTimeEdit_2.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
        
        # Get stations
        if self.checkBox_3.isChecked():
            url = self.lineEdit_6.text()
            client = obspy.clients.fdsn.Client(url)
            networks = self.lineEdit.text()
            stations = self.lineEdit_2.text()
            channels = self.lineEdit_3.text()
            inventory = client.get_stations(network=networks, station=stations, starttime=starttime,
                                            endtime=endtime)
            print(inventory)
        
        elif self.checkBox_2.isChecked():
            ip_address = self.lineEdit_4.text()
            port = self.lineEdit_5.text()
            client = obspy.clients.earthworm.Client(ip_address, int(port))
            networks = self.lineEdit.text()
            stations = self.lineEdit_2.text()
            channels = self.lineEdit_3.text()
            inventory = client.get_stations(network=networks, station=stations, starttime=starttime,
                                            endtime=endtime)
        
        model = obspy.taup.TauPyModel(model="iasp91")
        for event in event_dict.keys():
            otime = obspy.UTCDateTime(event_dict[event]['otime'])
            evla = float(event_dict[event]['lat'])
            evlo = float(event_dict[event]['lon'])
            evdp = float(event_dict[event]['depth'])

            for ntwk in inventory:
                ntwknm = ntwk.code
                for stn in ntwk:
                    stnm = stn.code
                    stla = stn.latitude
                    stlo = stn.longitude
                    stev = stn.elevation
                     
                     
                    # Distance, azimuth and back_azimuth for event:
                    m_dist, az, back_az = obspy.geodetics.base.gps2dist_azimuth(evla, evlo,
                                                                                stla, stlo)
                    deg_dist = obspy.geodetics.base.kilometers2degrees(m_dist/1000)
                    atime = model.get_travel_times(source_depth_in_km=evdp,
                                                   distance_in_degree=deg_dist,
                                                   phase_list=['P'],
                                                   receiver_depth_in_km=0.0)

                    p_onset = otime + atime[0].time
                    start = p_onset - self.doubleSpinBox_2.value()
                    end = p_onset + self.doubleSpinBox_3.value()
                    
                    start_str = "{}.{}.{}{}{}".format(start.year, start.julday, start.hour, start.minute, start.second)
                    end_str = "{}.{}.{}{}{}".format(end.year, end.julday, end.hour, end.minute, end.second)
                    
                    #st = obspy.core.stream.Stream()
                    if channels != "*":
                        print(channels)
                        for chnm in channels:
                            try:
                                st = client.get_waveforms(ntwknm, stnm, "*", chnm, start, end)
                            except:
                                continue
                            fname = "{}.{}.{}.".format(ntwknm, stnm, chnm) + start_str + "." + end_str + ".mseed"
                            path = os.path.join(outdir, fname)
                            st.write(path, format="MSEED")
                    else:
                        st = client.get_waveforms(ntwknm, stnm, "*", channels, start, end)
                        for tr in st:
                            chnm = tr.stats.channel
                            fname = "{}.{}.{}.".format(ntwknm, stnm, chnm) + start_str + "." + end_str + ".mseed"
                            path = os.path.join(outdir, fname)
                            tr.write(path, format="MSEED")

    def download_stationxml(self, tab):
        fname = QtWidgets.QFileDialog.getSaveFileName()[0]

        if tab == "time_series":
            # datetimeEdits
            starttime = obspy.UTCDateTime(self.dateTimeEdit_3.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
            endtime = obspy.UTCDateTime(self.dateTimeEdit_4.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
            # checkboxes
            fdsn_checkbox = self.checkBox_4
            earthworm_checkbox = self.checkBox_5
            # lineEdits
            fdsn_url_line = self.lineEdit_7
            earthworm_ip_line = self.lineEdit_9
            earthworm_port_line = self.lineEdit_9
            networks_line = self.lineEdit_12
            stations_line = self.lineEdit_11
            channels_line = self.lineEdit_10
        elif tab == "events":
            starttime = obspy.UTCDateTime(self.dateTimeEdit.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
            endtime = obspy.UTCDateTime(self.dateTimeEdit_2.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
            fdsn_checkbox = self.checkBox_3
            earthworm_checkbox = self.checkBox_2
            # lineEdits
            fdsn_url_line = self.lineEdit_6
            earthworm_ip_line = self.lineEdit_4
            earthworm_port_line = self.lineEdit_5
            networks_line = self.lineEdit
            stations_line = self.lineEdit_2
            channels_line = self.lineEdit_3

        networks = networks_line.text()
        stations = stations_line.text()
    
        if fdsn_checkbox.isChecked():
            url = fdsn_url_line.text()
            client = obspy.clients.fdsn.Client(url)
        elif earthworm_checkbox.isChecked():
            ip_address = self.lineEdit_4.text()
            port = self.lineEdit_5.text()
            client = obspy.clients.earthworm.Client(ip_address, int(port))

        inventory = client.get_stations(network=networks, station=stations, starttime=starttime,
                                        endtime=endtime, level="response")
        
        inventory.write(fname, format="STATIONXML")
        

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = DataDownloadTool()
    app.exec_()
