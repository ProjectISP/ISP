# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 00:39:39 2020

@author: Cabieces & Olivar
"""

# isp imports
from isp.Gui import pw
from isp.Gui.Frames import BaseFrame, CartopyCanvas, MessageDialog
from isp.Gui.Frames.uis_frames import UiDataDownloadFrame
from PyQt5 import QtWidgets
import os
import obspy
import obspy.clients.fdsn
import obspy.taup
from functools import partial

from isp.Gui.Utils.pyqt_utils import convert_qdatetime_utcdatetime
from isp.retrieve_events import retrieve


class DataDownloadFrame(BaseFrame, UiDataDownloadFrame):
    def __init__(self):
        super(BaseFrame, self).__init__()
        self.setupUi(self)
        self.inventory = {}
        self.network_list = []
        self.stations_list = []
        self.catalogBtn.clicked.connect(self.get_catalog)
        self.pushButton_2.clicked.connect(self.download_events)
        self.plotstationsBtn.clicked.connect(self.stations)
        self.pushButton_3.clicked.connect(partial(self.download_stationxml, "events"))
        self.LoadBtn.clicked.connect(self.load_inventory)
        # Map
        self.cartopy_canvas = CartopyCanvas(self.map)
        self.cartopy_canvas.global_map(0)
        self.cartopy_canvas.figure.subplots_adjust(left=0.00, bottom=0.055, right=0.97, top=0.920, wspace=0.0,
                                                   hspace=0.0)
        self.activated_colorbar = True
        #self.pushButton_8.clicked.connect(partial(self.download_stationxml, "time_series"))
        self.cartopy_canvas.on_double_click(self.on_click_matplotlib)
        self.cartopy_canvas.mpl_connect('key_press_event', self.key_pressed)
        self.cartopy_canvas.mpl_connect('button_press_event', self.press_right)

    def get_catalog(self):

        latitudes = []
        longitudes = []
        depths = []
        magnitudes = []
        starttime = convert_qdatetime_utcdatetime(self.start_dateTimeEdit)
        endtime = convert_qdatetime_utcdatetime(self.end_dateTimeEdit)
        minmagnitude = self.min_magnitudeCB.value()
        maxmagnitude = self.max_magnitudeCB.value()
        mindepth = self.depth_minCB.value()
        maxdepth = self.depth_maxCB.value()
        # catalog = self.client.get_events(starttime=starttime, endtime=endtime, minmagnitude= self.min_magnitudeCB.value(),
        #          maxmagnitude = self.max_magnitudeCB.value(), mindepth = ,
        #                                  )
        catalog = self.client.get_events(starttime=starttime, endtime=endtime,
                                         minmagnitude=minmagnitude, maxmagnitude = maxmagnitude)

        for event in catalog:
            otime = event.origins[0].time
            lat = event.origins[0].latitude
            lon = event.origins[0].longitude
            depth = event.origins[0].depth
            magnitude = event.magnitudes[0].mag
            magnitude_type = event.magnitudes[0].magnitude_type
            # append results
            latitudes.append(lat)
            longitudes.append(lon)
            depths.append(depth)
            magnitudes.append(magnitude)


            self.tableWidget.insertRow(self.tableWidget.rowCount())
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 0, QtWidgets.QTableWidgetItem(str(otime)))
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, QtWidgets.QTableWidgetItem(str(lat)))
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 2, QtWidgets.QTableWidgetItem(str(lon)))
            try:

                self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 3, QtWidgets.QTableWidgetItem(str(depth/1000)))

            except TypeError:

                self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 3, QtWidgets.QTableWidgetItem("N/A"))

            self.tableWidget.setItem(self.tableWidget.rowCount() - 1,4,QtWidgets.QTableWidgetItem(str(magnitude)))
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1,5,QtWidgets.QTableWidgetItem(str(magnitude_type)))

        selection_range = QtWidgets.QTableWidgetSelectionRange(0,0,self.tableWidget.rowCount() - 1, self.tableWidget.columnCount() - 1)
        #print(selection_range.bottomRow())

        self.tableWidget.setRangeSelected(selection_range, True)
        #print(selection_range)
        self.catalog_out=[latitudes, longitudes, depths, magnitudes]
        # plot earthquakes
        self.cartopy_canvas.global_map(0, plot_earthquakes= True, show_colorbar = self.activated_colorbar,
                                       lat=latitudes, lon=longitudes, depth=depths, magnitude = magnitudes,
                                       resolution = self.typeCB.currentText() )

        self.activated_colorbar = False
        #obspy.clients.fdsn.client.Client


    def download_events(self):

        starttime = convert_qdatetime_utcdatetime(self.start_dateTimeEdit)
        endtime = convert_qdatetime_utcdatetime(self.end_dateTimeEdit)

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

        # Get stations

        #if self.FDSN_CB.isChecked():
        #    url = self.URL_CB.currentText()
        #    client = obspy.clients.fdsn.Client(url)
        networks = self.networksLE.text()
        stations = self.stationsLE.text()
        channels = self.channelsLE.text()
        inventory = self.client.get_stations(network=networks, station=stations, starttime=starttime,
                                            endtime=endtime)

        # elif self.Earthworm_CB.isChecked():
        #     ip_address = self.IP_LE.text()
        #     port = self.portLE.text()
        #     client = obspy.clients.earthworm.Client(ip_address, int(port))
        #     networks = self.networksLE.text()
        #     stations = self.stationsLE.text()
        #     channels = self.channelsLE.text()
        #     inventory = client.get_stations(network=networks, station=stations, starttime=starttime,
        #                                     endtime=endtime)
        
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
                    #stev = stn.elevation
                     
                     
                    # Distance, azimuth and back_azimuth for event:
                    m_dist, az, back_az = obspy.geodetics.base.gps2dist_azimuth(evla, evlo,
                                                                                stla, stlo)
                    deg_dist = obspy.geodetics.base.kilometers2degrees(m_dist/1000)
                    atime = model.get_travel_times(source_depth_in_km=evdp,distance_in_degree=deg_dist,
                                                   receiver_depth_in_km=0.0)

                    p_onset = otime + atime[0].time
                    start = p_onset - self.doubleSpinBox_2.value()
                    end = p_onset + self.doubleSpinBox_3.value()

                    try:
                        st = self.client.get_waveforms(ntwknm, stnm, "*", self.channelsLE.text(), start, end)
                        print(st)
                    except:
                        continue
                    try:
                        self.write(st)
                        md = MessageDialog(self)
                        md.set_info_message("Download completed")
                    except:
                        pass


    def download_stationxml(self, tab):
        fname = QtWidgets.QFileDialog.getSaveFileName()[0]

        # if tab == "time_series":
        #     # datetimeEdits
        #     starttime = obspy.UTCDateTime(self.dateTimeEdit_3.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
        #     endtime = obspy.UTCDateTime(self.dateTimeEdit_4.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z"))
        #     # checkboxes
        #     fdsn_checkbox = self.checkBox_4
        #     earthworm_checkbox = self.checkBox_5
        #     # lineEdits
        #     fdsn_url_line = self.lineEdit_7
        #     earthworm_ip_line = self.lineEdit_9
        #     earthworm_port_line = self.lineEdit_9
        #     networks_line = self.lineEdit_12
        #     stations_line = self.lineEdit_11
        #     channels_line = self.lineEdit_10
        #elif tab == "events":

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

    def write(self, st):
        root_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        if dir_path:
            n=len(st)
            try:
                for j in range(n):
                    tr = st[j]
                    t1 = tr.stats.starttime
                    id = tr.id+"."+"D"+"."+str(t1.year)+"."+str(t1.julday)
                    print(tr.id, "Writing data processed")
                    path_output = os.path.join(dir_path, id)
                    tr.write(path_output, format="MSEED")
            except:
                    md = MessageDialog(self)
                    md.set_info_message("Nothing to write")

    def load_inventory(self):
        #self.networksLE.setText("")
        self.stationsLE.setText("")
        self.channelsLE.setText("")
        starttime = convert_qdatetime_utcdatetime(self.start_dateTimeEdit)
        endtime = convert_qdatetime_utcdatetime(self.end_dateTimeEdit)
        self.retrivetool = retrieve()

        try:
            self.inventory, self.client = self.retrivetool.get_inventory(self.URL_CB.currentText(), starttime, endtime,
            self.networksLE.text(), self.stationsLE.text(), use_networks=self.netsCB.isChecked(), FDSN=self.FDSN_CB.isChecked(),
            ip_address=self.IP_LE.text(), port=self.portLE.text())

            md = MessageDialog(self)
            md.set_info_message("Loaded Inventory from Address")
            self.plotstationsBtn.setEnabled(True)
            self.catalogBtn.setEnabled(True)
        except ValueError:
            md = MessageDialog(self)
            md.set_info_message("The current client does not have a station service")

    def stations(self):

        coordinates = self.retrivetool.get_inventory_coordinates(self.inventory)
        self.cartopy_canvas.global_map(0, plot_earthquakes=False, show_colorbar=False,
           show_stations = True, show_station_names = self.namesCB.isChecked(), clear_plot = False,
                                       coordinates = coordinates, resolution = self.typeCB.currentText())

    def on_click_matplotlib(self, event, canvas):
        self.retrivetool = retrieve()
        if isinstance(canvas, CartopyCanvas):
            x1, y1 = event.xdata, event.ydata
            data = self.retrivetool.get_station_id(x1, y1, self.inventory)

            if len(data[0])>0 and len(data[1])>0:

                if data[0] not in self.network_list:
                    self.network_list.append(data[0])
                    self.networksLE.setText(",".join(self.network_list))
                if data[1] not in self.stations_list:
                    self.stations_list.append(data[1])
                    self.stationsLE.setText(",".join(self.stations_list))

    def key_pressed(self, event):

        self.network_list = []
        self.stations_list = []

        if event.key == 'c':

            self.networksLE.setText(",".join(self.network_list))
            self.stationsLE.setText(",".join(self.stations_list))

    def press_right(self, event):
        self.retrivetool = retrieve()
        if event.dblclick:
            if event.button == 3:
                x1, y1 = event.xdata, event.ydata
                data = self.retrivetool.get_station_id(x1, y1, self.inventory)

                if len(data[0]) > 0 and len(data[1]) > 0:

                    if data[0] in self.network_list:
                        self.network_list.remove(data[0])
                        self.networksLE.setText(",".join(self.network_list))
                    if data[1] in self.stations_list:
                        self.stations_list.remove(data[1])
                        self.stationsLE.setText(",".join(self.stations_list))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    #window = DataDownloadTool()
    app.exec_()
