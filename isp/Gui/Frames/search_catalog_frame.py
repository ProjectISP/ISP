from obspy.geodetics import gps2dist_azimuth
from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiSearch_Catalog
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.Gui.Frames import BaseFrame, MessageDialog, CartopyCanvas
import pandas as pd
import os
from PyQt5 import QtWidgets
from obspy import UTCDateTime

from isp.seismogramInspector.signal_processing_advanced import find_nearest


@add_save_load()
class SearchCatalogViewer(BaseFrame, UiSearch_Catalog):

        """

        Search Catalog is build to facilitate the search of an Earthquake inside your project from a catalog

        :param params required to initialize the class:


        """
        def __init__(self):
            super(SearchCatalogViewer, self).__init__()
            self.setupUi(self)

            # Binding
            self.root_path_bind = BindPyqtObject(self.rootPathForm)

            # Bind buttons
            self.selectCatalogBtn.clicked.connect(lambda: self.on_click_select_catalog_file(self.root_path_bind))

            # action Buttons
            self.select_eventBtn.clicked.connect(self.select_event)

            # Map
            self.cartopy_canvas = CartopyCanvas(self.map)
            self.cartopy_canvas.mpl_connect('key_press_event', self.key_pressed)
            self.cartopy_canvas.global_map(0)
            self.cartopy_canvas.figure.subplots_adjust(left=0.00, bottom=0.055, right=0.97, top=0.920, wspace=0.0,
                                                       hspace=0.0)
            self.activated_colorbar = True

        def on_click_select_catalog_file(self, bind: BindPyqtObject):
            selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
            if isinstance(selected[0], str) and os.path.isfile(selected[0]):
                bind.value = selected[0]
                self.get_catalog()

        def get_catalog(self):
            try:
                md = MessageDialog(self)
                md.hide()
                self.df_catalog = pd.read_csv(self.root_path_bind.value, sep=";")
                self.set_catalog()
                md.set_info_message("Catalog loaded succesfully!!!")
                md.show()

            except:
                md.set_error_message("Catalog coudn't be loade loaded")
                md.show()

        def set_catalog(self):
            latitudes = []
            longitudes = []
            depths = []
            magnitudes = []

            for index, event in self.df_catalog.iterrows():

                otime = event["Date"]+"TT"+event["Time"]
                lat = event["Latitude"]
                lon = event["Longitude"]
                depth = event["Depth"]
                magnitude = event["Magnitude"]
                magnitude_type = event["mag_type"]
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

                    self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 3, QtWidgets.QTableWidgetItem(str(depth)))

                except TypeError:

                    self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 3, QtWidgets.QTableWidgetItem("N/A"))

                self.tableWidget.setItem(self.tableWidget.rowCount() - 1,4,QtWidgets.QTableWidgetItem(str(magnitude)))
                self.tableWidget.setItem(self.tableWidget.rowCount() - 1,5,QtWidgets.QTableWidgetItem(str(magnitude_type)))

            self.cartopy_canvas.global_map(0, plot_earthquakes=True, show_colorbar=self.activated_colorbar,
                                           lat=latitudes, lon=longitudes, depth=depths, magnitude=magnitudes)

            self.activated_colorbar = False

        def select_event(self):

            selected_items = self.tableWidget.selectedItems()
            event_dict = {}
            errors = False
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

            for event in event_dict.keys():

                date = event_dict[event]['otime']
                date_full = date.split("TT")
                date = date_full[0].split("/")
                time = date_full[1].split(":")
                tt = UTCDateTime(int(date[2]), int(date[1]), int(date[0]), int(time[0]), int(time[1]), float(time[2]))

                self.otime = UTCDateTime(tt)
                self.evla = float(event_dict[event]['lat'])
                self.evlo = float(event_dict[event]['lon'])
                self.evdp = float(event_dict[event]['depth'])
                self.export_event_location_to_earthquake_analysis()

        def key_pressed(self, event):

            if event.key == 't':
                x1, y1 = event.xdata, event.ydata
                self.dataSelect(x1, y1)

        def dataSelect(self, lon1, lat1):

            dist = []
            for row in range(self.tableWidget.rowCount()):
                lat2 = float(self.tableWidget.item(row, 1).text())
                lon2 = float(self.tableWidget.item(row, 2).text())

                great_arc, az0, az2 = gps2dist_azimuth(lat1, lon1, lat2, lon2, a=6378137.0, f=0.0033528106647474805)
                dist.append(great_arc)

            idx, val = find_nearest(dist, min(dist))
            self.tableWidget.selectRow(idx)

        def export_event_location_to_earthquake_analysis(self):
            from isp.Gui.controllers import Controller

            controller: Controller = Controller()
            controller.earthquake_analysis_frame.set_catalog_values([self.otime, self.evla, self.evlo, self.evdp])








