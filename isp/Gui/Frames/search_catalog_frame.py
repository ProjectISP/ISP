from obspy.geodetics import gps2dist_azimuth
from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiSearch_Catalog
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from isp.Gui.Frames import BaseFrame, MessageDialog, CartopyCanvas
import pandas as pd
import os
from PyQt5 import QtWidgets
from obspy import UTCDateTime, read_events, Catalog
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
            self.df_catalog = None
            self.reLoadCatalogBtn.clicked.connect(self.get_catalog)

        def on_click_select_catalog_file(self, bind: BindPyqtObject):
            selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
            if isinstance(selected[0], str) and os.path.isfile(selected[0]):
                bind.value = selected[0]
                self.get_catalog()

        def get_catalog(self):
            md = MessageDialog(self)
            md.hide()

            try:
                # Try reading the file as an ObsPy Catalog
                catalog = read_events(self.root_path_bind.value)
                if isinstance(catalog, Catalog):
                    print("File contains a valid ObsPy Catalog object. Extracting data...")
                    self.df_catalog = self._extract_from_catalog(catalog)
                    self.set_catalog()
                    md.set_info_message("Catalog loaded successfully!")
                    md.show()
                else:
                    raise ValueError("File is not a valid ObsPy Catalog.")  # Explicitly raise if not Catalog

            except Exception as obspy_error:
                # If ObsPy read fails, try reading with pandas
                try:
                    self.df_catalog = pd.read_csv(self.root_path_bind.value, sep=";")
                    self.set_catalog()
                    md.set_info_message("Text file loaded successfully!")
                    md.show()
                except Exception as pandas_error:
                    # Handle the error if both attempts fail
                    error_message = f"Failed to load catalog: {obspy_error} | Text file error: {pandas_error}"
                    print(error_message)  # Optional: log error details
                    md.set_error_message("Catalog couldn't be loaded. Please check the file format!")
                    md.show()




        def _extract_from_catalog(self, catalog):
            # Initialize a list to store each event's data as a dictionary
            events_data = {}
            latitude_list = []
            longitude_list = []
            depth_list = []
            event_date_list = []
            event_time_list = []
            mag_value_list = []
            mag_type_list = []

            for event in catalog:
                # Extract latitude, longitude, depth, and origin time
                origin = event.preferred_origin() or event.origins[0]
                latitude = origin.latitude
                latitude_list.append(latitude)
                longitude = origin.longitude
                longitude_list.append(longitude)
                depth = origin.depth / 1000  # Convert depth to km
                depth_list.append(depth)
                # Format origin time
                origin_time = origin.time.strftime("%d/%m/%YTT%H:%M:%S")
                event_date = origin_time.split("TT")[0]
                event_time = origin_time.split("TT")[1]
                event_date_list.append(event_date)
                event_time_list.append(event_time)
                # Extract magnitude and magnitude type
                magnitude = event.preferred_magnitude() or event.magnitudes[0]
                mag_value = magnitude.mag
                mag_value_list.append(mag_value)
                mag_type = magnitude.magnitude_type
                mag_type_list.append(mag_type)

            # Store event data as a dictionary
            event_data = {
                "Latitude": latitude_list,
                "Longitude": longitude_list,
                "Depth": depth_list,
                "Magnitude": mag_value_list,
                "mag_type": mag_type_list,
                "Date": event_date_list,
                "Time": event_time_list
            }
                #events_data.append(event_data)

            # Convert the list of dictionaries to a DataFrame
            return pd.DataFrame(event_data)

        def set_catalog(self):

            self.tableWidget.clearContents()
            # Update the row count based on data length
            # rows, columns = df.shape
            self.tableWidget.setRowCount(0)

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
            try:
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

                md = MessageDialog(self)
                md.set_info_message("Send Event info to Earthquake Analysis, "
                                    "Ready to filter your Project")
            except:
                md = MessageDialog(self)
                md.set_info_message("Error Sending Event info to Earthquake Analysis \n "
                                    "Please check that you have selected an event from the table")

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








