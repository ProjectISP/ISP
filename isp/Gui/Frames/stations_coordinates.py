from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiStationCoords
import pandas as pd
from isp import ROOT_DIR
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
import os

class StationsCoords(pw.QFrame, UiStationCoords):
    def __init__(self):
        super(StationsCoords, self).__init__()
        self.setupUi(self)
        self.addBtn.clicked.connect(self.on_add_action_pushed)
        self.orderWidgetsList = []

        self.saveBtn.clicked.connect(self.save_stations_coordinates)

    def on_add_action_pushed(self):

        PB_del = pw.QPushButton("-")
        layoutPB = pw.QHBoxLayout()
        layoutPB.addWidget(PB_del)
        order_widget = pw.QWidget()
        order_widget.setLayout(layoutPB)
        PB_del.clicked.connect(lambda parent=order_widget: self.removeRow(order_widget))
        self.orderWidgetsList.append(order_widget)
        self.stations_table.setRowCount(self.stations_table.rowCount() + 1)
        self.stations_table.setCellWidget(self.stations_table.rowCount() - 1, 0, order_widget)

    def removeRow(self, order_widget):
        current_row = self.orderWidgetsList.index(order_widget)
        if current_row == ValueError:
            return
        self.stations_table.removeRow(current_row)
        self.orderWidgetsList.pop(current_row)

    def __getCoordinates(self):
        coordinates = []
        for i in range(self.stations_table.rowCount()):
            Name = self.stations_table.item(i, 1).data(0)
            Latitude = self.stations_table.item(i, 2).data(0)
            Longitude = self.stations_table.item(i, 3).data(0)
            Depth = self.stations_table.item(i, 4).data(0)
            coordinates.append([Name, Latitude, Longitude, Depth])

        return coordinates

    def save_stations_coordinates(self):
         folder = pw.QFileDialog.getExistingDirectory(self, 'Select a directory', ROOT_DIR)
         file_path = os.path.join(folder, self.rootPathForm.text())
         station_names = []
         station_latitudes = []
         station_longitudes = []
         station_depths = []
         coordinates = self.__getCoordinates()

         for j in range(len(coordinates)):
             station_names.append(coordinates[j][0])
             station_latitudes.append(coordinates[j][1])
             station_longitudes.append(coordinates[j][2])
             station_depths.append(coordinates[j][3])

         coord = {'Name': station_names, 'Lat': station_latitudes, 'Lon': station_longitudes, 'Depth': station_depths}
         df = pd.DataFrame(coord, columns=['Name', 'Lat', 'Lon', 'Depth'])
         df.to_csv(file_path, sep=' ', index=False)





