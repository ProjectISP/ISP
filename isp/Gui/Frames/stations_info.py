from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiStationInfo


class StationsInfo(pw.QFrame, UiStationInfo):
    def __init__(self, st, check = False):
        super(StationsInfo, self).__init__()
        self.setupUi(self)
        self.stations_table.setRowCount(len(st))

        for i, info in enumerate(st):
            for k, parameter in enumerate(info):
                self.stations_table.setItem(i, k, pw.QTableWidgetItem(str(parameter)))

        if check:
            #Add new columns for check
            self.stations_table.setColumnCount(self.stations_table.columnCount()+1)
            self.stations_table.setHorizontalHeaderItem(self.stations_table.columnCount()-1,
                                                       pw.QTableWidgetItem("Check Component"))
            for j in range(len(st)):
                check = pw.QCheckBox()
                check.setChecked(True)
                self.stations_table.setCellWidget(j, self.stations_table.columnCount()-1, check)


        self.stations_table.resizeColumnsToContents()

    def get_stations_map(self):
        stations_map = []
        for i in range(self.stations_table.rowCount()):
            station_name = self.stations_table.item(i, 1).text()
            channel = self.stations_table.item(i, 3).text()
            checked = self.stations_table.cellWidget(i, 8).isChecked()
            stations_map.append([station_name, channel, checked])
        return stations_map
