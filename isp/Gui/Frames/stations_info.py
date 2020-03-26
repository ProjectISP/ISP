from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiStationInfo


class StationsInfo(pw.QFrame, UiStationInfo):
    def __init__(self, st):
        super(StationsInfo, self).__init__()
        self.setupUi(self)
        self.stations_table.setRowCount(len(st))

        for i, info in enumerate(st):
            for k, parameter in enumerate(info):
                self.stations_table.setItem(i, k, pw.QTableWidgetItem(str(parameter)))
        
        self.stations_table.resizeColumnsToContents()