from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiStationCoords


class StationsCoords(pw.QFrame, UiStationCoords):
    def __init__(self):
        super(StationsCoords, self).__init__()
        self.setupUi(self)
        self.addBtn.clicked.connect(self.on_add_action_pushed)
        self.orderWidgetsList = []

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

    def getCoordinates(self):
        coordinates = []
        for i in range(self.stations_table.rowCount()):
            Name = self.stations_table.item(i, 1).data(0)
            Latitude = self.stations_table.item(i, 2).data(0)
            #Latitude = float(Latitude)
            Longitude = self.stations_table.item(i, 3).data(0)
            Depth = self.stations_table.item(i, 4).data(0)
            coordinates.append([Name, Latitude, Longitude, Depth])

        return coordinates


