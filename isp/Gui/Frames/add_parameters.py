from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiAdditionalParameters


class AdditionalParameters(pw.QDialog, UiAdditionalParameters):
    def __init__(self):
        super(AdditionalParameters, self).__init__()
        self.setupUi(self)

        self.addPushButton.clicked.connect(self.on_add_button_clicked)


    def on_add_button_clicked(self):
        self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
        sb = pw.QSpinBox()
        self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, sb)
        dsb = pw.QDoubleSpinBox()
        self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 1, dsb)

    def getData(self):
        data = []
        for n in range(self.tableWidget.rowCount()):
            file = self.tableWidget.cellWidget(n,0).value()
            operand = self.tableWidget.cellWidget(n,1).value()
            data.append([file, operand])

        return data


