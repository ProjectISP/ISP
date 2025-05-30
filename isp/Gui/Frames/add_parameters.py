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
        dsb.setMinimum(-9999);
        dsb.setMaximum(9999);
        dsb.setSingleStep(0.01)
        self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 1, dsb)

    def getData(self):
        data = []
        for n in range(self.tableWidget.rowCount()):
            file = self.tableWidget.cellWidget(n,0).value()
            operand = self.tableWidget.cellWidget(n,1).value()
            data.append([file, operand])

        return data

    def setData(self, data):
        for i in range(len(data)):
            self.on_add_button_clicked()
            self.tableWidget.cellWidget(i,0).setValue(data[i][0])
            self.tableWidget.cellWidget(i,1).setValue(data[i][1])


