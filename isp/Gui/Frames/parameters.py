from isp.Gui.Frames.add_parameters import AdditionalParameters
from isp.Gui.Frames.uis_frames import UiParametersFrame
from isp.Gui import pw


class ParametersSettings(pw.QWidget, UiParametersFrame):
    def __init__(self):
        super(ParametersSettings, self).__init__()
        self.setupUi(self)
        print("Set Parameters")
        self.tableWidget.setRowCount(10)




    # Do
        item = pw.QTableWidgetItem("rmean")
        self.tableWidget.setItem(0, 0, item)
        item = pw.QTableWidgetItem("taper")
        self.tableWidget.setItem(1, 0, item)
        item = pw.QTableWidgetItem("Normalize")
        self.tableWidget.setItem(2, 0, item)
        item = pw.QTableWidgetItem("Filter")
        self.tableWidget.setItem(3, 0, item)
        item = pw.QTableWidgetItem("sum")
        self.tableWidget.setItem(4, 0, item)
        item = pw.QTableWidgetItem("Multiply")
        self.tableWidget.setItem(5, 0, item)
        item = pw.QTableWidgetItem("Differenciate")
        self.tableWidget.setItem(6, 0, item)
        item = pw.QTableWidgetItem("Integrate")
        self.tableWidget.setItem(7, 0, item)
        item = pw.QTableWidgetItem("Add Noise")
        self.tableWidget.setItem(8, 0, item)
        item = pw.QTableWidgetItem("Whiten")
        self.tableWidget.setItem(9, 0, item)
        item = pw.QTableWidgetItem("Time Normalization")
        self.tableWidget.setItem(10, 0, item)
        item = pw.QTableWidgetItem("Remove Response")
        self.tableWidget.setItem(11, 0, item)



    # Check
        check  = pw.QCheckBox()
        self.tableWidget.setCellWidget(0, 1, check)
        self.tableWidget.setCellWidget(1, 1, check)
    # Button
        button = pw.QPushButton("Display")
        button.clicked.connect(self.execAdditionalParameters)
        self.tableWidget.setCellWidget(2, 3, button)



    # Order
        sb = pw.QSpinBox()
        self.tableWidget.setCellWidget(0, 2, sb)


    # Parameters
        combo = pw.QComboBox()
        combo.addItem("demean")
        combo.addItem("detrend")
        self.tableWidget.setCellWidget(0, 3, combo)

    def execAdditionalParameters(self):
        additionalParams = AdditionalParameters()
        additionalParams.exec()
        additionalParams.getData()