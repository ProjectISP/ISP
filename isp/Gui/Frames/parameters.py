from isp.Gui import pw
from isp.Gui.Frames.add_parameters import AdditionalParameters
from isp.Gui.Frames.uis_frames import UiParametersFrame
import copy
import enum


class ActionEnum (enum.Enum):
    RMEAN = "rmean"
    TAPER = "taper"
    NORMALIZE = "normalize"
    FILTER = "filter"
    DIFFERENTIATE = "differentiate"
    INTEGRATE = "integrate"
    SHIFT = "shift"

    def __str__(self):
        return str(self.value)


class ParametersSettings(pw.QDialog, UiParametersFrame):
    def __init__(self):
        super(ParametersSettings, self).__init__()
        self.setupUi(self)

        self.orderWidgetsList = []

        self.addCombo.clear()
        for action in ActionEnum:
            self.addCombo.addItem(str(action), action)

        self.addBtn.clicked.connect(self.on_add_action_pushed)
        self.additionalParams = None



    def execAdditionalParameters(self):
        if self.additionalParams is None:
            self.additionalParams = AdditionalParameters()
        self.additionalParams.exec()


    def on_add_action_pushed(self):
        PB_up = pw.QPushButton("Up")
        PB_down = pw.QPushButton("down")
        PB_del = pw.QPushButton("-")
        layoutPB = pw.QHBoxLayout()
        layoutPB.addWidget(PB_up)
        layoutPB.addWidget(PB_down)
        layoutPB.addWidget(PB_del)
        order_widget = pw.QWidget()
        order_widget.setLayout(layoutPB)
        PB_up.clicked.connect(lambda parent=order_widget: self.swapRows(True, order_widget))
        PB_down.clicked.connect(lambda parent=order_widget: self.swapRows(False, order_widget))
        PB_del.clicked.connect(lambda parent=order_widget: self.removeRow(order_widget))

        self.orderWidgetsList.append(order_widget)

        if self.addCombo.currentData() is ActionEnum.RMEAN:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.RMEAN.value))
            combo_param = pw.QComboBox()
            combo_param.addItems(["simple", 'linear', 'demean', 'polynomial', 'spline'])
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, combo_param)

        elif self.addCombo.currentData() is ActionEnum.TAPER:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.TAPER.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            layout = pw.QHBoxLayout()
            combo_param = pw.QComboBox()
            combo_param.addItems(["cosine", 'barthann', 'bartlett', 'blackman', 'blackmanharris', 'bohman', 'boxcar',
                                  'chebwin', 'flattop', 'gaussian', 'general_gaussian', 'hamming', 'hann','kaiser',
                                  'nuttall', 'parzen', 'slepian', 'triang'])
            spin_param = pw.QDoubleSpinBox()
            spin_param.setMaximum(0.5)
            spin_param.setMinimum(0)
            spin_param.setSingleStep(0.01)
            layout.addWidget(combo_param)
            layout.addWidget(spin_param)
            layout.setContentsMargins(0,0,0,0)
            widget = pw.QWidget()
            widget.setLayout(layout)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, widget)

        elif self.addCombo.currentData() is ActionEnum.NORMALIZE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.NORMALIZE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            spin_param = pw.QDoubleSpinBox()
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, spin_param)

        elif self.addCombo.currentData() is ActionEnum.SHIFT:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.SHIFT.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            add_param = pw.QPushButton("Display")
            add_param.clicked.connect(self.execAdditionalParameters)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, add_param)

        elif self.addCombo.currentData() is ActionEnum.FILTER:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.FILTER.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            layout = pw.QHBoxLayout()
            combo_param = pw.QComboBox()
            combo_param.addItems(["bandpass", 'bandstop', 'lowpass', 'highpass', 'lowpass_cheby_2'])

            label_freqmin = pw.QLabel("Freq min")
            label_freqmax = pw.QLabel("Freq max")
            label_number_poles =  pw.QLabel("Number of Poles")

            freq_minDB = pw.QDoubleSpinBox()
            freq_minDB.setMinimum(0)
            freq_minDB.setSingleStep(0.01)

            freq_maxDB = pw.QDoubleSpinBox()
            freq_maxDB.setMinimum(0)
            freq_maxDB.setSingleStep(0.01)

            zero_phaseCB = pw.QCheckBox("zero phase")

            orderSB = pw.QSpinBox()

            layout.addWidget(combo_param)
            layout.addWidget(label_freqmin)
            layout.addWidget(freq_minDB)
            layout.addWidget(label_freqmax)
            layout.addWidget(freq_maxDB)
            layout.addWidget(zero_phaseCB)
            layout.addWidget(label_number_poles)
            layout.addWidget(orderSB)


            layout.setContentsMargins(0, 0, 0, 0)
            widget = pw.QWidget()
            widget.setLayout(layout)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, widget)


        elif self.addCombo.currentData() is ActionEnum.DIFFERENTIATE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.DIFFERENTIATE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            spin_param = pw.QDoubleSpinBox()
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, spin_param)

        elif self.addCombo.currentData() is ActionEnum.INTEGRATE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.INTEGRATE.value))
            combo_param = pw.QComboBox()
            combo_param.addItems(['cumtrapz', 'spline'])
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, combo_param)

    def removeRow(self, order_widget):
        current_row = self.orderWidgetsList.index(order_widget)
        if current_row == ValueError:
            return

        self.tableWidget.removeRow(current_row)
        self.orderWidgetsList.pop(current_row)

    def swapRows(self, up, order_widget):

        current_row = self.orderWidgetsList.index(order_widget)


        if current_row == ValueError :
            return

        if (current_row is 0 and up is True) :
            return

        if current_row is (self.tableWidget.rowCount() - 1) and up is False:
            return

        a = []
        b = []

        dest_row = (current_row - 1) if up else (current_row + 1)

        current_row_action = self.tableWidget.takeItem(current_row, 1)
        dest_row_action = self.tableWidget.takeItem(dest_row, 1)
        self.tableWidget.setItem(current_row, 1, dest_row_action)
        self.tableWidget.setItem(dest_row,1, current_row_action)

        current_row_parameters = []
        dest_row_parameters = []

        print(self.tableWidget.cellWidget(current_row,2).layout().count())
        print(self.tableWidget.cellWidget(dest_row,2).layout().count())
        layout_item = self.tableWidget.cellWidget(current_row,2).layout().takeAt(0)
        while layout_item:
            current_row_parameters.append(layout_item)
            layout_item = self.tableWidget.cellWidget(current_row, 2).layout().takeAt(0)

        layout_item = self.tableWidget.cellWidget(dest_row, 2).layout().takeAt(0)
        while layout_item:
            dest_row_parameters.append(layout_item)
            layout_item = self.tableWidget.cellWidget(dest_row, 2).layout().takeAt(0)

        print(len(current_row_parameters))
        print(len(dest_row_parameters))
        print(self.tableWidget.cellWidget(current_row, 2).layout().count())
        print(self.tableWidget.cellWidget(dest_row, 2).layout().count())
        for item in current_row_parameters:
            self.tableWidget.cellWidget(dest_row, 2).layout().addItem(item)

        for item in dest_row_parameters:
            self.tableWidget.cellWidget(current_row, 2).layout().addItem(item)

        print(self.tableWidget.cellWidget(current_row, 2).layout().count())
        print(self.tableWidget.cellWidget(dest_row, 2).layout().count())

        widget = self.tableWidget.cellWidget(current_row, 2)
        self.tableWidget.setCellWidget(current_row, 2, widget)
        #self.tableWidget.cellWidget(current_row, 2).show()
        #self.tableWidget.cellWidget(dest_row, 2).hide()
        #self.tableWidget.cellWidget(dest_row, 2).setCellWidget(dest_row, 2)
        widget = self.tableWidget.cellWidget(dest_row, 2)
        self.tableWidget.setCellWidget(dest_row, 2, widget)



    def getParameters(self):
        parameters = []
        for i in range(self.tableWidget.rowCount()) :
            action = self.tableWidget.item(i, 1).data(0)

            if (action == ActionEnum.RMEAN.value):
                # TODO: pasar texto o enumerado en action y en parameters?
                parameters.append([action, self.tableWidget.cellWidget(i, 2).currentText()])
            elif (action == ActionEnum.TAPER.value):
                combo_value = self.tableWidget.cellWidget(i, 2).layout().itemAt(0).widget().currentText()
                spin_value = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().value()
                parameters.append([action, combo_value, spin_value])

            elif (action == ActionEnum.SHIFT.value and self.additionalParams is not None):
                parameters.append([action, self.additionalParams.getData()])

        return parameters
