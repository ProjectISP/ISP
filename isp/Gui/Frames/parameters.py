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
    REMOVE_RESPONSE = "remove response"

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
        layoutPB.setContentsMargins(0, 0, 0, 0)
        order_widget.setLayout(layoutPB)
        PB_up.clicked.connect(lambda parent=order_widget: self.swapRows(True, order_widget))
        PB_down.clicked.connect(lambda parent=order_widget: self.swapRows(False, order_widget))
        PB_del.clicked.connect(lambda parent=order_widget: self.removeRow(order_widget))

        self.orderWidgetsList.append(order_widget)
        param_widget = pw.QWidget()
        param_layout = pw.QHBoxLayout()
        param_layout.setContentsMargins(0,0,0,0)
        param_widget.setLayout(param_layout)

        if self.addCombo.currentData() is ActionEnum.RMEAN:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.RMEAN.value))
            combo_param = pw.QComboBox()
            combo_param.addItems(["simple", 'linear', 'demean', 'polynomial', 'spline'])
            param_layout.addWidget(combo_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif self.addCombo.currentData() is ActionEnum.TAPER:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.TAPER.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            combo_param = pw.QComboBox()
            combo_param.addItems(["cosine", 'barthann', 'bartlett', 'blackman', 'blackmanharris', 'bohman', 'boxcar',
                                  'chebwin', 'flattop', 'gaussian', 'general_gaussian', 'hamming', 'hann','kaiser',
                                  'nuttall', 'parzen', 'slepian', 'triang'])
            spin_param = pw.QDoubleSpinBox()
            spin_param.setMaximum(0.5)
            spin_param.setMinimum(0)
            spin_param.setSingleStep(0.01)
            param_layout.addWidget(combo_param)
            param_layout.addWidget(spin_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif self.addCombo.currentData() is ActionEnum.NORMALIZE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.NORMALIZE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            spin_param = pw.QDoubleSpinBox()
            spin_param.setSingleStep(0.01)
            param_layout.addWidget(spin_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2,  param_widget)

        elif self.addCombo.currentData() is ActionEnum.SHIFT:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.SHIFT.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            add_param = pw.QPushButton("Display")
            add_param.clicked.connect(self.execAdditionalParameters)
            param_layout.addWidget(add_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif self.addCombo.currentData() is ActionEnum.FILTER:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.FILTER.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
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

            param_layout.addWidget(combo_param)
            param_layout.addWidget(label_freqmin)
            param_layout.addWidget(freq_minDB)
            param_layout.addWidget(label_freqmax)
            param_layout.addWidget(freq_maxDB)
            param_layout.addWidget(zero_phaseCB)
            param_layout.addWidget(label_number_poles)
            param_layout.addWidget(orderSB)

            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif self.addCombo.currentData() is ActionEnum.REMOVE_RESPONSE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.REMOVE_RESPONSE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)

            label_freqmin = pw.QLabel("Corner Freq min")
            label_freqmax = pw.QLabel("Corner Freq max (%Fn)")
            label_water_level = pw.QLabel("Water level")
            label_units = pw.QLabel("deconvolve to")
            combo_param = pw.QComboBox()
            combo_param.addItems(["DISP", "VEL", "ACC", "Wood Anderson"])

            freq_minDB1 = pw.QDoubleSpinBox()
            freq_minDB1.setMinimum(0)
            freq_minDB1.setSingleStep(0.01)

            freq_minDB2 = pw.QDoubleSpinBox()
            freq_minDB2.setMinimum(0)
            freq_minDB2.setSingleStep(0.01)

            freq_maxDB1 = pw.QDoubleSpinBox()
            freq_maxDB1.setMinimum(0)
            freq_maxDB1.setSingleStep(0.01)

            freq_maxDB2 = pw.QDoubleSpinBox()
            freq_maxDB2.setMinimum(0)
            freq_maxDB2.setSingleStep(0.01)

            water_levelDB = pw.QDoubleSpinBox()
            water_levelDB.setMinimum(0)
            water_levelDB.setMaximum(100)
            water_levelDB.setSingleStep(1)

            param_layout.addWidget(label_freqmin)
            param_layout.addWidget(freq_minDB1)
            param_layout.addWidget(freq_minDB2)
            param_layout.addWidget(label_freqmax)
            param_layout.addWidget(freq_maxDB1)
            param_layout.addWidget(freq_maxDB2)
            param_layout.addWidget(label_water_level)
            param_layout.addWidget(water_levelDB)
            param_layout.addWidget(label_units)
            param_layout.addWidget(combo_param)

            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif self.addCombo.currentData() is ActionEnum.DIFFERENTIATE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.DIFFERENTIATE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            combo_param = pw.QComboBox()
            combo_param.addItems(['gradient'])
            param_layout.addWidget(combo_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif self.addCombo.currentData() is ActionEnum.INTEGRATE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.INTEGRATE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            combo_param = pw.QComboBox()
            combo_param.addItems(['cumtrapz', 'spline'])
            param_layout.addWidget(combo_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif self.addCombo.currentData() is ActionEnum.Remove_Response:
             self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
             self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.Remove_Response.value))
             self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)


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
    
        dest_row = (current_row - 1) if up else (current_row + 1)
    
        # Exchange actions
        current_row_action = self.tableWidget.takeItem(current_row, 1)
        dest_row_action = self.tableWidget.takeItem(dest_row, 1)
        self.tableWidget.setItem(current_row, 1, dest_row_action)
        self.tableWidget.setItem(dest_row,1, current_row_action)

        # Exchange parameters 
        current_row_params = []
        current_row_layout = self.tableWidget.cellWidget(current_row, 2).layout()
        dest_row_params = []
        dest_row_layout = self.tableWidget.cellWidget(dest_row, 2).layout()

        while current_row_layout.count() > 0:
            current_row_params.append(current_row_layout.takeAt(0))

        while dest_row_layout.count() > 0:
            dest_row_params.append(dest_row_layout.takeAt(0))

        new_current_layout = pw.QHBoxLayout()
        new_current_widget = pw.QWidget()

        for i in dest_row_params : 
            new_current_layout.addItem(i)

        new_current_layout.setContentsMargins(0,0,0,0)
        new_current_widget.setLayout(new_current_layout)

        new_dest_layout = pw.QHBoxLayout()
        new_dest_widget = pw.QWidget()

        for i in current_row_params : 
            new_dest_layout.addItem(i)

        new_dest_layout.setContentsMargins(0,0,0,0)        
        new_dest_widget.setLayout(new_dest_layout)

        self.tableWidget.setCellWidget(current_row,2,new_current_widget)
        self.tableWidget.setCellWidget(dest_row,2, new_dest_widget)


    def getParameters(self):
        parameters = []
        for i in range(self.tableWidget.rowCount()) :
            action = self.tableWidget.item(i, 1).data(0)

            if (action == ActionEnum.RMEAN.value):
                parameters.append([action, self.tableWidget.cellWidget(i, 2).layout().itemAt(0).widget().currentText()])

            elif (action == ActionEnum.TAPER.value):
                combo_value = self.tableWidget.cellWidget(i, 2).layout().itemAt(0).widget().currentText()
                spin_value = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().value()
                parameters.append([action, combo_value, spin_value])

            elif (action == ActionEnum.NORMALIZE.value):
                spin_value = self.tableWidget.cellWidget(i, 2).layout().itemAt(0).widget().value()
                parameters.append([action, spin_value])

            elif (action == ActionEnum.SHIFT.value and self.additionalParams is not None):
                parameters.append([action, self.additionalParams.getData()])

            elif (action == ActionEnum.FILTER.value):
                combo_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(0).widget().currentText()
                spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(2).widget().value()
                spin_value2 = self.tableWidget.cellWidget(i, 2).layout().itemAt(4).widget().value()
                check_box = self.tableWidget.cellWidget(i, 2).layout().itemAt(5).widget().isChecked()
                spin_value3 = self.tableWidget.cellWidget(i, 2).layout().itemAt(7).widget().value()
                parameters.append([action, combo_value1, spin_value1, spin_value2, check_box, spin_value3])

            elif (action == ActionEnum.REMOVE_RESPONSE.value):
                spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().value()
                spin_value2 = self.tableWidget.cellWidget(i, 2).layout().itemAt(2).widget().value()
                spin_value3 = self.tableWidget.cellWidget(i, 2).layout().itemAt(4).widget().value()
                spin_value4 = self.tableWidget.cellWidget(i, 2).layout().itemAt(5).widget().value()
                spin_value5 = self.tableWidget.cellWidget(i, 2).layout().itemAt(7).widget().value()
                combo_value = self.tableWidget.cellWidget(i, 2).layout().itemAt(9).widget().currentText()
                parameters.append([action, spin_value1, spin_value2, spin_value3, spin_value4, spin_value5,combo_value])

            elif (action == ActionEnum.DIFFERENTIATE.value):
                combo_value = self.tableWidget.cellWidget(i, 2).layout().itemAt(0).widget().currentText()
                parameters.append([action, combo_value])

            elif (action == ActionEnum.INTEGRATE.value):
                combo_value = self.tableWidget.cellWidget(i, 2).layout().itemAt(0).widget().currentText()
                parameters.append([action, combo_value])


        return parameters
