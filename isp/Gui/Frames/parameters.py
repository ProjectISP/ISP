from isp.Gui import pw
from isp.Gui.Frames.add_parameters import AdditionalParameters
from isp.Gui.Frames.uis_frames import UiParametersFrame
from isp import MACROS_PATH
import enum
import pickle

class ActionEnum (enum.Enum):

    RMEAN = "rmean"
    TAPER = "taper"
    NORMALIZE = "normalize"
    FILTER = "filter"
    WIENER = "wiener filter"
    RESAMPLE = "resample"
    FILL_GAPS = "fill gaps"
    DIFFERENTIATE = "differentiate"
    INTEGRATE = "integrate"
    SHIFT = "shift"
    REMOVE_RESPONSE = "remove response"
    ADD_WHITE_NOISE = "add white noise"
    WHITENING = "whitening"
    TNOR = "time normalization"
    WAVELET_DENOISE = "wavelet denoise"
    SMOOTHING = "smoothing"
    REMOVE_SPIKES = "remove spikes"

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

        self.currentFilename = None
        self.saveBtn.clicked.connect(self.on_save_action_pushed)
        self.loadBtn.clicked.connect(self.on_load_action_pushed)

    def execAdditionalParameters(self):
        if self.additionalParams is None:
            self.additionalParams = AdditionalParameters()
        self.additionalParams.exec()

    def on_save_action_pushed(self):
        path = self.currentFilename
        if not path:
            path = MACROS_PATH

        selected = pw.QFileDialog.getSaveFileName( 
            self, "Select target macros file", path)
        if selected:
            s = selected[0]
            if '.pkl' not in s:
                s = selected[0] + '.pkl'
            with open(s, 'wb') as f:
                pickle.dump(self.getParameters(), f)
                self.currentFilename = s

    def on_load_action_pushed(self):
        path = self.currentFilename
        if not path:
            path = MACROS_PATH

        selected = pw.QFileDialog.getOpenFileName(
            self, "Select macros file", path)

        if selected:
            with open(selected[0], 'rb') as f:
                parameters = pickle.load(f)
                self.tableWidget.setRowCount(0)
                self.orderWidgetsList.clear()
                for p in parameters:
                    self._add_row(ActionEnum(p[0]), *p[1:])
                self.currentFilename = selected[0]

    def on_add_action_pushed(self):
        self._add_row(self.addCombo.currentData())

    def _add_row(self, action, *params):
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

        if action is ActionEnum.RMEAN:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.RMEAN.value))
            combo_param = pw.QComboBox()
            combo_param.addItems(["simple", 'linear', 'demean', 'polynomial', 'spline'])

            if len(params) > 0:
                combo_param.setCurrentText(params[0])

            param_layout.addWidget(combo_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.TAPER:
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

            if len(params) > 1 :
                combo_param.setCurrentText(params[0])
                spin_param.setValue(params[1])

            param_layout.addWidget(combo_param)
            param_layout.addWidget(spin_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.NORMALIZE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.NORMALIZE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            spin_param = pw.QDoubleSpinBox()
            spin_param.setSingleStep(0.01)

            if len(params) > 0:
                spin_param.setValue(params[0])

            param_layout.addWidget(spin_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2,  param_widget)

        elif action is ActionEnum.SHIFT:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.SHIFT.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            add_param = pw.QPushButton("Display")
            add_param.clicked.connect(self.execAdditionalParameters)

            if len(params) > 0:
                self.additionalParams = AdditionalParameters()
                self.additionalParams.setData(params[0])

            param_layout.addWidget(add_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.FILTER:
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

            if len(params) > 4:
                combo_param.setCurrentText(params[0])
                freq_minDB.setValue(params[1])
                freq_maxDB.setValue(params[2])
                zero_phaseCB.setChecked(params[3])
                orderSB.setValue(params[4])

            param_layout.addWidget(combo_param)
            param_layout.addWidget(label_freqmin)
            param_layout.addWidget(freq_minDB)
            param_layout.addWidget(label_freqmax)
            param_layout.addWidget(freq_maxDB)
            param_layout.addWidget(zero_phaseCB)
            param_layout.addWidget(label_number_poles)
            param_layout.addWidget(orderSB)

            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.REMOVE_RESPONSE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.REMOVE_RESPONSE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)

            label_freqmin = pw.QLabel("Corner Freq min")
            label_freqmax = pw.QLabel("Corner Freq max")
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

            if len(params) > 5:
                freq_minDB1.setValue(params[0])
                freq_minDB2.setValue(params[1])
                freq_maxDB1.setValue(params[2])
                freq_maxDB2.setValue(params[3])
                water_levelDB.setValue(params[4])
                combo_param.setCurrentText(params[5])

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

        elif action is ActionEnum.DIFFERENTIATE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.DIFFERENTIATE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            combo_param = pw.QComboBox()
            combo_param.addItems(['gradient'])

            if len(params) > 0:
                combo_param.setCurrentText(params[0])

            param_layout.addWidget(combo_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.INTEGRATE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, pw.QTableWidgetItem(ActionEnum.INTEGRATE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            combo_param = pw.QComboBox()
            combo_param.addItems(['cumtrapz', 'spline'])

            if len(params) > 0:
                combo_param.setCurrentText(params[0])

            param_layout.addWidget(combo_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.ADD_WHITE_NOISE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1,
                                     pw.QTableWidgetItem(ActionEnum.ADD_WHITE_NOISE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            label_power_Db = pw.QLabel("Noise Power [db]")
            power = pw.QDoubleSpinBox()
            power.setMinimum(1)
            power.setSingleStep(1)
            power.setMaximum(100)

            if len(params) > 0:
                power.setValue(params[0])

            param_layout.addWidget(label_power_Db)
            param_layout.addWidget(power)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.WHITENING:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1,
                                     pw.QTableWidgetItem(ActionEnum.WHITENING.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            label_freq_min = pw.QLabel("Spectral Width [Hz]")
            freq_min = pw.QDoubleSpinBox()
            freq_min.setMinimum(0)
            freq_min.setSingleStep(0.01)

            if len(params) > 1:
                freq_min.setValue(params[0])

            param_layout.addWidget(label_freq_min)
            param_layout.addWidget(freq_min)

            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.REMOVE_SPIKES:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1,
                                     pw.QTableWidgetItem(ActionEnum.REMOVE_SPIKES.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            label_time = pw.QLabel("Temporal Window [s]")
            label_sigma = pw.QLabel("Threshold [standard deviations]")

            time_window = pw.QDoubleSpinBox()
            time_window.setMinimum(0)
            time_window.setSingleStep(0.1)

            sigma_window = pw.QSpinBox()
            sigma_window.setMinimum(1)
            sigma_window.setMinimum(3)
            sigma_window.setSingleStep(1)

            if len(params) > 1:
                time_window.setValue(params[0])
                sigma_window.setValue(params[1])

            param_layout.addWidget(label_time)
            param_layout.addWidget(time_window)
            param_layout.addWidget(label_sigma)
            param_layout.addWidget(sigma_window)

            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.TNOR:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1,
                                     pw.QTableWidgetItem(ActionEnum.TNOR.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            time_window_label = pw.QLabel("time window")
            time_window = pw.QDoubleSpinBox()
            time_window.setMinimum(0)
            time_window.setSingleStep(0.1)
            method_label = pw.QLabel("method")
            combo_param = pw.QComboBox()
            combo_param.addItems(['time normalization', '1bit', 'clipping', 'clipping iteration'])

            if len(params) > 1:
                time_window.setValue(params[0])
                combo_param.setCurrentText(params[1])

            param_layout.addWidget(time_window_label)
            param_layout.addWidget(time_window)
            param_layout.addWidget(method_label)
            param_layout.addWidget(combo_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.WAVELET_DENOISE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1,
                                     pw.QTableWidgetItem(ActionEnum.WAVELET_DENOISE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            family_label = pw.QLabel("wavelet family")
            param_layout.addWidget(family_label)
            combo_param = pw.QComboBox()
            combo_param.addItems(['db2', 'db4', 'db6', 'db8', 'db10',  'db12',  'db14',  'db16',  'db18', 'db19', 'db20',
               'sym2', 'sym4', 'sym6', 'sym8',  'sym10',  'sym12',  'sym14',  'sym16',  'sym18',  'sym20',
                'coif2', 'coif3', 'coif4', 'coif6', 'coif8', 'coif10', 'coif12', 'coif14', 'coif16',
                'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3',
                                  'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8'])
            param_layout.addWidget(combo_param)
            threshold_label = pw.QLabel("threshold")
            param_layout.addWidget(threshold_label)
            threshold= pw.QDoubleSpinBox()
            threshold.setMinimum(0)
            threshold.setSingleStep(0.01)

            if len(params) > 1:
                combo_param.setCurrentText(params[0])
                threshold.setValue(params[1])

            param_layout.addWidget(threshold)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)


        elif action is ActionEnum.RESAMPLE:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1,
                                     pw.QTableWidgetItem(ActionEnum.RESAMPLE.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            sampling_rate_label = pw.QLabel("new sampling rate [sps]")
            sampling_rate = pw.QDoubleSpinBox()
            sampling_rate.setMinimum(0)
            sampling_rate.setSingleStep(0.01)
            pre_filterCB = pw.QCheckBox("pre-filter")

            if len(params) > 1:
                sampling_rate.setValue(params[0])
                pre_filterCB.setChecked(params[1])

            param_layout.addWidget(sampling_rate_label)
            param_layout.addWidget(sampling_rate)
            param_layout.addWidget(pre_filterCB)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.FILL_GAPS:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1,
                                     pw.QTableWidgetItem(ActionEnum.FILL_GAPS.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            method_label = pw.QLabel("method")
            combo_param = pw.QComboBox()
            combo_param.addItems(['latest','interpolate'])

            if len(params) > 0:
                combo_param.setCurrentText(params[0])

            param_layout.addWidget(method_label)
            param_layout.addWidget(combo_param)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.SMOOTHING:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1,
                                     pw.QTableWidgetItem(ActionEnum.SMOOTHING.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)
            method_label = pw.QLabel("method")
            combo_param = pw.QComboBox()
            combo_param.addItems(['mean', 'gaussian', 'tkeo'])
            k_label = pw.QLabel("Time window [s]")

            k_value = pw.QDoubleSpinBox()
            k_value.setMinimum(0.0)
            k_value.setSingleStep(0.1)
            fwhm_label = pw.QLabel("FWHM [s]")
            param_layout.addWidget(k_label)
            fwhm = pw.QDoubleSpinBox()
            fwhm.setMinimum(0.00)
            fwhm.setSingleStep(0.01)

            if len(params) > 0:
                combo_param.setCurrentText(params[0])
                k_value.setValue(params[1])
                fwhm.setValue(params[2])

            param_layout.addWidget(method_label)
            param_layout.addWidget(combo_param)
            param_layout.addWidget(k_label)
            param_layout.addWidget(k_value)
            param_layout.addWidget(fwhm_label)
            param_layout.addWidget(fwhm)

            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

        elif action is ActionEnum.WIENER:
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
            self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1,
                                     pw.QTableWidgetItem(ActionEnum.WIENER.value))
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 0, order_widget)

            time_window_label = pw.QLabel("Time window [s]")

            time_window = pw.QDoubleSpinBox()
            time_window.setMinimum(0.0)
            time_window.setSingleStep(0.1)
            noise_label = pw.QLabel("Noise power [number of std]")
            param_layout.addWidget(time_window_label)
            noise = pw.QSpinBox()
            noise.setMinimum(0)
            noise.setSingleStep(1)

            if len(params) > 0:

                time_window.setValue(params[1])
                noise.setValue(params[2])

            param_layout.addWidget(time_window_label)
            param_layout.addWidget(time_window)
            param_layout.addWidget(noise_label)
            param_layout.addWidget(noise)

            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, 2, param_widget)

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

            elif (action == ActionEnum.WIENER.value):
                spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().value()
                spin_value2 = self.tableWidget.cellWidget(i, 2).layout().itemAt(3).widget().value()
                parameters.append([action, spin_value1, spin_value2])

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

            elif (action == ActionEnum.ADD_WHITE_NOISE.value):
                spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().value()
                parameters.append([action, spin_value1])

            elif (action == ActionEnum.WHITENING.value):
                spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().value()
                #spin_value2 = self.tableWidget.cellWidget(i, 2).layout().itemAt(3).widget().value()
                #parameters.append([action, spin_value1,spin_value2])
                parameters.append([action, spin_value1])

            elif (action == ActionEnum.REMOVE_SPIKES.value):
                spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().value()
                spin_value2 = self.tableWidget.cellWidget(i, 2).layout().itemAt(3).widget().value()
                parameters.append([action, spin_value1, spin_value2])

            elif (action == ActionEnum.TNOR.value):
                spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().value()
                combo_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(3).widget().currentText()
                parameters.append([action, spin_value1, combo_value1])

            elif (action == ActionEnum.WAVELET_DENOISE.value):
                combo_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().currentText()
                spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(3).widget().value()
                parameters.append([action, combo_value1, spin_value1])

            elif (action == ActionEnum.RESAMPLE.value):
                spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().value()
                check_box1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(2).widget().isChecked()
                parameters.append([action, spin_value1, check_box1])

            elif (action == ActionEnum.FILL_GAPS.value):
                 combo_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().currentText()
                 parameters.append([action, combo_value1])

            elif (action == ActionEnum.SMOOTHING.value):
                 combo_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(1).widget().currentText()
                 spin_value1 = self.tableWidget.cellWidget(i, 2).layout().itemAt(3).widget().value()
                 spin_value2 = self.tableWidget.cellWidget(i, 2).layout().itemAt(5).widget().value()
                 parameters.append([action, combo_value1, spin_value1, spin_value2])

        return parameters
