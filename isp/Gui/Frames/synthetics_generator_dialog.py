from isp.Gui import pw, pyc, qt
from isp.Gui.Frames import UiSyntheticsGeneratorDialog, SettingsLoader
from isp.Gui.Utils.pyqt_utils import add_save_load

from obspy.clients.syngine import Client

from concurrent.futures.thread import ThreadPoolExecutor
import os
import pickle
from sys import platform
from datetime import datetime

@add_save_load()
class SyntheticsGeneratorDialog(pw.QDialog, UiSyntheticsGeneratorDialog, metaclass=SettingsLoader):
    def __init__(self, parent=None):
        super(SyntheticsGeneratorDialog, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowIcon(parent.windowIcon())

        self.setWindowTitle('Synthetics Generator Dialog')

        self.progress_dialog = pw.QProgressDialog(self)
        self.progress_dialog.setRange(0,0)
        self.progress_dialog.setLabelText('Requesting synthetic waveforms.')
        self.progress_dialog.setWindowIcon(self.windowIcon())
        self.progress_dialog.setWindowTitle(self.windowTitle())
        self.progress_dialog.close()
        self._client = Client()
        self.comboBoxModels.addItems(self._client.get_available_models().keys())
        self.radioButtonMT.toggled.connect(self._buttonMTFPClicked)
        self.radioButtonMT.setChecked(True)
        self._buttonMTFPClicked(True)
        self.pushButtonAddStation.clicked.connect(self._buttonAddStationClicked)
        self.pushButtonRemStation.setEnabled(False)
        self.pushButtonRemStation.clicked.connect(self._buttonRemoveStationClicked)
        self.tableWidget.setSelectionBehavior(pw.QAbstractItemView.SelectRows)
        self.tableWidget.itemSelectionChanged.connect(
            lambda : self.pushButtonRemStation.setEnabled(
                bool(self.tableWidget.selectedIndexes())))
        self.buttonBox.clicked.connect(self._buttonBoxClicked)

        # TODO Add inventory for selecting stations database location.
       
    def closeEvent(self, ce):
        self.load_values()

    def __load__(self):
        self.load_values()

    def _buttonBoxClicked(self, button):
        if (pw.QDialogButtonBox.ApplyRole == self.buttonBox.buttonRole(button)):
            if not self.tableWidget.rowCount() :
                pw.QMessageBox.warning(self, self.windowTitle(), 
                                       "At least one station must be inserted")
                return
            bulk = []
            for i in range(self.tableWidget.rowCount()):
                lat_str = self.tableWidget.item(i,0).data(0)
                lon_str = self.tableWidget.item(i,1).data(0)

                if lat_str and lon_str:
                    try:
                        lat = float(lat_str)
                        lon = float(lon_str)
                    except ValueError:
                        pw.QMessageBox.warning(self, self.windowTitle(), 
                                               f"Lat/lon not valid numbers at row {i + 1}")
                        return
                    if not(-90 <= lat <= 90 and -180 <= lon <= 180):
                        pw.QMessageBox.warning(self, self.windowTitle(), 
                                               f"Lat/lon out of range at row {i + 1}")
                        return

                net = self.tableWidget.item(i,2).data(0)
                stat = self.tableWidget.item(i,3).data(0)
                if lat_str and lon_str and net and stat:
                    bulk.append({"latitude": lat, "longitude": lon, 
                                 "networkcode": net, "stationcode": stat})
                elif lat_str and lon_str and not net and not stat:
                    bulk.append([lat, lon])
                elif net and stat and not lat_str and not lon_str:
                    bulk.append([net, stat])
                else:
                    message = (
                        f"Invalid station at row {i + 1}. Must insert" 
                        f"lat/lon or net/stat or the four parameters."
                    )
                    pw.QMessageBox.warning(self, self.windowTitle(), 
                                           message)
                    return

            params = {"model" : self.comboBoxModels.currentText(),
                      "bulk" : bulk,
                      "sourcelatitude" : self.doubleSBSrcLat.value(),
                      "sourcelongitude" : self.doubleSBSrcLon.value(),
                      "sourcedepthinmeters" : self.doubleSBDep.value(),
                      "units": self.unitsCB.currentText(),
                      "origintime" : self.dateTimeEditOrigin.dateTime().toPyDateTime(),
                      "starttime" : self.dateTimeEditStart.dateTime().toPyDateTime(),
                      "endtime" : self.dateTimeEditEnd.dateTime().toPyDateTime(),
                      "format" : "miniseed"}

            if self.radioButtonMT.isChecked() :
                mrr_str = self.lineEditMrr.text()
                mtt_str = self.lineEditMtt.text()
                mpp_str = self.lineEditMpp.text()
                mrt_str = self.lineEditMrt.text()
                mrp_str = self.lineEditMrp.text()
                mtp_str = self.lineEditMtp.text()
                if (not mrr_str or not mtt_str or not mpp_str or 
                    not mrt_str or not mrp_str or not mtp_str):
                    pw.QMessageBox.warning(self, self.windowTitle(), 
                                           "Some moment tensor value is missing")
                    return
                try:
                    mrr = float(mrr_str)
                    mtt = float(mtt_str)
                    mpp = float(mpp_str)
                    mrt = float(mrt_str)
                    mrp = float(mrp_str)
                    mtp = float(mtp_str)
                except ValueError:
                    pw.QMessageBox.warning(self, self.windowTitle(), 
                                           "Moment tensor values are invalid")
                    return
                mtparams = [mrr, mtt, mpp, mrt, mrp, mtp]
                params["sourcemomenttensor"] = mtparams
            else :
                fpparams = [self.doubleSBStrike.value(),
                            self.doubleSBDip.value(),
                            self.doubleSBRake.value()]
                if self.lineEditM0.text():
                    try:
                        m0 = float(self.lineEditM0.text())
                        fpparams.append(m0)
                    except:
                        pw.QMessageBox.warning(self, self.windowTitle(), 
                                               "M0 value is invalid")
                        return

                params["sourcedoublecouple"] = fpparams

     
            with ThreadPoolExecutor(1) as executor:
                f = executor.submit(lambda : self._requestSynthetics(params))
                r = self.progress_dialog.exec()
                if r == pw.QDialog.Accepted :
                    st = f.result()
                    if not st:
                        pw.QMessageBox.warning(self, self.windowTitle(), 
                                           "Synthetics generator request failed.")
                        return

                    if "darwin" == platform:
                        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory')
                    else:
                        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', '',
                                                                       pw.QFileDialog.DontUseNativeDialog)
                    if not dir_path:
                        return

                    for tr in st:
                        path_output = os.path.join(dir_path, tr.id)
                        tr.write(path_output, format="MSEED")
                    current = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    name = 'generation_params' + current + '.pkl'
                    with open(os.path.join(dir_path, name), 'wb') as f:
                        pickle.dump(params, f)

                    self.save_values()

                    pw.QMessageBox.information(self, self.windowTitle(),
                                           "Synthetics generation done !!!")
            

    def _requestSynthetics(self, params):
        try :
            st = self._client.get_waveforms_bulk(**params)
        except Exception as e:
            print(e)

        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', qt.QueuedConnection)
        return st

    def _buttonAddStationClicked(self):
        self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1)
        item = pw.QTableWidgetItem()
        self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 0, item)
        item = pw.QTableWidgetItem()
        self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, item)
        item = pw.QTableWidgetItem()
        self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 2, item)
        item = pw.QTableWidgetItem()
        self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 3, item)

    def _buttonRemoveStationClicked(self):
        rows = self.tableWidget.selectionModel().selectedRows()
        if rows:
            rows = [r.row() for r in rows]
            rows.sort(reverse=True)
            for r in rows:
                self.tableWidget.removeRow(r)

    def _buttonMTFPClicked(self, mt_checked):
        # When button 0 (MT) is clicked, activate MT parameters
        # and disable FP parameters and viceversa.
        self.groupBoxMT.setEnabled(mt_checked)
        self.groupBoxFP.setEnabled(not mt_checked)
