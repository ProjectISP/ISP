from isp.Gui import pw, pyc, qt
from isp.Gui.Frames import UiSyntheticsGeneratorDialog

from obspy.clients.syngine import Client

from concurrent.futures.thread import ThreadPoolExecutor
import os

class SyntheticsGeneratorDialog(pw.QDialog, UiSyntheticsGeneratorDialog):
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


    def _buttonBoxClicked(self, button):
        if (pw.QDialogButtonBox.ApplyRole == self.buttonBox.buttonRole(button)):
            if not self.tableWidget.rowCount() :
                pw.QMessageBox.warning(self, self.windowTitle(), 
                                       "At least one station must be inserted")
                return
            bulk = []
            for i in range(self.tableWidget.rowCount()):
                bulk.append([self.tableWidget.item(i,0).data(0), 
                             self.tableWidget.item(i,1).data(0)])

            params = {"model" : self.comboBoxModels.currentText(),
                      "bulk" : bulk,
                      "sourcelatitude" : self.doubleSBSrcLat.value(),
                      "sourcelongitude" : self.doubleSBSrcLon.value(),
                      "sourcedepthinmeters" : self.doubleSBDep.value(),
                      "origintime" : self.dateTimeEditOrigin.dateTime().toPyDateTime(),
                      "starttime" : self.dateTimeEditStart.dateTime().toPyDateTime(),
                      "endtime" : self.dateTimeEditEnd.dateTime().toPyDateTime(),
                      "format" : "miniseed"}

            if self.buttonGroupMTFP.checkedId() == 0 :
                mtparams = [self.doubleSBMrr.value(),
                            self.doubleSBMtt.value(),
                            self.doubleSBMpp.value(),
                            self.doubleSBMrt.value(),
                            self.doubleSBMrp.value(),
                            self.doubleSBMtp.value()]
                params["sourcemomenttensor"] = mtparams
            else :
                fpparams = [self.doubleSBStrike.value(),
                            self.doubleSBDip.value(),
                            self.doubleSBRake.value(),
                            self.doubleSBM0.value()]
                params["sourcedoublecouple"] = fpparams

     
            with ThreadPoolExecutor(1) as executor:
                f = executor.submit(lambda : self._requestSynthetics(params))
                r = self.progress_dialog.exec()
                if r == pw.QDialog.Accepted :
                    st = f.result()
                    dir_path = pw.QFileDialog.getExistingDirectory(
                        self, 'Select Output Directory')
                    for tr in st:
                        path_output =  os.path.join(dir_path, tr.id)
                        tr.write(path_output, format="MSEED")

            

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
        item.setData(0, 0.0)
        self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 0, item)
        item = pw.QTableWidgetItem()
        item.setData(0, 0.0)
        self.tableWidget.setItem(self.tableWidget.rowCount() - 1, 1, item)

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

