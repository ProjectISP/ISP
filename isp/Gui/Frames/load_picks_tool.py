#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
load_picks_tool
"""

import os
from isp import ROOT_DIR, PICKING_DIR
from isp.Gui import pw, pyc
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiLoadPick
from isp.Gui.Utils.pyqt_utils import add_save_load, convert_qdatetime_utcdatetime
from isp.Utils import MseedUtil
from isp.earthquakeAnalysis import PickerManager


@add_save_load()
class LoadPick(pw.QDialog, UiLoadPick):

    signal_picks = pyc.pyqtSignal()
    def __init__(self, parent=None):
        super(LoadPick, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

        self.test_pick_attribute = "something"
        self.loadFileBtn.clicked.connect(self.load_file)
        self.loadpicksBtn.clicked.connect(self.load_picks)
        self.defaultpickBtn.clicked.connect(self.set_default)

    def set_default(self):
        self.pickpathTE.setText(os.path.join(PICKING_DIR, "output.txt"))

    def load_file(self):
        file_selected = pw.QFileDialog.getOpenFileName(self, "Select Project", ROOT_DIR)[0]
        if isinstance(file_selected, str) and os.path.isfile(file_selected):
            self.pickpathTE.setText(file_selected)
            md = MessageDialog(self)
            md.set_info_message("Fill time span (<=24h) and num max of picks")

    def load_picks(self, default=True, reset=True):
        print("Importing Picks")
        self.pick_times_imported = {}

        def _fmt_utc(t):
            return t.isoformat() if t is not None else "—"

        if self.filterTimeRB.isChecked():
            starttime = convert_qdatetime_utcdatetime(self.dateTimeEdit1)
            endtime = convert_qdatetime_utcdatetime(self.dateTimeEdit2)
            self.pick_times_imported = MseedUtil.get_NLL_phase_picks(input_file=self.pickpathTE.text(),
                                                                     starttime=starttime,
                                                                     endtime=endtime)
        else:
            self.pick_times_imported = MseedUtil.get_NLL_phase_picks(input_file=self.pickpathTE.text())

        info = MseedUtil.summarize_picks(self.pick_times_imported)
        earliest = info.get("earliest_pick")
        latest = info.get("latest_pick")
        duration = info.get("span_seconds")
        if reset:
            self.pm = PickerManager()

        # save data
        if len(self.pick_times_imported) > 0:
            print("Creating pick manager")
            for key in self.pick_times_imported:
                pick = self.pick_times_imported[key]
                for j in range(len(pick)):
                    pick_value = pick[j][1]

                    self.pm.add_data(pick_value, pick[j][5], pick[j][7], key.split(".")[0], pick[j][0],
                                     First_Motion=pick[j][3], Component=pick[j][2])

        md = MessageDialog(self)
        if self.pm.is_empty():
            md.set_info_message("No picks incorporated")
        else:
            self.pm.save()
            body = (
                f"Total Picks: {info.get('total_picks', 0)}\n"
                f"Stations: {info.get('stations', 0)}\n"
                f"Earliest pick: {_fmt_utc(earliest)}\n"
                f"Latest pick: {_fmt_utc(latest)}\n"
                f"Span Time: {info.get('span_readable', '0 s')}\n"
            )

            if not self.filterTimeRB.isChecked():

                md.set_info_message(
                    "Loaded Picks Done, Warnning no filter applied, Might be too much picks. \n"
                    "Plot seismograms to see your picks",
                    body
                )

            elif self.filterTimeRB.isChecked() and duration > 3600:
                md.set_info_message(
                    "Loaded Picks Done, Warnning more than 1 hour picks, Might be too much picks. \n"
                    "Plot seismograms to see your picks",
                    body
                )

            else:

                md.set_info_message(
                    "Loaded Picks Done \n"
                    "Plot seismograms to see your picks",
                    body
                )

            self.signal_picks.emit()









