#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
load_picks_tool
"""

import os
from isp import ROOT_DIR
from isp.Gui import pw, pyc
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiLoadPick
from isp.Gui.Utils.pyqt_utils import add_save_load
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


    def load_file(self):
        file_selected = pw.QFileDialog.getOpenFileName(self, "Select Project", ROOT_DIR)[0]
        if isinstance(file_selected, str) and os.path.isfile(file_selected):
            self.pickpathTE.setText(file_selected)
            md = MessageDialog(self)
            md.set_info_message("Fill time span (<=24h) and num max of picks")

    def load_picks(self, default=True, reset=True):
        print("Importing Picks")
        self.pick_times_imported = {}
        md = MessageDialog(self)
        self.pick_times_imported = MseedUtil.get_NLL_phase_picks(input_file=self.pickpathTE.text())

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


        if self.pm.is_empty():
            md.set_info_message("No picks incorporated")

        else:
            self.pm.save()
            md.set_info_message("Loaded Picks Done, Plot seismograms to see your picks")
            self.signal_picks.emit()








