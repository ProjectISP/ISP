import subprocess
import sys

from PyQt5 import QtCore, QtWidgets

from isp.Gui import MainFrame
from isp.Gui.frames import SeismogramFrame


class Controller:

    def __init__(self):
        self.main_frame = None
        self.seismogram_frame = None

    def open_main_window(self):
        # Start the ui designer
        self.main_frame = MainFrame()
        # bind clicks
        self.main_frame.SeismogramAnalysis.clicked.connect(self.open_seismogram_window)

        # show frame
        self.main_frame.show()

    def open_seismogram_window(self):
        # Start the ui designer
        self.seismogram_frame = SeismogramFrame()
        self.seismogram_frame.show()
