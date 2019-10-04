import os
import subprocess as sp

from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow

from isp import ROOT_DIR, IMAGES_PATH
from isp.Gui import UiMainFrame


class MainFrame(QMainWindow, UiMainFrame):

    def __init__(self):
        super(MainFrame, self).__init__()

        # Set up the user interface from Designer.
        self.setupUi(self)

        icon1 = QtGui.QPixmap(os.path.join(IMAGES_PATH, '02.png'))
        icon2 = QtGui.QPixmap(os.path.join(IMAGES_PATH, '03.png'))
        icon3 = QtGui.QPixmap(os.path.join(IMAGES_PATH, '04.png'))
        icon4 = QtGui.QPixmap(os.path.join(IMAGES_PATH, '05.png'))
        icon5 = QtGui.QPixmap(os.path.join(IMAGES_PATH, '01.png'))
        iconLogo = QtGui.QPixmap(os.path.join(IMAGES_PATH, 'LOGO.png'))

        self.LOGO.setPixmap(iconLogo)
        self.labelManage.setPixmap(icon1)
        self.labelseismogram.setPixmap(icon2)
        self.labelearthquake.setPixmap(icon3)
        self.labelMTI.setPixmap(icon4)
        self.labelarray.setPixmap(icon5)

        self.SeismogramAnalysis.clicked.connect(self.runSeismogram)
        self.ArrayAnalysis.clicked.connect(self.array)

    def runSeismogram(self):
        print("Hello")
        # path = ROOT_DIR + "/seismogramInspector/"
        # command = "cd "+path+";"+"python main.py"
        # sp.Popen(command, shell=True)

    def array(self):
        path = ROOT_DIR + "/arrayanalysis/"
        command = "cd " + path + ";" + "python main.py"
        sp.Popen(command, shell=True)
