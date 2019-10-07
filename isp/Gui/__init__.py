import PyQt5 as PyQt
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


# pyqt5 vars

pyqt = PyQt
pqg = QtGui
pw = QtWidgets
qt = Qt

from isp.Gui.uis_frames import UiMainFrame, UiSeismogramFrame
from isp.Gui.main import MainFrame
from isp.Gui.controllers import Controller

controller = Controller()

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    controller.open_main_window()
    sys.exit(app.exec_())

