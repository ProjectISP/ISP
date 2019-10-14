import PyQt5 as PyQt
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import Qt


# pyqt5 vars

pyqt = PyQt
pqg = QtGui
pw = QtWidgets
pyc = QtCore
qt = Qt

# file to save ui fields
user_preferences = pyc.QSettings("isp", "user_pref")

from isp.Gui.controllers import Controller

controller = Controller()


def get_settings_file():
    return user_preferences.fileName()


if __name__ == '__main__':
    import sys
    from isp import app_logger

    print(user_preferences.fileName())
    app = QtWidgets.QApplication(sys.argv)
    controller.open_main_window()
    app_logger.info("ISP GUI Started")
    sys.exit(app.exec_())

