import PyQt5 as PyQt
from PyQt5 import QtGui, QtWebEngineWidgets
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

# pyqt5 vars

pyqt = PyQt
pqg = QtGui
pw = QtWidgets
pyc = QtCore
qt = Qt
qteng = QtWebEngineWidgets  # TODO Why use this? We should remove if not necessary.
# file to save ui fields
user_preferences = pyc.QSettings(pyc.QSettings.NativeFormat, pyc.QSettings.UserScope, "isp", "user_pref")


def controller():
    from isp.Gui.controllers import Controller
    return Controller()


def get_settings_file():
    return user_preferences.fileName()


if __name__ == '__main__':
    import sys
    from isp.Gui.StyleLib import set_isp_mpl_style_file
    from isp import app_logger

    # print(user_preferences.fileName())
    app = QtWidgets.QApplication(sys.argv)
    controller().open_main_window()
    # sets the matplolib style from file isp.mplstyle.
    set_isp_mpl_style_file()

    app_logger.info("ISP GUI Started")
    app_logger.info("User preferences is at: {}".format(get_settings_file()))
    sys.exit(app.exec_())

