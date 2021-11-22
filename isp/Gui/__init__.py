import sys
import PyQt5 as PyQt
from PyQt5 import QtGui, QtWebEngineWidgets
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from isp.db import db
import os
#from isp.db.models import FirstPolarityModel, MomentTensorModel, EventArrayModel, PhaseInfoModel,\
#    EventLocationModel, ArrayAnalysisModel
# pyqt5 vars

pyqt = PyQt
pqg = QtGui
pw = QtWidgets
pyc = QtCore
qt = Qt
qteng = QtWebEngineWidgets  # TODO Why use this? We should remove if not necessary.
# file to save ui fields
user_preferences = pyc.QSettings(pyc.QSettings.NativeFormat, pyc.QSettings.UserScope, "isp", "user_pref")
print(user_preferences.fileName())

def controller():
    from isp.Gui.controllers import Controller
    return Controller()


def get_settings_file():
    return user_preferences.fileName()


def start_isp():
    from isp.Gui.StyleLib import set_isp_mpl_style_file
    from isp import app_logger

    # Initializa DB
    dir_path = os.path.dirname(os.path.abspath(__file__))
    db.set_db_url("sqlite:///{}/isp_test.db".format(dir_path))
    db.start()

    # print(user_preferences.fileName())
    app = QtWidgets.QApplication(sys.argv)
    controller().open_main_window()
    # sets the matplolib style from file isp.mplstyle.
    set_isp_mpl_style_file()

    app_logger.info("ISP GUI Started")
    app_logger.info("User preferences is at: {}".format(get_settings_file()))
    sys.exit(app.exec_())


def except_hook(cls, exception, exc_traceback):
    sys.__excepthook__(cls, exception, exc_traceback)
    controller().exception_parse(cls, exception, exc_traceback)

    #print(str(exception))


if __name__ == '__main__':
    sys.excepthook = except_hook
    start_isp()

