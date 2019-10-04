import sys

import PyQt5 as PyQt
from PyQt5.QtWidgets import QApplication

from isp.Gui.Utils.pyqt_utils import load_ui_designers

pyqt = PyQt

# Add new ui designers here. The *.ui files must be placed inside resources/designer_uis
UiMainFrame = load_ui_designers("MainFrame.ui")


def window():
    from isp.Gui.main import MainFrame

    app = QApplication(sys.argv)
    # Start the ui designer
    win = MainFrame()

    win.show()
    sys.exit(app.exec_())


window()
