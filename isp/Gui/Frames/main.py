from isp.Gui import pw, qt, user_preferences, pyc, pqg
from isp.Gui.Frames import UiMainFrame
from isp.Gui.Utils.pyqt_utils import save_preferences, load_preferences


class SettingsLoader(type(pyc.QObject), type):

    def __call__(cls, *args, **kwargs):
        """Called when you call SettingsLoader() """
        obj = type.__call__(cls, *args, **kwargs)
        obj.__load__()
        return obj


class BaseFrame(pw.QMainWindow, metaclass=SettingsLoader):

    def __init__(self):
        super().__init__()

    def __load__(self):
        """ Method called after __init__"""
        self.load()
        self.apply_shadow()

    # Press esc key event
    def keyPressEvent(self, e):
        if e.key() == qt.Key_Escape:
            self.close()

    def closeEvent(self, ce):
        self.save()
        # print("Closed")

    def save(self):
        """
        Save fields QDoubleSpinBox, QSpinBox, QLineEdit into a file.

        :return:
        """
        save_preferences(self)

    def load(self):
        """
        Load data from user_pref to fields QDoubleSpinBox, QSpinBox, QLineEdit

        :return:
        """
        load_preferences(self)

    def apply_shadow(self):
        for child in self.findChildren((pw.QPushButton, pw.QCheckBox, pw.QLineEdit)):
            ge = pw.QGraphicsDropShadowEffect()
            ge.setBlurRadius(5)
            ge.setOffset(2)
            if "noShadow".lower() not in child.objectName().lower() and \
                    "qt_spinbox_lineedit" not in child.objectName():
                child.setGraphicsEffect(ge)


class MainFrame(BaseFrame, UiMainFrame):

    def __init__(self):
        super(MainFrame, self).__init__()

        # Set up the user interface from Designer.
        self.setupUi(self)


