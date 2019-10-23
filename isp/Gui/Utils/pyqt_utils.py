import os

from PyQt5 import uic

from isp import UIS_PATH
from isp.Gui import pyc, pw


def load_ui_designers(ui_file):

    ui_path = os.path.join(UIS_PATH, ui_file)
    if not os.path.isfile(ui_path):
        raise FileNotFoundError("The file {} can't be found at the location {}".
                                format(ui_file, UIS_PATH))
    ui_class, _ = uic.loadUiType(ui_path, from_imports=True, import_from='isp.resources')
    return ui_class


class BindPyqtObject(pyc.QObject):
    """
    Bind a pyqt object to this class
    """

    def __init__(self, pyqt_obj, callback=None):
        """
        Create an instance that has value property bind to the pyqt object value.

        :param pyqt_obj: The pyqt object to bind to, i.e: QLineEdit, QSpinBox, etc..

        :param callback: A callback function that is called when the pyqt object change value.
        """
        super().__init__()

        if isinstance(pyqt_obj, pw.QLineEdit):
            value = pyqt_obj.text()
            pyqt_obj.textChanged.connect(self.__valueChanged)
            self.__set_value = lambda v: pyqt_obj.setText(v)

        elif isinstance(pyqt_obj, pw.QSpinBox) or isinstance(pyqt_obj, pw.QDoubleSpinBox):
            value = pyqt_obj.value()
            pyqt_obj.valueChanged.connect(self.__valueChanged)
            self.__set_value = lambda v: pyqt_obj.setValue(v)

        elif isinstance(pyqt_obj, pw.QComboBox):
            value = pyqt_obj.currentText()
            pyqt_obj.currentTextChanged.connect(self.__valueChanged)
            self.__set_value = lambda v: pyqt_obj.setCurrentText(v)

        else:
            raise AttributeError("The object {} don't have a valid type".format(pyqt_obj))

        self.__value = value
        self.__callback_val_changed = callback
        self.set_valueChanged_callback(callback)

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        if self.__value != value:
            self.__value = value
            self.__set_value(value)
            if self.__callback_val_changed:
                self.__callback_val_changed(value)

    @pyc.pyqtSlot(float)
    @pyc.pyqtSlot(int)
    @pyc.pyqtSlot(str)
    def __valueChanged(self, value):
        self.value = value

    def set_valueChanged_callback(self, func):
        if func is not None:
            self.__callback_val_changed = lambda v: func(v)

