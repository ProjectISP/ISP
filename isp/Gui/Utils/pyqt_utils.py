import os
from cmath import isinf

from PyQt5 import uic

from isp import UIS_PATH
from isp.Gui import pyc, pw, user_preferences


def load_ui_designers(ui_file):

    ui_path = os.path.join(UIS_PATH, ui_file)
    if not os.path.isfile(ui_path):
        raise FileNotFoundError("The file {} can't be found at the location {}".
                                format(ui_file, UIS_PATH))
    ui_class, _ = uic.loadUiType(ui_path, from_imports=True, import_from='isp.resources')
    return ui_class


def save_preferences(pyqt_object, ui_name=None):
    """
    Save the fields QDoubleSpinBox, QSpinBox, QLineEdit from the pyqt_object into a file.

    :param pyqt_object: A pyqt object like QFrame or QWidget.
    :param ui_name: The name to use in the group, If not given it will use the object name to group it.
    :return:
    """

    ui_name = type(pyqt_object).__name__ if ui_name is None else ui_name
    user_preferences.beginGroup(ui_name)
    for key, item in pyqt_object.__dict__.items():
        if isinstance(item, pw.QDoubleSpinBox) or isinstance(item, pw.QSpinBox):
            user_preferences.setValue(key, item.value())
        elif isinstance(item, pw.QLineEdit):
            user_preferences.setValue(key, item.text())
        elif hasattr(item, "save_values"):
            item.save_values()

    user_preferences.endGroup()


def load_preferences(pyqt_object, ui_name=None):
    """
    Load data from user_pref to fields QDoubleSpinBox, QSpinBox, QLineEdit

    :param pyqt_object: A pyqt object like QFrame or QWidget.
    :param ui_name: The name to use in the group, If not given it will use the object name to group it.
    :return:
    """
    ui_name = type(pyqt_object).__name__ if ui_name is None else ui_name
    user_preferences.beginGroup(ui_name)
    for key, item in pyqt_object.__dict__.items():
        value = user_preferences.value(key)
        if value and not "":
            if isinstance(item, pw.QDoubleSpinBox):
                item.setValue(float(value))
            elif isinstance(item, pw.QSpinBox):
                item.setValue(int(value))
            elif isinstance(item, pw.QLineEdit):
                item.setText(value)
        elif hasattr(item, "load_values"):
            item.load_values()

    user_preferences.endGroup()


def save_values(self):
    if hasattr(self, "parent_name"):
        save_preferences(self, ui_name=self.parent_name)
    else:
        save_preferences(self)


def load_values(self):
    if hasattr(self, "parent_name"):
        load_preferences(self, ui_name=self.parent_name)
    else:
        load_preferences(self)


def add_save():
    def wrapper(cls):
        name = save_values.__name__
        setattr(cls, name, eval(name))
        return cls
    return wrapper


def add_load():
    def wrapper(cls):
        name = load_values.__name__
        setattr(cls, name, eval(name))
        return cls
    return wrapper


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
            self.__set_value(value)

    @pyc.pyqtSlot(float)
    @pyc.pyqtSlot(int)
    @pyc.pyqtSlot(str)
    def __valueChanged(self, value):
        self.__value = value
        if self.__callback_val_changed:
            self.__callback_val_changed(value)

    def set_valueChanged_callback(self, func):
        if func is not None:
            self.__callback_val_changed = lambda v: func(v)

    def unblind_valueChanged(self):
        self.__callback_val_changed = None


