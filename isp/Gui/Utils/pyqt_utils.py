import os
from contextlib import suppress
from datetime import datetime


from PyQt5 import uic
from obspy import UTCDateTime

from isp import UIS_PATH
from isp.Gui import pyc, pw, user_preferences, pqg


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
        elif isinstance(item, pw.QDateTimeEdit):
            user_preferences.setValue(key, item.dateTime().toPyDateTime())
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
        if hasattr(item, "load_values"):
            item.load_values()
        else:
            value = user_preferences.value(key)
            if value is not None:
                with suppress(TypeError):
                    str(value, "utf-8")
                value = value.strip() if type(value) == str else value
                if value is not "" or type(value) is not str:
                    if isinstance(item, pw.QDoubleSpinBox):
                        item.setValue(float(value))
                    elif isinstance(item, pw.QSpinBox):
                        item.setValue(int(value))
                    elif isinstance(item, pw.QLineEdit):
                        item.setText(value)
                    elif isinstance(item, pw.QDateTimeEdit):
                        set_qdatetime(value, item)

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


def add_save_load():
    """
    Class decorator to add save_values and load_values methods. Using this decorator will make this class
    to save and load its pyqt widgets values.

    :return:
    """
    def wrapper(cls):
        name = save_values.__name__
        setattr(cls, name, eval(name))
        name = load_values.__name__
        setattr(cls, name, eval(name))
        return cls

    return wrapper


def set_qdatetime(time, pyqt_time_object: pw.QDateTimeEdit):
    """
    Set the datetime to an edit time qt object.

    :param time: A str or obspy.UTCDateTime.

    :param pyqt_time_object: A QDateTimeEdit pyqt object to set the time.

    :return:
    """
    if type(time) is str:
        time = UTCDateTime(time)
    elif isinstance(time, UTCDateTime):
        time = time.datetime
    elif not isinstance(time, datetime):
        raise ValueError("Time must by either str, UTCDatetime or datetime")

    pyqt_time_object.setDateTime(time)


def convert_qdatetime_utcdatetime(q_datetime_edit: pw.QDateTimeEdit):
    """
    Convert the datetime from QDateTimeEdit widget to a obspy.UTCDatetime

    :param q_datetime_edit: The QDateTimeEdit widget
    :return: The obspy.UTCDatetime
    """
    py_time = q_datetime_edit.dateTime().toPyDateTime()
    return UTCDateTime(py_time)


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
        self.pyqt_obj = pyqt_obj

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

    @staticmethod
    def __validate_event(event: pqg.QDropEvent):
        data = event.mimeData()
        urls = data.urls()
        accept = True
        for url in urls:
            if not url.isLocalFile():
                accept = False

        if accept:
            event.acceptProposedAction()

    def __drop_event(self, event: pqg.QDropEvent, func):
        if func:
            func(event, self)

    def accept_dragFile(self, drop_event_callback=None):
        """
        Makes this object accept drops.

        :param drop_event_callback: A callback function that is called when the object is drop. The callback must
            have the parameters event and a BindPyqtObject object.

        :return:
        """
        if hasattr(self.pyqt_obj, "setDragEnabled"):
            self.pyqt_obj.setDragEnabled(True)
            self.pyqt_obj.dragEnterEvent = self.__validate_event
            self.pyqt_obj.dragMoveEvent = self.__validate_event
            self.pyqt_obj.dropEvent = lambda event: self.__drop_event(event, drop_event_callback)
        else:
            raise AttributeError("The object {} doesn't have the method setDragEnabled".
                                 format(self.pyqt_obj.objectName()))
