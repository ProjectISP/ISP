import math
import os
from datetime import datetime
from types import FunctionType

from matplotlib.lines import Line2D
from obspy import UTCDateTime

from isp.DataProcessing import SeismogramAnalysis
from isp.Gui import pw
from isp.Gui.Frames import UiPaginationWidget, UiFilterGroupBox, UiEventInfoGroupBox
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, add_save_load
from isp.Structures.structures import StationsStats
from isp.Utils import Filters


class ParentWidget:

    @staticmethod
    def set_parent(parent, obj, current_index=-1):
        if parent and (isinstance(parent, pw.QWidget) or isinstance(parent, pw.QFrame)):
            if parent.layout() is not None:
                layout = parent.layout()
                for child in parent.findChildren(pw.QTreeView):
                    child.setParent(None)
            else:
                parent.setLayout(pw.QVBoxLayout())
                layout = parent.layout()

            layout.insertWidget(current_index, obj)


class FilesView(pw.QTreeView):

    def __init__(self, root_path, parent=None, on_change_file_callback=None):
        super().__init__(parent)

        self.__file_path = None
        self.__file_name = None
        self.__on_change_file_callback = on_change_file_callback

        self.__model = pw.QFileSystemModel()
        self.__model.setReadOnly(True)

        self.setModel(self.__model)

        if self.is_valid_dir(root_path):
            self.__model.setRootPath(root_path)
            self.__parent_index = self.__model.index(root_path)
            self.setRootIndex(self.__parent_index)

        self.setColumnWidth(0, 300)
        self.hideColumn(2)
        self.hideColumn(3)
        self.setAlternatingRowColors(True)

        self.clicked.connect(self.__onClick_file)
        self.__model.directoryLoaded.connect(self.__directoryLoaded)
        self.__model.rootPathChanged.connect(self.__rootPathChanged)

        # set the parent properly
        ParentWidget.set_parent(parent, self)

    @property
    def file_path(self):
        """
        Gets the full file path.

        :return: A string containing the full path of the file
        """
        return self.__file_path

    @property
    def file_name(self):
        """
        Gets the file's name.

        :return: A string containing the name of the file.
        """
        return self.__file_name

    def __onClick_file(self, index):
        if self.__file_path != self.__model.filePath(index) and self.__on_change_file_callback:
            self.__on_change_file_callback(self.__model.filePath(index))

        self.__file_path = self.__model.filePath(index)
        self.__file_name = self.__model.fileName(index)

    def __directoryLoaded(self, path):
        index = self.__model.index(0, 0, self.__parent_index)
        self.__file_path = self.__model.filePath(index)
        self.__file_name = self.__model.fileName(index)
        if self.__on_change_file_callback:
            self.__on_change_file_callback(self.__model.filePath(index))

    def __rootPathChanged(self, root_path):
        """
        Fired when root path is changed

        :param root_path: The full path of the directory

        :return:
        """
        self.__parent_index = self.__model.index(root_path)
        self.setRootIndex(self.__parent_index)

    @staticmethod
    def is_valid_dir(dir_path):
        return os.path.isdir(dir_path)

    def set_new_rootPath(self, root_path):
        """
        Change the root directory

        :param root_path: The full path of the directory.

        :return:
        """
        if self.is_valid_dir(root_path):
            self.__model.setRootPath(root_path)

    def dragMoveEvent(self, event):
        print(event)


class Pagination(pw.QWidget, UiPaginationWidget):

    def __init__(self, parent, total_items: int, items_per_page: int = 1):
        super(Pagination, self).__init__(parent)
        self.setupUi(self)

        self.page_buttons = [self.page1Btn, self.page2Btn, self.page3Btn,
                             self.page4Btn, self.page5Btn]
        self.__num_of_page_btn = len(self.page_buttons)

        # set the parent properly
        ParentWidget.set_parent(parent, self)

        self.__items_per_page = items_per_page
        self.__current_page = 1
        self.__total_items = total_items
        self.__number_of_pages = self.number_of_pages

        self.firstPageBtn.clicked.connect(lambda: self.__onPage_changed(1))
        self.lastPageBtn.clicked.connect(lambda: self.__onPage_changed(self.number_of_pages))
        self.previousPageBtn.clicked.connect(lambda:
                                             self.__onPage_changed(max(self.__current_page - 1, 1)))
        self.nextPageBtn.clicked.connect(lambda:
                                         self.__onPage_changed((min(self.__current_page + 1,
                                                                    self.number_of_pages))))

        self.page_pick_bind = BindPyqtObject(self.itemsPerPagePicker,
                                             self.__onChange_items_per_page)

        self.__update_buttons()

        for btn in self.page_buttons:
            self.__bind_page_btn_click(btn)

        self.__onPageChange_callback = None
        self.__onItemPerPageChange_callback = None

    @property
    def number_of_pages(self):
        return math.ceil(self.__total_items / self.__items_per_page)

    @property
    def items_per_page(self):
        return self.__items_per_page

    @property
    def current_page(self):
        return self.__current_page

    @property
    def __current_page_roll(self):
        return math.floor((self.__current_page - 1) / self.__num_of_page_btn)

    def __bind_page_btn_click(self, btn: pw.QPushButton):
        btn.clicked.connect(lambda: self.__onClick_page_button(btn))

    def set_total_items(self, total_items: int):
        self.__total_items = total_items
        self.__number_of_pages = self.number_of_pages
        self.__update_buttons()

    def __update_buttons(self):
        self.__deselect_buttons()
        index = self.__current_page % self.__num_of_page_btn - 1
        self.page_buttons[index].setFlat(True)
        self.__update_buttons_text()

    def __update_buttons_text(self):
        start_at = self.__current_page_roll * self.__num_of_page_btn
        i = start_at + 1
        for btn in self.page_buttons:
            text = str(i)
            btn.setText(text)
            if i > self.number_of_pages:
                btn.setDisabled(True)
            else:
                btn.setDisabled(False)
            i += 1

    def __deselect_buttons(self):
        for btn in self.page_buttons:
            btn.setFlat(False)

    def __onClick_page_button(self, btn):
        page = int(btn.text())
        self.__onPage_changed(page)

    def __onPage_changed(self, page):
        if self.__current_page != page:
            self.__current_page = page
            if self.__onPageChange_callback:
                self.__onPageChange_callback(page)
        self.__update_buttons()

    def __onChange_items_per_page(self, value):
        value = int(value)
        if self.__items_per_page != value:
            self.__items_per_page = value
            self.__number_of_pages = self.number_of_pages
            self.__onPage_changed(1)
            if self.__onItemPerPageChange_callback:
                self.__onItemPerPageChange_callback(value)

    def bind_onPage_changed(self, func):
        if not isinstance(func, FunctionType):
            pass
            # raise AttributeError("The parameter func must be a function")

        self.__onPageChange_callback = lambda v: func(v)

    def bind_onItemPerPageChange_callback(self, func):
        if not isinstance(func, FunctionType):
            pass
            # raise AttributeError("The parameter func must be a function")
        self.__onItemPerPageChange_callback = lambda v: func(v)


class MessageDialog(pw.QMessageBox):

    def __init__(self, parent):
        super(MessageDialog, self).__init__(parent)
        self.setParent(parent)

        self.setWindowTitle("Message")
        # style
        self.setStyleSheet("QLabel#qt_msgbox_informativelabel {min-width:300px; font-size: 16px;}"
                           "QLabel#qt_msgbox_label {min-width:300px; font: bold 18px;}"
                           "QPushButton{ background-color: rgb(85, 87, 83); border-style: outset; font: 12px;"
                           "border-width: 1px; border-radius: 10px; border-color:rgb(211, 215, 207); "
                           "padding: 2px; color:white}"
                           "QPushButton:hover:pressed "
                           "{ background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, "
                           "stop: 0 #dadbde, stop: 1 #f6f7fa);} "
                           "QPushButton:hover{background-color:rgb(211, 215, 207); color: black;}")

        # self.setStandardButtons(pw.QMessageBox.Ok | pw.QMessageBox.Close) # Example with two buttons
        # self.setStandardButtons(pw.QMessageBox.NoButton) # Example with no buttons
        self.setStandardButtons(pw.QMessageBox.Close)

        self.accepted.connect(self.on_accepted)
        self.rejected.connect(self.on_reject)
        self.finished.connect(self.on_finished)

        self.show()

    def on_accepted(self):
        pass

    def on_reject(self):
        pass

    def on_finished(self):
        pass

    def __set_message(self, message: str, additional_msg: str, msg_type):
        self.setIcon(msg_type)
        self.setText(message)
        self.setInformativeText(additional_msg)

    def set_info_message(self, message: str):
        """
        Set an info message to the message dialog.

        :param message: The message to be display.

        :return:
        """
        self.__set_message("Info", message, pw.QMessageBox.Information)

    def set_warning_message(self, message: str):
        """
        Set a warning message to the message dialog.

        :param message: The message to be display.

        :return:
        """
        self.__set_message("Warning", message, pw.QMessageBox.Warning)

    def set_error_message(self, message):
        """
        Set an error message to the message dialog.

        :param message: The message to be display.

        :return:
        """
        self.__set_message("Error", message, pw.QMessageBox.Critical)


@add_save_load()
class FilterBox(pw.QDockWidget, UiFilterGroupBox):

    def __init__(self, parent: pw.QWidget, current_index=-1):
        super(FilterBox, self).__init__(parent)
        self.setupUi(self)
        self.__parent_name = parent.objectName()  # used to save values in a group.

        # set the parent properly
        ParentWidget.set_parent(parent, self, current_index)
        # force parent to have the same maximumSize and minimumSize of FileBox
        parent.setMaximumSize(self.maximumSize())
        parent.setMinimumSize(self.minimumSize())

        # Clean and add options to combo box.
        self.fiterComboBox.clear()
        self.fiterComboBox.addItems(Filters.get_filters())

        # bind widgets from filter box.
        self.__filter_pick_bind = BindPyqtObject(self.fiterComboBox)
        self.__poles_bind = BindPyqtObject(self.nPolesSb)
        self.__min_freq_bind = BindPyqtObject(self.lowFreqDsb)
        self.__max_freq_bind = BindPyqtObject(self.upperFreqDsb)

    @property
    def parent_name(self):
        return self.__parent_name

    @property
    def min_freq(self) -> float:
        return self.__min_freq_bind.value

    @property
    def max_freq(self) -> float:
        return self.__max_freq_bind.value

    @property
    def filter_value(self) -> str:
        return self.__filter_pick_bind.value

    @property
    def poles(self) -> int:
        return self.__poles_bind.value

    @property
    def is_zero_phase(self) -> bool:
        return self.zeroPhaseCheckBox.isChecked()


@add_save_load()
class EventInfoBox(pw.QGroupBox, UiEventInfoGroupBox):

    def __init__(self, parent: pw.QWidget, canvas):
        super(EventInfoBox, self).__init__(parent)
        self.setupUi(self)
        self.__parent_name = parent.objectName()  # used to save values in a group.
        self.__canvas = None
        self.__on_click_plot_arrivals_callback = None
        self.__arrivals_lines = []

        self.set_canvas(canvas)

        # set the parent properly
        ParentWidget.set_parent(parent, self)
        # force parent to have the same maximumSize and minimumSize of FileBox
        parent.setMaximumSize(self.maximumSize())
        parent.setMinimumSize(self.minimumSize())

        # bind widgets from EventInfo box.
        self.__latitude_bind = BindPyqtObject(self.latitudeLineEdit)
        self.__longitude_bind = BindPyqtObject(self.longitudeLineEdit)
        self.__depth_bind = BindPyqtObject(self.depthLineEdit)

        # button bind
        self.plotArrivalsBtn.clicked.connect(self.__on_click_plot_arrivals)
        self.clearArrivalsBtn.clicked.connect(self.__on_click_clear_arrivals)

    @property
    def parent_name(self):
        return self.__parent_name

    @property
    def latitude(self) -> float:
        try:
            latitude = float(self.__latitude_bind.value)
        except ValueError:
            latitude = None
            self.message("Latitude must be provided.")

        return latitude

    @property
    def longitude(self) -> float:
        try:
            longitude = float(self.__longitude_bind.value)
        except ValueError:
            self.message("Longitude must be provided.")
            longitude = None

        return longitude

    @property
    def event_depth(self) -> float:
        try:
            depth = float(self.__depth_bind.value)
        except ValueError:
            self.message("Depth must be provided.")
            depth = None

        return depth

    @property
    def event_time(self):
        py_time = self.originDateTimeEdit.dateTime().toPyDateTime()
        utc_time = UTCDateTime(py_time)
        return utc_time

    def set_canvas(self, canvas):
        from isp.Gui.Frames import MatplotlibCanvas
        if isinstance(canvas, MatplotlibCanvas):
            self.__canvas = canvas
        else:
            raise AttributeError("canvas is not an instance of MatplotlibCanvas")

    def register_plot_arrivals_click(self, func):
        self.__on_click_plot_arrivals_callback = lambda time, lat, long, depth: func(time, lat, long, depth)

    def message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def set_time(self, time):
        """
        Set the event time.

        :param time: A str or obspy.UTCDateTime.

        :return:
        """
        if type(time) is str:
            time = UTCDateTime(time)
        elif isinstance(time, UTCDateTime):
            time = time.datetime
        elif not isinstance(time, datetime):
            raise ValueError("time must by either str, UTCDatetime or datetime")

        self.originDateTimeEdit.setDateTime(time)

    def add_arrivals_line(self, line: Line2D):
        self.__arrivals_lines.append(line)

    def clear_arrivals(self):
        self.__canvas.remove_arrows(self.__arrivals_lines)
        self.__arrivals_lines = []

    def __on_click_plot_arrivals(self):
        if self.__on_click_plot_arrivals_callback:
            self.__on_click_plot_arrivals_callback(self.event_time, self.latitude, self.longitude, self.event_depth)

    def __on_click_clear_arrivals(self):
        self.clear_arrivals()

    def plot_arrivals(self, axe_index, start_time: UTCDateTime, station_stats: StationsStats):
        delta_time = self.event_time - start_time
        line = self.__canvas.draw_arrow(delta_time, axe_index, "Event time", color="red", linestyles='--',
                                        picker=False)

        sma = SeismogramAnalysis(station_stats.Lat, station_stats.Lon)
        phases, times = sma.get_phases_and_arrivals(self.latitude, self.longitude, self.event_depth)
        self.add_arrivals_line(line)
        for phase, time in zip(phases, times):
            line = self.__canvas.draw_arrow(time + delta_time, axe_index, phase, color="green", linestyles='--',
                                            picker=False)
            self.add_arrivals_line(line)
