import math
import os

from isp.Gui import pw
from isp.Gui.Frames import UiPaginationWidget
from isp.Gui.Utils.pyqt_utils import BindPyqtObject


class ParentWidget:

    @staticmethod
    def set_parent(parent, obj):
        if parent and (isinstance(parent, pw.QWidget) or isinstance(parent, pw.QFrame)):
            if parent.layout() is not None:
                layout = parent.layout()
                for child in parent.findChildren(pw.QTreeView):
                    child.setParent(None)
            else:
                layout = pw.QVBoxLayout(parent)

            layout.addWidget(obj)


class FilesView(pw.QTreeView):

    def __init__(self, root_path, parent=None):
        super().__init__(parent)

        self.__file_path = None
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
        Gets the full file path

        :return: A string containing the full path for the file
        """
        return self.__file_path

    def __onClick_file(self, index):
        self.__file_path = self.__model.filePath(index)

    def __directoryLoaded(self, path):
        index = self.__model.index(0, 0, self.__parent_index)
        self.__file_path = self.__model.filePath(index)

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

        self.firstPageBtn.clicked.connect(lambda: self.onPage_Change(1))
        self.lastPageBtn.clicked.connect(lambda: self.onPage_Change(self.number_of_pages))
        self.previousPageBtn.clicked.connect(lambda:
                                             self.onPage_Change(max(self.__current_page - 1, 1)))
        self.nextPageBtn.clicked.connect(lambda:
                                         self.onPage_Change((min(self.__current_page + 1,
                                                                 self.number_of_pages))))

        self.page_pick_bind = BindPyqtObject(self.itemsPerPagePicker,
                                             self.onChange_items_per_page)

        self.__highlight_current_button()

        for btn in self.page_buttons:
            self.__bind_page_btn_click(btn)

    @property
    def number_of_pages(self):
        return math.floor(self.__total_items / self.__items_per_page)

    @property
    def __current_page_roll(self):
        return math.floor((self.__current_page - 1) / self.__num_of_page_btn)

    def __bind_page_btn_click(self, btn: pw.QPushButton):
        btn.clicked.connect(lambda: self.onClick_page_button(btn))

    def set_total_items(self, total_items: int):
        self.__total_items = total_items
        self.__number_of_pages = self.number_of_pages

    def __highlight_current_button(self):
        self.__deselect_buttons()
        index = self.__current_page % self.__num_of_page_btn - 1
        self.page_buttons[index].setFlat(True)
        self.__update_buttons_text()

    def __update_buttons_text(self):
        start_at = self.__current_page_roll * self.__num_of_page_btn
        i = start_at + 1
        b = pw.QPushButton()
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

    def onPage_Change(self, page):
        self.__current_page = page
        self.__highlight_current_button()
        print("Page {}".format(page))

    def onChange_items_per_page(self, value):
        self.__items_per_page = int(value)
        self.__number_of_pages = self.number_of_pages
        self.onPage_Change(1)

    def onClick_page_button(self, btn):
        page = int(btn.text())
        self.onPage_Change(page)
