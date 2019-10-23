import os

from isp.Gui import pw


class FilesView(pw.QTreeView):

    def __init__(self, root_path, parent=None):
        super().__init__(parent)

        self.validate_dir(root_path)

        self.__file_path = None
        self.__model = pw.QFileSystemModel()
        self.__model.setReadOnly(True)

        self.__model.setRootPath(root_path)
        self.__parent_index = self.__model.index(root_path)

        self.setModel(self.__model)
        self.setRootIndex(self.__parent_index)

        self.setColumnWidth(0, 300)
        self.hideColumn(2)
        self.hideColumn(3)
        self.setAlternatingRowColors(True)

        self.clicked.connect(self.__onClick_file)
        self.__model.directoryLoaded.connect(self.__directoryLoaded)
        self.__model.rootPathChanged.connect(self.__rootPathChanged)

        if parent and isinstance(parent, pw.QWidget):
            if parent.layout() is not None:
                layout = parent.layout()
                for child in parent.findChildren(pw.QTreeView):
                    child.setParent(None)
            else:
                layout = pw.QVBoxLayout(parent)

            layout.addWidget(self)

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
    def validate_dir(dir_path):
        if not os.path.isdir(dir_path):
            raise FileExistsError("The path {} is either not a directory or it doesn't exists.".format(dir_path))

    def set_new_rootPath(self, root_path):
        """
        Change the root directory

        :param root_path: The full path of the directory.

        :return:
        """
        self.validate_dir(root_path)
        self.__model.setRootPath(root_path)