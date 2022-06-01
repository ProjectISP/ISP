import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

from isp.Gui import pw
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiProject
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from sys import platform

from isp.Utils import MseedUtil


class Project(pw.QDialog, UiProject):

    def __init__(self, parent=None):
        super(Project, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.pathFilesBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.openProjectBtn.clicked.connect(lambda: self.openProject())
        self.saveBtn.clicked.connect(lambda: self.saveProject())


    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        pass


    def on_click_select_directory(self, bind: BindPyqtObject):

        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)

        if dir_path:
            bind.value = dir_path

    def openProject(self):
        print(self.root_path_bind.value)
        self.project = MseedUtil.search_files(self.root_path_bind.value)
        print(self.project)

    # def openProject(self):
    #
    #     md = MessageDialog(self)
    #     md.hide()
    #     try:
    #         self.progressbar.reset()
    #         self.progressbar.setLabelText(" Reading Files ")
    #         self.progressbar.setRange(0, 0)
    #         with ThreadPoolExecutor(1) as executor:
    #             f = executor.submit(lambda: MseedUtil.search_files(self.root_path_bind.value))
    #             self.progressbar.exec()
    #             self.project = f.result()
    #             f.cancel()
    #
    #     # self.files_path = self.get_files(self.root_path_bind.value)
    #
    #         md.set_info_message("Created Project")
    #
    #     except:
    #
    #         md.set_error_message("Something went wrong. Please check your data files are correct mseed files")
    #
    #
    #     md.show()



    def saveDB(self):

        try:
            if "darwin" == platform:
                path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind.value)
            else:
                path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
            if not path:
                return

            file_to_store = open(self.nameForm.text(), "wb")
            pickle.dump(self.project, file_to_store)

            md = MessageDialog(self)
            md.set_info_message("DB saved successfully")
        except:
            md = MessageDialog(self)
            md.set_info_message("No data to save in DB")
