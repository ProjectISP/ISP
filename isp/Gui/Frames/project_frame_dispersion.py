import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

from isp import ROOT_DIR
from isp.Gui import pw, pqg, pyc
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiProject_Dispersion
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from sys import platform

from isp.Utils import MseedUtil


class Project(pw.QDialog, UiProject_Dispersion):

    def __init__(self, parent=None):
        super(Project, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

        self.project_dispersion = None
        self.progressbar = pw.QProgressDialog(self)
        self.progressbar.setWindowTitle('Project')
        self.progressbar.setLabelText("Computing Project ")
        self.progressbar.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.progressbar.close()

        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.pathFilesBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.openProjectBtn.clicked.connect(lambda: self.openProject())
        self.saveBtn.clicked.connect(lambda: self.saveProject())

    @pyc.Slot()
    def _increase_progress(self):
        self.progressbar.setValue(self.progressbar.value() + 1)

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
        # TODO CHECK IF SOME FILES ARE ALREADY IN MEMORY IN THE FILE DIAOLOG SELECTOR
        if "darwin" == platform:
            selected = pw.QFileDialog.getOpenFileName(self, "Select Project", ROOT_DIR)
        else:
            selected = pw.QFileDialog.getOpenFileName(self, "Select Project", ROOT_DIR,
                                                      pw.QFileDialog.DontUseNativeDialog)

        md = MessageDialog(self)

        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            try:
                self.current_project_file = selected[0]
                self.project_dispersion = self.load_project(file = selected[0])
                project_name = os.path.basename(selected[0])
                md.set_info_message("Project {} loaded  ".format(project_name))
            except:
                md.set_error_message("Project couldn't be loaded ")
        else:
            md.set_error_message("Project couldn't be loaded ")


    @staticmethod
    def load_project(file: str):
        project = {}
        try:
            project = pickle.load(open(file, "rb"))
            print(project)
        except:
            pass
        return project


    def saveProject(self):
        new_project = {}
        try:

            file_to_store = open(os.path.join(self.rootPathForm.text(),self.nameForm.text()), "wb")
            pickle.dump(new_project, file_to_store)

            md = MessageDialog(self)
            md.set_info_message("Project saved successfully")

        except:

            md = MessageDialog(self)
            md.set_info_message("No data to save in Project")