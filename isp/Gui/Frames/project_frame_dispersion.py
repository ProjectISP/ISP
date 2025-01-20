import os
import pickle
from isp import ROOT_DIR
from isp.Gui import pw, pqg, pyc
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiProject_Dispersion
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from sys import platform
import pandas as pd


class Project(pw.QDialog, UiProject_Dispersion):

    signal_proj = pyc.pyqtSignal()
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
                self.project_dispersion = self.load_project(file=selected[0])
                self.send_signal()
                project_name = os.path.basename(selected[0])
                md.set_info_message("Project {} loaded  ".format(project_name))
            except:
                md.set_error_message("Project couldn't be loaded ")
        else:
            md.set_error_message("Project couldn't be loaded ")



    def load_project(self, file: str):
        project = {}

        try:
            project = pickle.load(open(file, "rb"))
            # get info
            if len(project) > 0:
                for key in project.keys():
                     print(key)

        except Exception as e:
             # Print the exception details
             print(f"An error occurred: {e}")
             # Optionally, log the exception or perform additional handling
             # Re-raise the exception
             return project
        return project


    def send_signal(self):

        # Connect end of picking with Earthquake Analysis

        self.signal_proj.emit()

    def saveProject(self):

        new_project = {}
        md = MessageDialog(self)
        try:
            file_path = os.path.join(self.rootPathForm.text(),self.nameForm.text())
            if len(self.rootPathForm.text()) > 0 and len(self.nameForm.text()) > 0:
                file_to_store = open(file_path, "wb")
                pickle.dump(new_project, file_to_store)
                self.project_dispersion = new_project
                self.current_project_file = file_path
                md.set_info_message("Project saved successfully")
            else:
                md.set_info_message("Please set a valid path to open and save the new project")

        except:

            md = MessageDialog(self)
            md.set_info_message("No data to save in Project")


    def save_project2txt(self):

        path_basename = os.path.dirname(self.current_project_file)
        path_txt_files = os.path.join(path_basename, "dispersion_txt")
        if not os.path.exists(path_txt_files):
           os.makedirs(path_txt_files)

        #project = pickle.load(open(self.current_project_file, "rb"))

        # loop over pickle
        for key in self.project_dispersion.keys():
            key_name = key+".txt"
            output_path = os.path.join(path_txt_files, key_name)
            disp_dict = {'period': self.project_dispersion[key]['period'], 'velocity':
                self.project_dispersion[key]['velocity']}
            df_disp = pd.DataFrame.from_dict(disp_dict)
            df_disp.to_csv(output_path, sep=";", index=False)
