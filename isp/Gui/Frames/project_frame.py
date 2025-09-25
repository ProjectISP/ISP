import os
import pickle
from surfquakecore.project.surf_project import SurfProject
from isp import ROOT_DIR
from isp.Gui import pw, pqg, pyc
from isp.Gui.Frames import MessageDialog
from isp.Gui.Frames.uis_frames import UiProject
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from sys import platform
from isp.Utils import MseedUtil


class WorkerSignals(pyc.QObject):
    finished = pyc.pyqtSignal()
    error = pyc.pyqtSignal(str)
    result = pyc.pyqtSignal(object)

class WorkerRunnable(pyc.QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit(str(e))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class Project(pw.QDialog, UiProject):

    def __init__(self, parent=None):
        super(Project, self).__init__(parent)
        self.setupUi(self)

        if parent is not None:
            self.setWindowTitle(parent.windowTitle())
            self.setWindowIcon(parent.windowIcon())

        self.progressbar = pw.QProgressDialog(self)
        self.progressbar.setWindowTitle('Project')
        self.progressbar.setLabelText("Computing Project ")
        self.progressbar.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.progressbar.close()

        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.pathFilesBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.openProjectBtn.clicked.connect(lambda: self.openProject())
        self.saveBtn.clicked.connect(lambda: self.saveProject())
        self.regularBtn.clicked.connect(lambda: self.load_files_done())

    @pyc.Slot()
    def _increase_progress(self):
        self.progressbar.setValue(self.progressbar.value() + 1)

    def onChange_root_path(self, value):
        pass

    def load_files_done(self):
        if self.rootPathForm_inv.text() == "":
            filter = "All files (*.*)"
        else:
            filter = self.rootPathForm_inv.text()

        selected_files, _ = pw.QFileDialog.getOpenFileNames(self, "Select Project", ROOT_DIR, filter=filter)

        md = MessageDialog(self)

        ms = MseedUtil()
        self.progressbar.reset()
        self.progressbar.setLabelText("Creating project")
        self.progressbar.setRange(0, 0)
        self.progressbar.show()

        def background_task():
            return ms.search_indiv_files(selected_files)

        def on_result(result):
            self.project = result
            info = ms.get_project_basic_info(self.project)

            if len(info) > 0:
                md.set_info_message("Loaded Project ",
                                    "Networks: " + ','.join(info["Networks"][0]) + "\n" +
                                    "Stations: " + ','.join(info["Stations"][0]) + "\n" +
                                    "Channels: " + ','.join(info["Channels"][0]) + "\n" + "\n" +

                                    "Networks Number: " + str(info["Networks"][1]) + "\n" +
                                    "Stations Number: " + str(info["Stations"][1]) + "\n" +
                                    "Channels Number: " + str(info["Channels"][1]) + "\n" +
                                    "Num Files: " + str(info["num_files"]) + "\n")
            else:
                md.set_warning_message("Empty Project ",
                                       "Please provide a root path with mseed files inside and check the query filters applied")

            md.show()
            self.progressbar.hide()

        def on_error(error_msg):
            md.set_error_message("Something went wrong. Please check your data files are correct mseed files.\n" + error_msg)
            md.show()
            self.progressbar.hide()

        runnable = WorkerRunnable(background_task)
        runnable.signals.result.connect(on_result)
        runnable.signals.error.connect(on_error)
        runnable.signals.finished.connect(self.progressbar.hide)

        pyc.QThreadPool.globalInstance().start(runnable)


    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)

        if dir_path:
            bind.value = dir_path


    def openProject(self):
        info = {}
        md = MessageDialog(self)
        md.hide()
        ms = MseedUtil()

        self.progressbar.reset()
        self.progressbar.setLabelText("Building Project")
        self.progressbar.setRange(0, 0)
        self.progressbar.show()

        def background_task():
            return ms.search_files(self.root_path_bind.value)

        def on_result(result):
            self.project = result
            info = ms.get_project_basic_info(self.project)

            if len(info) > 0:
                md.set_info_message("Opened Project ",
                                    "Networks: " + ','.join(info["Networks"][0]) + "\n" +
                                    "Stations: " + ','.join(info["Stations"][0]) + "\n" +
                                    "Channels: " + ','.join(info["Channels"][0]) + "\n" + "\n" +

                                    "Networks Number: " + str(info["Networks"][1]) + "\n" +
                                    "Stations Number: " + str(info["Stations"][1]) + "\n" +
                                    "Channels Number: " + str(info["Channels"][1]) + "\n" +
                                    "Num Files: " + str(info["num_files"]) + "\n")
            else:
                md.set_warning_message("Empty Project ",
                                       "Please provide a root path with mseed files inside and check the query filters applied")

            md.show()
            self.progressbar.hide()

        def on_error(error_msg):
            md.set_error_message("Something went wrong. Please check that your data files are correct mseed files\n" + error_msg)
            md.show()
            self.progressbar.hide()

        runnable = WorkerRunnable(background_task)
        runnable.signals.result.connect(on_result)
        runnable.signals.error.connect(on_error)
        runnable.signals.finished.connect(self.progressbar.hide)

        pyc.QThreadPool.globalInstance().start(runnable)


    def saveProject(self):

        if hasattr(self, 'project') and len(self.project) > 0:
            try:
                if "darwin" == platform:
                    path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind.value)
                else:
                    path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind.value,
                                                               pw.QFileDialog.DontUseNativeDialog)
                if not path:
                    return

                file_to_store = open(os.path.join(path, self.nameForm.text()), "wb")

                # Now writes surfQuake Project
                sp = SurfProject("")
                sp.project = self.project
                sp.data_files = MseedUtil.generate_data_files(self.project)
                pickle.dump(sp, file_to_store)

                md = MessageDialog(self)
                md.set_info_message("Project saved successfully")
                md.show()

            except Exception as e:
                md = MessageDialog(self)
                md.set_error_message("Failed to save project: " + str(e))
                md.show()
        else:
            md = MessageDialog(self)
            md.set_warning_message("No data to save in Project")
            md.show()