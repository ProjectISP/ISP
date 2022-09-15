import os
import pickle
from concurrent.futures import ThreadPoolExecutor
import matplotlib.dates as mdt
from obspy import Stream, read
from isp import ROOT_DIR
from isp.DataProcessing import SeismogramDataAdvanced
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import parse_excepts
from isp.Gui import pqg, pw, pyc, qt
from isp.Gui.Frames import Pagination, MatplotlibCanvas, MessageDialog
from isp.Gui.Frames.uis_frames import UiEGFFrame
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from isp.Utils import AsycTime, MseedUtil, ObspyUtil
from isp.ant.ambientnoise import noise_organize
from isp.ant.process_ant import process_ant
from isp.ant.crossstack import noisestack
from sys import platform
from isp.Gui.Utils.pyqt_utils import add_save_load
from isp.earthquakeAnalisysis.stations_map import StationsMap


@add_save_load()
class EGFFrame(pw.QWidget, UiEGFFrame):

    def __init__(self, parameters, settings):
        super(EGFFrame, self).__init__()
        self.setupUi(self)

        self.parameters = parameters
        self.settings_dialog = settings

        self.progressbar = pw.QProgressDialog(self)
        self.progressbar.setWindowTitle('Ambient Noise Tomography')
        self.progressbar.setLabelText(" Computing ")
        self.progressbar.setWindowIcon(pqg.QIcon(':\icons\map-icon.png'))
        self.progressbar.close()

        self.inventory = {}
        self.files = []
        self.total_items = 0
        self.items_per_page = 1
        self.__dataless_manager = None
        self.__metadata_manager = None
        self.st = None
        self.output = None
        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.root_path_bind2 = BindPyqtObject(self.rootPathForm2, self.onChange_root_path)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.updateBtn.clicked.connect(self.plot_egfs)
        self.output_bind = BindPyqtObject(self.outPathForm, self.onChange_root_path)
        self.pagination = Pagination(self.pagination_widget, self.total_items, self.items_per_page)
        self.pagination.set_total_items(0)

        self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=self.items_per_page, constrained_layout=False)
        self.canvas.set_xlabel(0, "Time (s)")
        self.canvas.figure.tight_layout()

        # Bind buttons

        self.readFilesBtn.clicked.connect(lambda: self.get_now_files())
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.selectDirBtn2.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind2))
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_file(self.metadata_path_bind))
        self.outputBtn.clicked.connect(lambda: self.on_click_select_directory(self.output_bind))
        self.preprocessBtn.clicked.connect(self.run_preprocess)
        self.cross_stackBtn.clicked.connect(self.stack)
        self.mapBtn.clicked.connect(self.map)
        self.settingsBtn.clicked.connect(self.settings_dialog.show)
        self.createProjectBtn.clicked.connect(self.create_project)
        self.loadProjectBtn.clicked.connect(self.load_project)
        self.saveProjectBtn.clicked.connect(self.save_project)
        self.searchSyncFileBtn.clicked.connect(self.load_file_sync)
        self.plot_dailyBtn.clicked.connect(self.plot_daily)


    @pyc.Slot()
    def _increase_progress(self):
        self.progressbar.setValue(self.progressbar.value() + 1)

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def onChange_root_path(self, value):

        """
        Fired every time the root_path is changed
        :param value: The path of the new directory.
        :return:
        """
        # self.read_files(value)
        pass

    def on_click_select_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]

    @parse_excepts(lambda self, msg: self.subprocess_feedback(msg))
    @AsycTime.run_async()
    def onChange_metadata_path(self, value):

        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
        except:
            raise FileNotFoundError("The metadata is not valid")

    def subprocess_feedback(self, err_msg: str, set_default_complete=True):
        """
        This method is used as a subprocess feedback. It runs when a raise expect is detected.
        :param err_msg: The error message from the except.
        :param set_default_complete: If True it will set a completed successfully message. Otherwise nothing will
            be displayed.
        :return:
        """
        if err_msg:
            md = MessageDialog(self)
            if "Error code" in err_msg:
                md.set_error_message("Click in show details detail for more info.", err_msg)
            else:
                md.set_warning_message("Click in show details for more info.", err_msg)
        else:
            if set_default_complete:
                md = MessageDialog(self)
                md.set_info_message("Loaded Metadata, please check your terminal for further details")

    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path

    def read_files(self, dir_path):
        md = MessageDialog(self)
        md.hide()
        try:
            self.progressbar.reset()
            self.progressbar.setLabelText(" Reading Files ")
            self.progressbar.setRange(0, 0)
            with ThreadPoolExecutor(1) as executor:
                self.ant = noise_organize(dir_path, self.inventory)
                self.ant.send_message.connect(self.receive_messages)

                def read_files_callback():

                    data_map, size, channels = self.ant.create_dict(net_list=self.params["nets_list"],
                                sta_list=self.params["stations_list"], chn_list=self.params["channels_list"])

                    pyc.QMetaObject.invokeMethod(self.progressbar, 'accept')
                    return data_map, size, channels

                f = executor.submit(read_files_callback)
                self.progressbar.exec()
                self.data_map, self.size, self.channels = f.result()
                f.cancel()

            md.set_info_message("Created Project Successfully, Run Pre-Process or Save the project")
        except:
            md.set_error_message("Something went wrong. Please check your data files are correct mseed files")

        md.show()

    def create_project(self):
        self.params = self.settings_dialog.getParameters()
        self.read_files(self.root_path_bind.value)

    def save_project(self):
        project = {"data_map":self.data_map, "size":self.size, "channels":self.channels}
        try:
            if "darwin" == platform:
                path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind.value)
            else:
                path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
            if not path:
                return

            file_to_store = open(os.path.join(path, "Project_EGFs"), "wb")
            pickle.dump(project, file_to_store)

            md = MessageDialog(self)
            md.set_info_message("Project saved successfully")

        except:

            md = MessageDialog(self)
            md.set_info_message("No data to save in Project")

    def load_project(self):

        selected = pw.QFileDialog.getOpenFileName(self, "Select Project", ROOT_DIR)

        md = MessageDialog(self)

        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            try:
                self.current_project_file = selected[0]
                project = MseedUtil.load_project(file = selected[0])
                self.data_map = project["data_map"]
                self.size = project["size"]
                self.channels = project["channels"]
                project_name = os.path.basename(selected[0])
                md.set_info_message("Project {} loaded  ".format(project_name))
            except:
                md.set_error_message("Project couldn't be loaded ")
        else:
            md.set_error_message("Project couldn't be loaded ")

    def run_preprocess(self):
        self.params = self.settings_dialog.getParameters()
        self.process()

    ####################################################################################################################

    def process(self):
        #
        self.process_ant = process_ant(self.output_bind.value, self.params, self.inventory)
        list_raw = self.process_ant.get_all_values(self.data_map)
        self.process_ant.create_all_dict_matrix(list_raw, self.channels)

    def stack(self):
        self.params = self.settings_dialog.getParameters()
        channels = self.params["channels"]
        stack_method = self.params["stack"]
        power = self.params["power"]
        autocorr = self.params["autocorr"]
        min_distance = self.params["max_distance"]
        dailyStacks = self.params["dailyStacks"]
        stack = noisestack(self.output_bind.value, channels, stack_method, power, autocorr=autocorr,
                           min_distance=min_distance, dailyStacks=dailyStacks)
        stack.run_cross_stack()
        stack.rotate_horizontals()

    @pyc.pyqtSlot(str)
    def receive_messages(self, message):
        self.listWidget.addItem(message)

    def get_files_at_page(self):
        n_0 = (self.pagination.current_page - 1) * self.pagination.items_per_page
        n_f = n_0 + self.pagination.items_per_page
        return self.files[n_0:n_f]

    def get_file_at_index(self, index):
        files_at_page = self.get_files_at_page()
        return files_at_page[index]

    def onChange_items_per_page(self, items_per_page):
        self.items_per_page = items_per_page

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def set_pagination_files(self, files_path):
        self.files = files_path
        self.total_items = len(self.files)
        self.pagination.set_total_items(self.total_items)

    def get_files(self, dir_path):
        files_path = MseedUtil.get_tree_hd5_files(self, dir_path, robust=False)
        self.set_pagination_files(files_path)
        # print(files_path)
        pyc.QMetaObject.invokeMethod(self.progressbar, 'accept', qt.AutoConnection)

        return files_path

    def get_now_files(self):

        md = MessageDialog(self)
        md.hide()
        try:

            self.progressbar.reset()
            self.progressbar.setLabelText(" Reading Files ")
            self.progressbar.setRange(0, 0)
            with ThreadPoolExecutor(1) as executor:
                f = executor.submit(lambda: self.get_files(self.root_path_bind2.value))
                self.progressbar.exec()
                self.files_path = f.result()
                f.cancel()

            # self.files_path = self.get_files(self.root_path_bind.value)

            md.set_info_message("Readed data files Successfully")

        except:

            md.set_error_message("Something went wrong. Please check your data files are correct mseed files")

        md.show()

    def plot_egfs(self):
        if self.st:
            del self.st

        self.canvas.clear()
        ##
        self.nums_clicks = 0
        all_traces = []
        if self.sortCB.isChecked():
            if self.comboBox_sort.currentText() == "Distance":
                self.files_path.sort(key=self.sort_by_distance_advance)

        elif self.comboBox_sort.currentText() == "Back Azimuth":
            self.files_path.sort(key=self.sort_by_baz_advance)

        self.set_pagination_files(self.files_path)
        files_at_page = self.get_files_at_page()
        ##
        if len(self.canvas.axes) != len(files_at_page):
            self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        last_index = 0
        min_starttime = []
        max_endtime = []
        parameters = self.parameters.getParameters()

        for index, file_path in enumerate(files_at_page):
            if os.path.basename(file_path) != ".DS_Store":
                sd = SeismogramDataAdvanced(file_path)

                tr = sd.get_waveform_advanced(parameters, self.inventory,
                                              filter_error_callback=self.filter_error_message, trace_number=index)
                print(tr.data)
                if len(tr) > 0:
                    t = tr.times("matplotlib")
                    s = tr.data
                    self.canvas.plot_date(t, s, index, color="black", fmt='-', linewidth=0.5)
                    if self.pagination.items_per_page >= 16:
                        ax = self.canvas.get_axe(index)
                        ax.spines["top"].set_visible(False)
                        ax.spines["bottom"].set_visible(False)
                        ax.tick_params(top=False)
                        ax.tick_params(labeltop=False)
                        if index != (self.pagination.items_per_page - 1):
                            ax.tick_params(bottom=False)

                    last_index = index

                    st_stats = ObspyUtil.get_stats(file_path)

                    if st_stats and self.sortCB.isChecked() == False:
                        info = "{}.{}.{}".format(st_stats.Network, st_stats.Station, st_stats.Channel)
                        self.canvas.set_plot_label(index, info)

                    elif st_stats and self.sortCB.isChecked() and self.comboBox_sort.currentText() == "Distance":

                        dist = self.sort_by_distance_advance(file_path)
                        dist = "{:.1f}".format(dist / 1000.0)
                        info = "{}.{}.{} Distance {} km".format(st_stats.Network, st_stats.Station, st_stats.Channel,
                                                                str(dist))
                        self.canvas.set_plot_label(index, info)

                    elif st_stats and self.sortCB.isChecked() and self.comboBox_sort.currentText() == "Back Azimuth":

                        back = self.sort_by_baz_advance(file_path)
                        back = "{:.1f}".format(back)
                        info = "{}.{}.{} Back Azimuth {}".format(st_stats.Network, st_stats.Station, st_stats.Channel,
                                                                 str(back))
                        self.canvas.set_plot_label(index, info)

                    try:
                        min_starttime.append(min(t))
                        max_endtime.append(max(t))
                    except:
                        print("Empty traces")

                all_traces.append(tr)

        self.st = Stream(traces=all_traces)
        try:
            if min_starttime and max_endtime is not None:
                auto_start = min(min_starttime)
                auto_end = max(max_endtime)
                self.auto_start = auto_start
                self.auto_end = auto_end

            ax = self.canvas.get_axe(last_index)
            ax.set_xlim(mdt.num2date(auto_start), mdt.num2date(auto_end))
            formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
            ax.xaxis.set_major_formatter(formatter)
            self.canvas.set_xlabel(last_index, "Date")
        except:
            pass

    def sort_by_distance_advance(self, file):

        geodetic = MseedUtil.get_geodetic(file)

        if geodetic[0] is not None:

            return geodetic[0]
        else:
            return 0.

    def sort_by_baz_advance(self, file):

        geodetic = MseedUtil.get_geodetic(file)

        if geodetic[1] is not None:

            return geodetic[0]
        else:
            return 0.

    def map(self):

        try:

            map_dict = {}
            sd = []

            for tr in self.st:

                station = tr.stats.station.split("_")[0]
                name = tr.stats.network+"."+station
                sd.append(name)
                map_dict[name] = [tr.stats.mseed['coordinates'][0], tr.stats.mseed['coordinates'][1]]

            self.map_stations = StationsMap(map_dict)
            self.map_stations.plot_stations_map(latitude = 0, longitude=0)

        except:
            md = MessageDialog(self)
            md.set_error_message("couldn't plot stations map, please check your metadata and the trace headers")


################## clock sync ######################

    def load_file_sync(self):

        md = MessageDialog(self)
        md.hide()

        try:

            selected_file, _ = pw.QFileDialog.getOpenFileName(self, "Select Project", ROOT_DIR)
            self.clockLE.setText(selected_file)
            md.set_info_message("Loaded Stream to Sync Ready")

        except:

            md.set_error_message("Something went wrong. Please check that your data files are correct stream files")

        md.show()

    def plot_daily(self):
        #colors = ['black','indianred','chocolate','darkorange','olivedrab','lightseagreen',
        #         'royalblue','darkorchid','magenta']
        self.canvas.clear()
        self.canvas.set_new_subplot(nrows=1, ncols=1)
        parameters = self.parameters.getParameters()
        st = read(self.clockLE.text())
        cte = 0

        for tr in st:
            sd = SeismogramDataAdvanced(file_path=None, realtime=True, stream=tr)
            tr = sd.get_waveform_advanced(parameters, self.inventory,
                                          filter_error_callback=self.filter_error_message, trace_number=0)
            if len(tr) > 0:
                t = tr.times("matplotlib")
                tr.detrend(type="simple")
                tr.normalize()
                s = tr.data+cte
                self.canvas.plot_date(t, s, 0, color="black", clear_plot=False, fmt='-', alpha=0.75, linewidth=0.5, label= tr.id)
                cte = cte + 2

        ax = self.canvas.get_axe(0)
        #ax.set_xlim(mdt.num2date(auto_start), mdt.num2date(auto_end))
        formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S')
        ax.xaxis.set_major_formatter(formatter)
        self.canvas.set_xlabel(0, "Date")



