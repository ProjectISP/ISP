import os

from obspy.geodetics import gps2dist_azimuth

from isp import ROOT_DIR
from isp.DataProcessing import SeismogramData, DatalessManager
from isp.Gui import pw
from isp.Gui.Frames import BaseFrame, UiEarthquakeAnalysisFrame, Pagination, MessageDialog
from isp.Gui.Frames.matplotlib_frame import MatplotlibCanvas
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from isp.Structures.structures import StationsStats
from isp.Utils import MseedUtil


class EarthquakeAnalysisFrame(BaseFrame, UiEarthquakeAnalysisFrame):

    def __init__(self, ):
        super(EarthquakeAnalysisFrame, self).__init__()
        self.setupUi(self)

        self.files = []
        self.total_items = 0
        self.items_per_page = 1
        self.__dataless_manager = None
        self.dataless_not_found = set() # a set of mseed files that the dataless wasn't found.

        self.pagination = Pagination(self.pagination_widget, self.total_items, self.items_per_page)
        self.pagination.set_total_items(0)
        self.pagination.bind_onPage_changed(self.onChange_page)
        self.pagination.bind_onItemPerPageChange_callback(self.onChange_items_per_page)

        self.canvas = MatplotlibCanvas(self.plotMatWidget, nrows=self.items_per_page)
        self.canvas.set_xlabel(0, "Time (s)")
        self.canvas.on_double_click(self.on_click_matplotlib)

        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_dataless_path)

        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.selectDatalessDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        self.sortBtn.clicked.connect(self.on_click_sort)

    @property
    def dataless_manager(self):
        if not self.__dataless_manager:
            self.__dataless_manager = DatalessManager(self.dataless_path_bind.value)
        return self.__dataless_manager

    def message_dataless_not_found(self):
        if len(self.dataless_not_found) > 1:
            md = MessageDialog(self)
            md.set_info_message("Dataless not found.")
        else:
            for file in self.dataless_not_found:
                md = MessageDialog(self)
                md.set_info_message("Dataless for {} not found.".format(file))

        self.dataless_not_found.clear()

    def get_files_at_page(self):
        n_0 = (self.pagination.current_page - 1) * self.pagination.items_per_page
        n_f = n_0 + self.pagination.items_per_page
        return self.files[n_0:n_f]

    def onChange_page(self, page):
        self.plot_seismogram()

    def onChange_items_per_page(self, items_per_page):
        self.items_per_page = items_per_page
        self.plot_seismogram()

    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        self.files = MseedUtil.get_mseed_files(value)
        self.total_items = len(self.files)
        self.pagination.set_total_items(self.total_items)
        self.plot_seismogram()

    def onChange_dataless_path(self, value):
        self.__dataless_manager = DatalessManager(value)

    def sort_by_distance(self, file):
        st_stats = self.dataless_manager.get_station_stats_by_mseed_file(file)
        if st_stats:
            dist, _, _ = gps2dist_azimuth(st_stats.Lat, st_stats.Lon, 0., 0.)
            # print("File, dist: ", file, dist)
            return dist
        else:
            self.dataless_not_found.add(file)
            print("No dataless found for {} file.".format(file))
            return 0.

    def on_click_sort(self):
        self.files.sort(key=self.sort_by_distance)
        self.message_dataless_not_found()
        self.plot_seismogram()

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)

        if dir_path:
            bind.value = dir_path

    def plot_seismogram(self):
        self.canvas.clear()
        files_at_page = self.get_files_at_page()
        if len(self.canvas.axes) != len(files_at_page):
            self.canvas.set_new_subplot(nrows=len(files_at_page), ncols=1)
        last_index = 0
        for index, file_path in enumerate(files_at_page):
            sd = SeismogramData(file_path)
            t, s = sd.get_waveform()
            self.canvas.plot(t, s, index, color="black")
            last_index = index

        # set x-label at the last axes.
        self.canvas.set_xlabel(last_index, "Time (s)")

    def on_click_matplotlib(self, event, canvas):
        if isinstance(canvas, MatplotlibCanvas):
            phase = "Phase"
            click_at_index = event.inaxes.rowNum
            x1, y1 = event.xdata, event.ydata
            canvas.draw_arrow(x1, click_at_index, phase)


