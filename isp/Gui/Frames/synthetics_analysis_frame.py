import os
import matplotlib.dates as mdt
from obspy import Stream, read

from isp.DataProcessing import SeismogramDataAdvanced, SeismogramData
from isp.Exceptions import InvalidFile
from isp.Gui import pw, pqg
from isp.Gui.Frames import UiSyntheticsAnalisysFrame, MatplotlibCanvas, CartopyCanvas, FocCanvas
from isp.Gui.Frames.qt_components import ParentWidget, FilesView, MessageDialog
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.synthetics_generator_dialog import SyntheticsGeneratorDialog
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject, convert_qdatetime_utcdatetime

import pickle

@add_save_load()
class SyntheticsAnalisysFrame(pw.QMainWindow, UiSyntheticsAnalisysFrame):

    def __init__(self, parent: pw.QWidget = None):

        super(SyntheticsAnalisysFrame, self).__init__(parent)

        self.setupUi(self)
        ParentWidget.set_parent(parent, self)
        self.setWindowTitle('Synthetics Analysis Frame')
        self.setWindowIcon(pqg.QIcon(':\icons\pen-icon.png'))

        #Initialize parametrs for plot rotation
        self._z = {}
        self._r = {}
        self._t = {}
        self._st = {}
        self.inventory = {}
        parameters = {}

        self._generator = SyntheticsGeneratorDialog(self)

        # 3C_Component
        self.focmec_canvas = FocCanvas(self.widget_fp)
        self.canvas = MatplotlibCanvas(self.plotMatWidget_3C)
        self.canvas.set_new_subplot(3, ncols=1)

        # Map
        self.cartopy_canvas = CartopyCanvas(self.map_widget)

        # binds
        self.root_path_bind_3C = BindPyqtObject(self.rootPathForm_3C, self.onChange_root_path_3C)
        self.vertical_form_bind = BindPyqtObject(self.verticalQLineEdit)
        self.north_form_bind = BindPyqtObject(self.northQLineEdit)
        self.east_form_bind = BindPyqtObject(self.eastQLineEdit)
        self.generation_params_bind = BindPyqtObject(self.paramsPathLineEdit)

        # accept drops
        self.vertical_form_bind.accept_dragFile(drop_event_callback=self.drop_event)
        self.north_form_bind.accept_dragFile(drop_event_callback=self.drop_event)
        self.east_form_bind.accept_dragFile(drop_event_callback=self.drop_event)
        self.generation_params_bind.accept_dragFile(drop_event_callback=self.drop_event)
        self.paramsPathLineEdit.textChanged.connect(self._generationParamsChanged)

        # Add file selector to the widget
        self.file_selector = FilesView(self.root_path_bind_3C.value, parent=self.fileSelectorWidget)
        self.file_selector.setDragEnabled(True)

        self.selectDirBtn_3C.clicked.connect(self.on_click_select_directory_3C)
        self.plotBtn.clicked.connect(self.on_click_rotate)
        ###
        self.stationsBtn.clicked.connect(self.stationsInfo)
        ###

        self.actionGenerate_synthetics.triggered.connect(lambda : self._generator.show())

    def info_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def _generationParamsChanged(self):
        with open(self.generation_params_file, 'rb') as f:
            self.params = pickle.load(f)
            depth_est =self.params['sourcedepthinmeters']
            depth_est =(float(depth_est))/1000
           #self.paramsTextEdit.setPlainText(str(params))
            self.paramsTextEdit.setPlainText("Earth Model: {model}".format(model=self.params['model']))
            self.paramsTextEdit.appendPlainText("Event Coords: {lat} {lon} {depth}".format(
                lat=self.params['sourcelatitude'],lon=self.params['sourcelongitude'],
                              depth=str(depth_est)))
            self.paramsTextEdit.appendPlainText("Time: {time}".format(time=self.params['origintime']))
            self.paramsTextEdit.appendPlainText("Units: {units}".format(units=self.params['units']))

            if 'sourcedoublecouple' in self.params:
                self.paramsTextEdit.appendPlainText("Source: {source}".format(source= self.params['sourcedoublecouple']))
            if 'sourcemomenttensor' in self.params:
                self.paramsTextEdit.appendPlainText("Source: {source}".format(source= self.params['sourcemomenttensor']))


    @staticmethod
    def drop_event(event: pqg.QDropEvent, bind_object: BindPyqtObject):
        data = event.mimeData()
        url = data.urls()[0]
        bind_object.value = url.fileName()

    @property
    def north_component_file(self):
        return os.path.join(self.root_path_bind_3C.value, self.north_form_bind.value)

    @property
    def vertical_component_file(self):
        return os.path.join(self.root_path_bind_3C.value, self.vertical_form_bind.value)

    @property
    def east_component_file(self):
        return os.path.join(self.root_path_bind_3C.value, self.east_form_bind.value)

    @property
    def generation_params_file(self):
        return os.path.join(self.root_path_bind_3C.value, self.generation_params_bind.value)

    def onChange_root_path_3C(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        self.file_selector.set_new_rootPath(value)

    # Function added for 3C Components
    def on_click_select_directory_3C(self):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', self.root_path_bind_3C.value)

        if dir_path:
            self.root_path_bind_3C.value = dir_path

    def on_click_rotate(self, canvas):
        time1 = convert_qdatetime_utcdatetime(self.dateTimeEdit_4)
        time2 = convert_qdatetime_utcdatetime(self.dateTimeEdit_5)

        try:
            sd = SeismogramData(self.vertical_component_file)
            z = sd.get_waveform()
            sd = SeismogramData(self.north_component_file)
            n = sd.get_waveform()
            sd = SeismogramData(self.east_component_file)
            e = sd.get_waveform()
            seismograms = [z, n, e]
            time = z.times("matplotlib")
            self._st = Stream(traces=seismograms)
            for index, data in enumerate(seismograms):
                self.canvas.plot(time, data, index, color="black", linewidth=0.5)
                info = "{}.{}.{}".format(self._st[index].stats.network, self._st[index].stats.station,
                                         self._st[index].stats.channel)
                ax = self.canvas.get_axe(0)
                ax.set_xlim(time1.matplotlib_date, time2.matplotlib_date)
                formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
                ax.xaxis.set_major_formatter(formatter)
                self.canvas.set_plot_label(index, info)

            self.canvas.set_xlabel(2, "Time (s)")

            if 'sourcedoublecouple' in self.params:
                self.focmec_canvas.drawSynthFocMec(0, first_polarity = self.params['sourcedoublecouple'], mti = [])
            if 'sourcemomenttensor' in self.params:
                self.focmec_canvas.drawSynthFocMec(0, first_polarity= [], mti=self.params['sourcemomenttensor'])

        except InvalidFile:
            self.info_message("Invalid mseed files. Please, make sure you have generated correctly the synthetics")

        self.__map_coords()


    def stationsInfo(self):
        files = []
        try:
            if self.vertical_component_file and self.north_component_file and self.east_component_file:
                files = [self.vertical_component_file, self.north_component_file, self.east_component_file]
        except:
            pass

        sd = []
        if len(files)==3:
            for file in files:
                try:
                    st = SeismogramDataAdvanced(file)

                    station = [st.stats.Network,st.stats.Station,st.stats.Location,st.stats.Channel,st.stats.StartTime,
                           st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]

                    sd.append(station)
                except:
                    pass

            self._stations_info = StationsInfo(sd)
            self._stations_info.show()

    def __map_coords(self):
        map_dict = {}
        sd = []
        with open(self.generation_params_file, 'rb') as f:
            params = pickle.load(f)

        n = len(params["bulk"])
        for j in range(n):

            for key in params["bulk"][j]:
                if key == "latitude":
                   lat = params["bulk"][j][key]

                elif key == "longitude":
                    lon = params["bulk"][j][key]

                #elif key == "networkcode":
                #    net = params["bulk"][j][key]

                elif key == "stationcode":
                    sta = params["bulk"][j][key]

                    sd.append(sta)
                    map_dict[sta] = [lon, lat]


        self.cartopy_canvas.plot_map(params['sourcelongitude'], params['sourcelatitude'], 0, 0, 0, 0,
                                     resolution='low', stations=map_dict)




