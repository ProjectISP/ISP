import os
import matplotlib.dates as mdt
from obspy.core.event import Origin
from isp import ROOT_DIR
from isp.DataProcessing import SeismogramDataAdvanced
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import InvalidFile, parse_excepts
from isp.Gui import pw, pqg
from isp.Gui.Frames import UiSyntheticsAnalisysFrame, MatplotlibCanvas, CartopyCanvas, FocCanvas
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.plot_polarization import PlotPolarization
from isp.Gui.Frames.qt_components import ParentWidget, FilterBox, FilesView, MessageDialog
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.synthetics_generator_dialog import SyntheticsGeneratorDialog
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject, convert_qdatetime_utcdatetime
from isp.earthquakeAnalisysis import NllManager, PolarizationAnalyis, PickerManager, FirstPolarity


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
        #self.filter_3ca = FilterBox(self.toolQFrame, 1)  # add filter box component.
        self.parameters = ParametersSettings()
        self._generator = SyntheticsGeneratorDialog(self)

        # 3C_Component
        self.canvas = MatplotlibCanvas(self.plotMatWidget_3C)
        self.canvas.set_new_subplot(3, ncols=1)
        self.canvas_pol = MatplotlibCanvas(self.Widget_polarization)

        # binds
        self.root_path_bind_3C = BindPyqtObject(self.rootPathForm_3C, self.onChange_root_path_3C)
        self.degreeSB_bind = BindPyqtObject(self.degreeSB)
        self.vertical_form_bind = BindPyqtObject(self.verticalQLineEdit)
        self.north_form_bind = BindPyqtObject(self.northQLineEdit)
        self.east_form_bind = BindPyqtObject(self.eastQLineEdit)

        # accept drops
        self.vertical_form_bind.accept_dragFile(drop_event_callback=self.drop_event)
        self.north_form_bind.accept_dragFile(drop_event_callback=self.drop_event)
        self.east_form_bind.accept_dragFile(drop_event_callback=self.drop_event)

        # Add file selector to the widget
        self.file_selector = FilesView(self.root_path_bind_3C.value, parent=self.fileSelectorWidget)
        self.file_selector.setDragEnabled(True)

        self.selectDirBtn_3C.clicked.connect(self.on_click_select_directory_3C)
        self.rotateplotBtn.clicked.connect(lambda: self.on_click_rotate(self.canvas))
        self.rot_macroBtn.clicked.connect(lambda: self.open_parameters_settings())
        self.polarizationBtn.clicked.connect(self.on_click_polarization)
        ###
        self.plotpolBtn.clicked.connect(self.plot_particle_motion)
        self.stationsBtn.clicked.connect(self.stationsInfo)
        self.save_rotatedBtn.clicked.connect(self.save_rotated)
        ###

        self.actionGenerate_synthetics.triggered.connect(lambda : self._generator.show())

    def open_parameters_settings(self):
        self.parameters.show()

    def info_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

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
        angle = self.degreeSB.value()
        incidence_angle= self.incidenceSB.value()
        method = self.methodCB.currentText()
        parameters = self.parameters.getParameters()
        try:
            sd = PolarizationAnalyis(self.vertical_component_file, self.north_component_file, self.east_component_file)
            time, z, r, t, st = sd.rotate(self.inventory,time1, time2, angle, incidence_angle, method = method, parameters = parameters,
                                          trim = True)
            self._z = z
            self._r = r
            self._t = t
            self._st = st
            rotated_seismograms = [z, r, t]
            for index, data in enumerate(rotated_seismograms):
                self.canvas.plot(time, data, index, color="black", linewidth=0.5)
                info = "{}.{}.{}".format(self._st[index].stats.network, self._st[index].stats.station,
                                         self._st[index].stats.channel)
                ax = self.canvas.get_axe(0)
                ax.set_xlim(time1.matplotlib_date, time2.matplotlib_date)
                formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
                ax.xaxis.set_major_formatter(formatter)
                self.canvas.set_plot_label(index, info)

            canvas.set_xlabel(2, "Time (s)")

        except InvalidFile:
            self.info_message("Invalid mseed files. Please, make sure to select all the three components (Z, N, E) "
                         "for rotate.")
        except ValueError as error:
            self.info_message(str(error))


    def on_click_polarization(self):
        time1 = convert_qdatetime_utcdatetime(self.dateTimeEdit_4)
        time2 = convert_qdatetime_utcdatetime(self.dateTimeEdit_5)
        sd = PolarizationAnalyis(self.vertical_component_file, self.north_component_file,
                                 self.east_component_file)
        try:
            var = sd.polarize(time1, time2, self.doubleSpinBox_winlen.value(),
                              self.freq_minDB.value(), self.freq_maxDB.value())

            artist = self.canvas_pol.plot(var['time'], var[self.comboBox_yaxis.currentText()], 0, clear_plot=True,
                                          linewidth=0.5)
            self.canvas_pol.set_xlabel(0, "Time [s]")
            self.canvas_pol.set_ylabel(0, self.comboBox_yaxis.currentText())
            self.canvas_pol.set_yaxis_color(self.canvas_pol.get_axe(0), artist.get_color(), is_left=True)
            self.canvas_pol.plot(var['time'], var[self.comboBox_polarity.currentText()], 0, is_twinx=True, color="red",
                                 linewidth=0.5)
            self.canvas_pol.set_ylabel_twinx(0, self.comboBox_polarity.currentText())
        except InvalidFile:
            self.info_message("Invalid mseed files. Please, make sure to select all the three components (Z, N, E) "
                              "for polarization.")
        except ValueError as error:
            self.info_message(str(error))

    def plot_particle_motion(self):
        self._plot_polarization = PlotPolarization(self._z, self._r, self._t)
        self._plot_polarization.show()

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

    def save_rotated(self):
        import os
        root_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        if self._st:
            n = len(self._st)
            for j in range(n):
                tr = self._st[j]
                t1 = tr.stats.starttime
                id = tr.id+"."+"D"+"."+str(t1.year)+"."+str(t1.julday)
                print(tr.id, "Writing data processed")
                path_output = os.path.join(dir_path, id)
                tr.write(path_output, format="MSEED")


