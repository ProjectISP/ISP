#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
isola_ISP_frame

"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QWidget, QVBoxLayout
from obspy.core.event import Origin
from isp import ROOT_DIR, ALL_LOCATIONS
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import InvalidFile
from isp.Gui import pw, qt, pyc
from isp.Gui.Frames import BaseFrame, MessageDialog, UiMomentTensor, MatplotlibFrame, FilesView
from isp.Gui.Frames.crustal_model_parameters_frame import CrustalModelParametersFrame
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, add_save_load, convert_qdatetime_utcdatetime, set_qdatetime, \
    convert_qdatetime_datetime
from isp.Utils import MseedUtil, ObspyUtil, AsycTime
from isp.Utils.subprocess_utils import open_html_file, open_url
from isp.mti.mti_utilities import MTIManager
from isp.mti.class_isola_new import *
from obspy import Stream, UTCDateTime, Inventory
import platform

from surfquakecore.moment_tensor.mti_parse import read_isola_result, WriteMTI
from surfquakecore.moment_tensor.sq_isola_tools.sq_bayesian_isola import BayesianIsolaCore
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.moment_tensor.structures import MomentTensorInversionConfig, StationConfig, \
    InversionParameters, SignalProcessingParameters

from isp.mti.sq_bayesian_isola_core import BayesianIsolaGUICore


@add_save_load()
class MTIFrame(BaseFrame, UiMomentTensor):

    def __init__(self):
        super(MTIFrame, self).__init__()
        self.setupUi(self)
        self.inventory = {}
        self._stations_info = {}
        self.stream = None
        self.stations_check = False
        self.url = 'https://projectisp.github.io/ISP_tutorial.github.io/mti/'
        self.scroll_area_widget.setWidgetResizable(True)
        self.earth_model = CrustalModelParametersFrame()
        # Binding
        self.earth_path_bind = BindPyqtObject(self.earth_model_path)
        self.output_path_bind = BindPyqtObject(self.MTI_output_path)
        #self.rootPathFormView_bind = BindPyqtObject(self.rootPathFormView)

        self.earthModelMTIBtn.clicked.connect(lambda: self.on_click_select_file(self.earth_path_bind))
        self.outputDirectoryMTIBtn.clicked.connect(lambda: self.on_click_select_directory(self.output_path_bind))
        #self.selectDirViewBtn.clicked.connect(lambda: self.on_click_select_directory(self.rootPathFormView_bind))
        self.selectDirViewBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        # actions
        self.stationSelectBtn.clicked.connect(lambda: self.stationsInfo())
        self.runInversionMTIBtn.clicked.connect(lambda: self.run_inversion())
        self.printMTIresultsBtn.clicked.connect(lambda: self.print_mti())

        # self.file_selector = FilesView(self.rootPathFormView_bind.value, parent=self.fileSelectorWidget,
        #                                on_change_file_callback=lambda file_path: self.onChange_file(file_path))
        self.root_path_bind = BindPyqtObject(self.rootPathForm, self.onChange_root_path)
        self.file_selector = FilesView(self.root_path_bind.value, parent=self.fileSelectorWidget,
                                       on_change_file_callback=lambda file_path: self.onChange_file(file_path))

        self.plot_solutionBtn.clicked.connect(lambda: self.plot_solution())
        self.openHTML.clicked.connect(lambda: self.load_HTML_file())
        self.readDBBtn.clicked.connect(lambda: self.read_database())
        self.runInversionMTIDBBtn.clicked.connect(lambda: self.run_inversionDB())
        self.loadProjectBtn.clicked.connect(lambda: self.load_project())
        self.loadMetadataBtn.clicked.connect(lambda: self.load_metadata())
        self.earthModelMakerBtn.clicked.connect(lambda: self.open_earth_model())

    def open_earth_model(self):
        self.earth_model.show()
    def load_project(self):


        selected = pw.QFileDialog.getOpenFileName(self, "Select Project", ROOT_DIR)

        md = MessageDialog(self)

        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            try:

                self.current_project_file = selected[0]
                self.sp = SurfProject.load_project(self.current_project_file)
                project_name = os.path.basename(selected[0])
                info = MseedUtil.get_project_basic_info(self.sp.project)
                md.set_info_message("Project {} loaded  ".format(project_name))
                if len(info) > 0:
                    md.set_info_message("Project {} loaded  ".format(project_name),
                                        "Networks: " + ','.join(info["Networks"][0]) + "\n" +
                                        "Stations: " + ','.join(info["Stations"][0]) + "\n" +
                                        "Channels: " + ','.join(info["Channels"][0]) + "\n" + "\n"+

                                        "Networks Number: " + str(info["Networks"][1]) + "\n" +
                                        "Stations Number: " + str(info["Stations"][1]) + "\n" +
                                        "Channels Number: " + str(info["Channels"][1]) + "\n" +
                                        "Num Files: " + str(info["num_files"]) + "\n")

                else:
                    md.set_warning_message("Empty Project ", "Please provide a root path "
                                                             "with mseed files inside and check the wuery filters applied")

            except:
                md.set_error_message("Project couldn't be loaded ")
        else:
            md.set_error_message("Project couldn't be loaded ")

    def load_metadata(self):

        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")

        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            metadata_file = selected[0]
        try:
            self.__metadata_manager = MetadataManager(metadata_file)
            self.inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
        except:
            raise FileNotFoundError("The metadata is not valid")


    def controller(self):
        from isp.Gui.controllers import Controller
        return Controller()

    def read_database(self):
        Controller = self.controller()
        Controller.open_project()
        self.db_frame = Controller.project_frame

    def get_db(self):
        db = self.db_frame.get_entities()
        return db

    def get_model(self):
        # returns the database
        model = self.db_frame.get_model()
        return model


    def plot_solution(self):

        self._load_log_file()
        beach_ball_list = ['centroid.png', 'uncertainty_MT_DC.png', 'uncertainty_MT.png']
        synthetic_list = ['seismo_sharey.png', 'spectra.png']
        self.__plot_grid(beach_ball_list, self.pltGrid, max_cols=1, size=250)
        self.__plot_grid(synthetic_list, self.pltSynthetics, max_cols=0, size=550)

    def __plot_grid(self, good_list, grid, max_cols, size):

        # Clear the grid layout
        for i in reversed(range(grid.count())):
            widget = grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Load and display images with titles
        row, col = 0, 0
        for filename in os.listdir(self.root_path_bind.value):
            if filename.lower().endswith('.png') and filename in good_list:  # Only load PNG images
                image_path = os.path.join(self.root_path_bind.value, filename)
                pixmap = QPixmap(image_path)

                # Resize the image for thumbnails
                pixmap = pixmap.scaled(size, size, qt.KeepAspectRatio, qt.SmoothTransformation)

                # Create a vertical layout for the image and title
                vbox = QVBoxLayout()

                # Add a title (filename without extension)
                title = QLabel(self)
                title.setText(os.path.splitext(filename)[0])  # Remove .png from the filename
                title.setAlignment(qt.AlignCenter)  # Center-align the title
                title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 2px;")  # Optional styling
                vbox.addWidget(title)

                # Add the image
                image_label = ImageLabel(image_path, self)  # Use custom QLabel
                image_label.setPixmap(pixmap)
                image_label.setAlignment(qt.AlignCenter)
                vbox.addWidget(image_label)

                # Add the vertical layout to the grid
                container_widget = QWidget()
                container_widget.setLayout(vbox)
                grid.addWidget(container_widget, row, col)

                col += 1
                if col > max_cols:  # Adjust column count for grid
                    col = 0
                    row += 1
    def load_HTML_file(self):
        html_path = os.path.join(self.root_path_bind.value, 'index.html')
        open_html_file(html_path)

    def _load_log_file(self):
        logfile = os.path.join(self.root_path_bind.value, "log.txt")
        if logfile:
            try:
                # Read the content of the file
                with open(logfile, 'r') as file:
                    log_content = file.read()
                    self.infoTx.setPlainText(log_content)  # Display the content in QPlainTextEdit
            except Exception as e:
                self.infoTx.setPlainText(f"Failed to load file: {e}")


    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        self.file_selector.set_new_rootPath(value)

    def onChange_file(self, file_path):
        # Called every time user select a different file
        pass

    def get_inversion_parameters(self):

        parameters = {'origin_time': convert_qdatetime_datetime(self.origin_time),
                      'latitude': self.latDB.value(), 'longitude': self.lonDB.value(), 'depth': self.depthDB.value(),
                      'output_directory': self.MTI_output_path.text(),
                      'earth_model': self.earth_model_path.text(),
                      'location_unc': self.HorizontalLocUncertainityMTIDB.value(),
                      'magnitude': self.magnitudeDB.value(),
                      'time_unc': self.timeUncertainityMTIDB.value(), 'depth_unc': self.depthUncertainityMTIDB.value(),
                      'deviatoric': self.deviatoricCB.isChecked(), 'covariance': self.covarianceCB.isChecked(),
                      'plot_save': self.savePlotsCB.isChecked(), 'rupture_velocity': self.ruptureVelMTIDB.value(),
                      'source_type': self.sourceTypeCB.currentText(), 'min_dist': self.minDistMTIDB.value(),
                      'max_dist': self.maxDistMTIDB.value(), 'fmin': self.freq_minMTI.value(),
                      'fmax': self.freq_maxMTI.value(), 'rms_thresh': self.rms_threshMTI.value(),
                      'max_num_stationsMTI0': self.max_num_stationsMTI.value(),
                      'source_duration': self.sourceTypeLenthgMTIDB.value(),
                      'max_number_stations': self.max_num_stationsMTI.value()}

        return parameters


    def run_inversionDB(self):
        self.__send_run_mti_db()
        self.progress_dialog.exec()
        md = MessageDialog(self)
        md.set_info_message("Moment Tensor Inversion finished, Please see output directory and press "
                            "print results")

    @AsycTime.run_async()
    def __send_run_mti_db(self):

        parameters = self.get_inversion_parameters()
        bic = BayesianIsolaCore(project=self.sp, inventory_file=self.inventory,
                                output_directory=self.MTI_output_path.text(),
                                save_plots=parameters['plot_save'])

        bi = BayesianIsolaGUICore(bic, model=self.get_model(), entities=self.get_db(),
                                  parameters=parameters)
        bi.run_inversion()
        wm = WriteMTI(self.MTI_output_path.text())
        wm.mti_summary()
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)

    def run_inversion(self):
        self.__send_run_mti()
        self.progress_dialog.exec()
        md = MessageDialog(self)
        md.set_info_message("Moment Tensor Inversion finished, Please see output directory and press "
                            "print results")

    @AsycTime.run_async()
    def __send_run_mti(self):

        parameters = self.get_inversion_parameters()
        stations_map = None
        if isinstance(self._stations_info, StationsInfo):
            stations_map = self._stations_info.get_stations_map()

        mti_config = MomentTensorInversionConfig(
            origin_date=parameters['origin_time'],
            latitude=parameters['latitude'],
            longitude=parameters['longitude'],
            depth_km=parameters['depth'],
            magnitude=parameters['magnitude'],
            stations=[StationConfig(name=".", channels=["."])],

            inversion_parameters=InversionParameters(
                earth_model_file=parameters['earth_model'],
                location_unc=parameters['location_unc'],
                time_unc=parameters['time_unc'],
                depth_unc=parameters['depth_unc'],
                source_duration=parameters['source_duration'],
                rupture_velocity=parameters['rupture_velocity'],
                min_dist=parameters['min_dist'],
                max_dist=parameters['max_dist'],
                source_type=parameters['source_type'],
                deviatoric=parameters['deviatoric'],
                covariance=parameters['covariance'],
                max_number_stations=parameters['max_number_stations']),

            signal_processing_parameters=SignalProcessingParameters(remove_response=False,
                                                                    max_freq=parameters['fmax'],
                                                                    min_freq=parameters['fmin'],
                                                                    rms_thresh=parameters['rms_thresh']))

        bic = BayesianIsolaCore(project=self.st, inventory_file=self.inventory,
                                output_directory=self.output_path_bind.value,
                                save_plots=self.savePlotsCB.isChecked())

        # # Run Inversion
        bic.run_inversion(mti_config=mti_config, map_stations=stations_map)
        pyc.QMetaObject.invokeMethod(self.progress_dialog, 'accept', Qt.QueuedConnection)

    def on_click_select_file(self, bind: BindPyqtObject):
        file_path = pw.QFileDialog.getOpenFileName(self, 'Select Directory', bind.value)
        file_path = file_path[0]

        if file_path:
            bind.value = file_path

    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path

    def on_click_select_hyp_file(self):
        file_path = pw.QFileDialog.getOpenFileName(self, 'Select Directory', ALL_LOCATIONS)
        file_path = file_path[0]
        if isinstance(file_path, str):
            return file_path
        else:
            md = MessageDialog(self)
            md.set_info_message("No selected any file, please set hypocenter parameters manually")
            return None

    def load_event(self):
        hyp_file = self.on_click_select_hyp_file()
        if isinstance(hyp_file, str):
            origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file, modified=True)
            hyp_values = MTIManager.get_hyp_values(origin[0])
            self.__set_hyp(hyp_values)
            md = MessageDialog(self)
            md.set_info_message("Loaded information and set hypocenter parameters, please click stations info")

    def stationsInfo(self):

        self.stations_check = True
        md = MessageDialog(self)
        md.set_info_message("Select your Station/Channel", "Sorted by distance to the epicenter")

        all_stations = []

        # for stream in self.stream:
        for tr in self.st:
            station = [tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel, tr.stats.starttime,
                       tr.stats.endtime, tr.stats.sampling_rate, tr.stats.npts]
            all_stations.append(station)

        self._stations_info = StationsInfo(all_stations, check=True)
        self._stations_info.show()

    def send_mti(self, stream: Stream, inventory: Inventory, starttime: UTCDateTime, endtime: UTCDateTime,
                 option: str):

        self.st = stream
        print(self.st)

        self.inventory = inventory
        self.starttime = starttime
        self.endttime = endtime
        self.stream_frame = MatplotlibFrame(self.st, type='normal')
        self.stream_frame.show()

        # next set the hypocenter parameters
        if option == "manually":
            md = MessageDialog(self)
            md.set_info_message("Loaded information, please set hypocenter parameters by yourself, "
                                "then click stations info")

        elif option == "last":
            hyp_file = os.path.join(ROOT_DIR, "earthquakeAnalisysis", "location_output", "loc", "last.hyp")
            origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file, modified=True)
            hyp_values = MTIManager.get_hyp_values(origin[0])
            self.__set_hyp(hyp_values)
            md = MessageDialog(self)
            md.set_info_message("Loaded information and set hypocenter parameters, please click stations info")

        elif option == "other":
            hyp_file = self.on_click_select_hyp_file()
            if isinstance(hyp_file, str):
                origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file, modified=True)
                hyp_values = MTIManager.get_hyp_values(origin[0])
                self.__set_hyp(hyp_values)
                md = MessageDialog(self)
                md.set_info_message("Loaded information and set hypocenter parameters, please click stations info")

    def print_mti(self):
        self.mti_text.clear()
        iversion_json_files = []

        for foldername, subfolders, filenames in os.walk(self.MTI_output_path.text()):
            for filename in filenames:
                if filename == "inversion.json":
                    iversion_json_files.append(os.path.join(foldername, filename))

        for result_file in iversion_json_files:
            result = read_isola_result(result_file)
            self.mti_text.appendPlainText("#####################################################")
            formatted_date = result["centroid"]["time"]
            lat = str("{: .2f}".format(result["centroid"]["latitude"]))
            lon = str("{: .2f}".format(result["centroid"]["longitude"]))
            depth = str("{: .2f}".format(result["centroid"]["depth"]))
            self.mti_text.appendPlainText(formatted_date + "    " + lat +"º    "+ lon+"º    "+ depth+" km")
            mrr = str("{: .2e}".format(result["centroid"]["mrr"]))
            mtt = str("{: .2e}".format(result["centroid"]["mtt"]))
            mpp = str("{: .2e}".format(result["centroid"]["mpp"]))
            mrt = str("{: .2e}".format(result["centroid"]["mrt"]))
            mrp = str("{: .2e}".format(result["centroid"]["mtt"]))
            mtp = str("{: .2e}".format(result["centroid"]["mpp"]))
            self.mti_text.appendPlainText("mrr    " + mrr + "    mtt    " + mtt + "    mpp    " + mpp)
            self.mti_text.appendPlainText("mrt    " + mrt + "    mrp    " + mrp + "    mtp    " + mtp)
            cn = str("{: .2f}".format(result["centroid"]["cn"]))
            vr = str("{: .2f}".format(result["centroid"]["vr"]))
            self.mti_text.appendPlainText("CN    " + cn + "    VR    "+vr+" "+ "%")
            cvld = str("{: .2f}".format(result["scalar"]["clvd"]))
            dc = str("{: .2f}".format(result["scalar"]["dc"]))
            iso = str("{: .2f}".format(result["scalar"]["isotropic_component"]))
            mo = str("{: .2e}".format(result["scalar"]["mo"]))
            mw = str("{: .2f}".format(result["scalar"]["mw"]))
            rupture_length = str("{: .2f}".format(result["centroid"]["rupture_length"]))
            plane_1_dip = str("{: .1f}".format(result["scalar"]["plane_1_dip"]))
            plane_1_slip_strike = str("{: .1f}".format(result["scalar"]["plane_1_slip_rake"]))
            plane_1_strike = str("{: .1f}".format(result["scalar"]["plane_1_strike"]))
            plane_2_dip = str("{: .1f}".format(result["scalar"]["plane_2_dip"]))
            plane_2_slip_strike = str("{: .1f}".format(result["scalar"]["plane_2_slip_rake"]))
            plane_2_strike = str("{: .1f}".format(result["scalar"]["plane_2_strike"]))
            self.mti_text.appendPlainText("CVLD    " + cvld + " %" + "    DC    "+dc + " %"+"    ISO    "+iso + " %")
            self.mti_text.appendPlainText("Mo    " + mo + " Nm"+"    Mw    " + mw+"    Rupture Length    " +
                                          rupture_length + " km")
            self.mti_text.appendPlainText("strike A    " + plane_1_strike + "    slip A    " + plane_1_slip_strike +
                                          "    dip A    " + plane_1_dip)
            self.mti_text.appendPlainText("strike B    " + plane_2_strike + "    slip B    " + plane_2_slip_strike +
                                          "    dip B    " + plane_2_dip)
            self.mti_text.appendPlainText("#####################################################")


class ImageLabel(QLabel):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path  # Store the image file path

    def mouseDoubleClickEvent(self, event):
        if event.button() == qt.LeftButton:
            # Open the image with the system's default image viewer
            try:
                if platform.system() == 'Darwin':

                    subprocess.run(['open', self.image_path])
                else:
                    subprocess.run(['xdg-open', self.image_path])

            except Exception as e:
                print(f"Error opening image: {e}")
