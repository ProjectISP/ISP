from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QWidget, QVBoxLayout
from isp.Exceptions import InvalidFile
from isp.Gui import pw, qt
from isp.Gui.Frames import BaseFrame, MessageDialog, UiMomentTensor, MatplotlibFrame
from isp.Gui.Frames.crustal_model_parameters_frame import CrustalModelParametersFrame
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, add_save_load, convert_qdatetime_utcdatetime
from isp.Utils import MseedUtil
from isp.Utils.subprocess_utils import open_html_file, open_url
from isp.mti.mti_utilities import MTIManager
from isp.mti.class_isola_new import *
from isp.Gui.Frames.help_frame import HelpDoc
from obspy import Stream, UTCDateTime, Inventory
import platform

@add_save_load()
class MTIFrame(BaseFrame, UiMomentTensor):

    def __init__(self):
        super(MTIFrame, self).__init__()
        self.setupUi(self)
        self.__stations_dir = None
        self.__metadata_manager = None
        self.inventory = {}
        self._stations_info = {}
        self.stream = None
        self.stations_check = False
        self.url = 'https://projectisp.github.io/ISP_tutorial.github.io/mti/'
        self.scroll_area_widget.setWidgetResizable(True)
        # Binding

        self.earth_path_bind = BindPyqtObject(self.earth_modelPathForm)
        self.output_path_bind = BindPyqtObject(self.outputLE)
        # Binds
        self.earthmodelBtn.clicked.connect(lambda: self.on_click_select_file(self.earth_path_bind))
        self.setOutputBtn.clicked.connect(lambda: self.on_click_select_directory(self.output_path_bind))

        # Action Buttons

        self.actionEarth_Model.triggered.connect(lambda: self.open_earth_model())
        #self.actionFrom_File.triggered.connect(lambda: self.load_event_from_isolapath())
        self.actionOpen_Help.triggered.connect(lambda: self.open_help())
        self.run_inversionBtn.clicked.connect(lambda: self.run_inversion())
        self.plot_solutionBtn.clicked.connect(lambda: self.plot_solution())
        self.stationSelectBtn.clicked.connect(lambda: self.stationsInfo())
        self.openHTML.clicked.connect(lambda:self.load_HTML_file())
        self.earth_model = CrustalModelParametersFrame()
        # help Documentation

        self.help = HelpDoc()


    def open_earth_model(self):
        self.earth_model.show()

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path

    def validate_file(self):
        if not MseedUtil.is_valid_mseed(self.file_selector.file_path):
            msg = "The file {} is not a valid mseed. Please, choose a valid format". \
                format(self.file_selector.file_name)
            raise InvalidFile(msg)


    def on_click_select_file(self, bind: BindPyqtObject):
        file_path = pw.QFileDialog.getOpenFileName(self, 'Select Directory', bind.value)
        file_path = file_path[0]

        if file_path:
            bind.value = file_path


    def stationsInfo(self):

        self.stations_check = True
        md = MessageDialog(self)
        md.set_info_message("Select your Station/Channel", "Sorted by distance to the epicenter")
        parameters = self.get_inversion_parameters()
        lat = float(parameters['latitude'])
        lon = float(parameters['longitude'])

        if self.st:
            min_dist = self.min_distCB.value()
            max_dist = self.max_distCB.value()
            mt = MTIManager(self.st, self.inventory, lat, lon, min_dist, max_dist)
            [self.stream, self.deltas, self.stations_isola_path] = mt.get_stations_index()

        all_stations = []

        for stream in self.stream:
            for tr in stream:
                station = [tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel, tr.stats.starttime,
                           tr.stats.endtime, tr.stats.sampling_rate, tr.stats.npts]

                all_stations.append(station)

        self._stations_info = StationsInfo(all_stations, check=True)
        self._stations_info.show()


    # def read_earth_model(self):
    #     model = self.earth_model.getParametersWithFormat()
    #     print(model)


    def send_mti(self, stream: Stream, inventory: Inventory, starttime: UTCDateTime, endtime: UTCDateTime):

        self.st = stream
        print(self.st)

        self.inventory = inventory
        self.starttime = starttime
        self.endttime = endtime
        self.stream_frame = MatplotlibFrame(self.st, type='normal')
        self.stream_frame.show()

    #@AsycTime.run_async()
    def run_inversion(self):

        parameters = self.get_inversion_parameters()

        if self.st and self.stations_check:

            self.infoTx.clear()

            if parameters['GFs']:
                pass
            else:
                MTIManager.clean_and_create_symlinks()

            stations_map = self._stations_info.get_stations_map()

            if len(self.stream) and len(stations_map) > 0:

                isola = ISOLA(self.stream, self.deltas, location_unc=parameters['location_unc'], depth_unc=parameters['depth_unc'],
                               time_unc=parameters['time_unc'], deviatoric=parameters['deviatoric'], threads=8,
                               circle_shape=parameters['circle_shape'], use_precalculated_Green=parameters['GFs'])

                isola.set_event_info(parameters['latitude'], parameters['longitude'], parameters['depth'],
                                     parameters['magnitude'], parameters['origin_time'])

                print(isola.event)

                if self.stations_isola_path:
                    isola.read_network_coordinates(self.stations_isola_path)
                    isola.set_use_components(stations_map)
                    print(isola.stations)
                    isola.read_crust(self.earth_path_bind.value)

                    isola.set_parameters(parameters['freq_max'], parameters['freq_min'])
                    self.infoTx.setPlainText("Calculated GFs")
                    if not isola.calculate_or_verify_Green():
                        exit()
                    self.infoTx.appendPlainText("Filtered and trim")
                    isola.trim_filter_data()
                    try:

                        if parameters['covariance']:
                            self.infoTx.appendPlainText("Calculating Covariance Matrix")
                            isola.covariance_matrix(crosscovariance=True, save_non_inverted=True, save_covariance_function=True)
                    except:
                        md = MessageDialog(self)
                        md.set_error_message("No Possible calculate covariance matrix, "
                                             "please try increasing the noise time window")
                #
                    self.infoTx.appendPlainText("decimate and shift")
                    isola.decimate_shift()

                    self.infoTx.appendPlainText("Run inversion")
                    isola.run_inversion()

                    self.infoTx.appendPlainText("Finished Inversion")
                    isola.find_best_grid_point()
                    isola.print_solution()
                    isola.print_fault_planes()

                    self.infoTx.appendPlainText("Plotting Solutions")
                    if len(isola.grid) > len(isola.depths):
                        isola.plot_maps()
                        self.infoTx.appendPlainText("plot_maps")
                    if len(isola.depths) > 1:
                       isola.plot_slices()
                       self.infoTx.appendPlainText("plot_slices")
                    if len(isola.grid) > len(isola.depths) and len(isola.depths) > 1:
                        isola.plot_maps_sum()
                        self.infoTx.appendPlainText("plot_maps_sum")

                    try:

                        isola.plot_MT()
                        self.infoTx.appendPlainText("plot_MT")
                        isola.plot_uncertainty(n=400)
                        self.infoTx.appendPlainText("plot_uncertainty")
                        #plot_MT_uncertainty_centroid()
                        isola.plot_seismo('seismo.png')
                        isola.plot_seismo('seismo_sharey.png', sharey=True)
                        self.infoTx.appendPlainText("plot_seismo")

                        if self.covarianceCB.isChecked():
                            isola.plot_seismo('plot_seismo.png', cholesky=True)
                            self.infoTx.appendPlainText("plot_seismo_cova")
                            isola.plot_noise()
                            self.infoTx.appendPlainText("plot_noise")
                            isola.plot_spectra()
                            self.infoTx.appendPlainText("plot_spectra")

                        isola.plot_stations()
                        self.infoTx.appendPlainText("plot_stations")

                    except:
                        print("Couldn't Plot")

                    try:
                        if self.covarianceCB.isChecked():
                           isola.plot_covariance_matrix(colorbar=True)
                        #isola.plot_3D()
                    except:
                        pass

            try:
                isola.html_log(h1='ISP Moment Tensor inversion', plot_MT='centroid.png',
                               plot_uncertainty='uncertainty.png', plot_stations='stations.png',
                               plot_seismo_cova='seismo_cova.png',
                               plot_seismo_sharey='seismo_sharey.png', plot_spectra='spectra.png',
                               plot_noise='noise.png',
                               plot_covariance_matrix='covariance_matrix.png', plot_maps='map.png',
                               plot_slices='slice.png',
                               plot_maps_sum='map_sum.png')
            except:
                self.infoTx.appendPlainText("Couldn't load url")
            try:
                isola.html_log(h1='ISP Moment Tensor inversion', plot_MT='centroid.png',
                               plot_uncertainty='uncertainty.png', plot_stations='stations.png',
                               plot_seismo_sharey='seismo_sharey.png', plot_maps='map.png',
                               plot_slices='slice.png')
            except:
                self.infoTx.appendPlainText("Couldn't load url")


            self.infoTx.appendPlainText("Moment Tensor Inversion Successfully done !!!, please plot last solution")

            MTIManager.move_files(self.output_path_bind.value)

        else:
            md = MessageDialog(self)
            md.set_error_message(
                "Please review the following requirements before proceeding:",
                "1. Ensure seismograms are loaded.\n"
                "2. Ensure fill the parametroization box\n"
                "3. Ensure clicked at station channels"
            )


    def load_images(self):
        # Open folder dialog to select image folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.display_images(folder_path)


    def plot_solution(self):

        self._load_log_file()
        beach_ball_list = ['centroid.png', 'uncertainty_MT_DC.png', 'uncertainty_MT.png']
        synthetic_list = ['seismo.png', 'spectra.png']
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
        for filename in os.listdir(self.output_path_bind.value):
            if filename.lower().endswith('.png') and filename in good_list:  # Only load PNG images
                image_path = os.path.join(self.output_path_bind.value, filename)
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


    def get_inversion_parameters(self):
        parameters = {'latitude': self.latDB.value(), 'longitude': self.lonDB.value(), 'depth': self.depthDB.value(),
                      'origin_time': convert_qdatetime_utcdatetime(self.origin_time),
                      'location_unc': self.location_uncDB.value(), 'time_unc': self.timeDB.value(),
                      'magnitude': self.magnitudeDB.value(),
                      'depth_unc': self.depth_uncDB.value(), 'freq_min': self.freq_min_DB.value(),
                      'freq_max': self.freq_max_DB.value(), 'deviatoric': self.deviatoricCB.isChecked(),
                      'circle_shape': self.circle_shapeCB.isChecked(), 'GFs': self.gfCB.isChecked(),
                      'covariance': self.covarianceCB.isChecked()}
        return parameters

    def _load_log_file(self):
        logfile = os.path.join(self.output_path_bind.value, "log.txt")
        if logfile:
            try:
                # Read the content of the file
                with open(logfile, 'r') as file:
                    log_content = file.read()
                    self.infoTx.setPlainText(log_content)  # Display the content in QPlainTextEdit
            except Exception as e:
                self.infoTx.setPlainText(f"Failed to load file: {e}")

    def open_help(self):
        open_url(self.url)

    def load_HTML_file(self):
        html_path = os.path.join(self.output_path_bind.value, 'index.html')
        open_html_file(html_path)

    #
    # def plot_map_stations(self):
    #     md = MessageDialog(self)
    #     md.hide()
    #     try:
    #         stations = []
    #         obsfiles = MseedUtil.get_mseed_files(self.root_path_bind.value)
    #         obsfiles.sort()
    #         try:
    #             if len(self.stream) > 0:
    #                 stations = ObspyUtil.get_stations_from_stream(self.stream)
    #         except:
    #             pass
    #
    #         map_dict={}
    #         sd = []
    #
    #         for file in obsfiles:
    #             if len(stations) == 0:
    #                 st = SeismogramDataAdvanced(file)
    #
    #                 name = st.stats.Network+"."+st.stats.Station
    #
    #                 sd.append(name)
    #
    #                 st_coordinates = self.__metadata_manager.extract_coordinates(self.inventory, file)
    #
    #                 map_dict[name] = [st_coordinates.Latitude, st_coordinates.Longitude]
    #             else:
    #                 st = SeismogramDataAdvanced(file)
    #                 if st.stats.Station in stations:
    #                     name = st.stats.Network + "." + st.stats.Station
    #
    #                     sd.append(name)
    #
    #                     st_coordinates = self.__metadata_manager.extract_coordinates(self.inventory, file)
    #
    #                     map_dict[name] = [st_coordinates.Latitude, st_coordinates.Longitude]
    #                 else:
    #                     pass
    #
    #         self.map_stations = StationsMap(map_dict)
    #         self.map_stations.plot_stations_map(latitude=self.latDB.value(), longitude=self.lonDB.value())
    #
    #         md.set_info_message("Station Map OK !!! ")
    #     except:
    #         md.set_error_message(" Please check you have process and plot seismograms and opened stations info,"
    #                              "Please additionally check that your metada fits with your mseed files")
    #
    #     md.show()




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
