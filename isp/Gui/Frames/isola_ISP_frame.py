from obspy import Stream, UTCDateTime
from isp import ROOT_DIR
from isp.DataProcessing import SeismogramDataAdvanced
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Exceptions import InvalidFile
from isp.Gui import pw, pyc
from isp.Gui.Frames import BaseFrame, MatplotlibCanvas, MessageDialog, UiMomentTensor, MatplotlibFrame
from isp.Gui.Frames.crustal_model_parameters_frame import CrustalModelParametersFrame
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, add_save_load, convert_qdatetime_utcdatetime
from isp.Utils import MseedUtil, ObspyUtil, AsycTime
from isp.earthquakeAnalisysis.stations_map import StationsMap
from isp.mti.mti_utilities import MTIManager
from isp.mti.class_isola_new import *
from isp.Gui.Frames.help_frame import HelpDoc
import pandas as pd

@add_save_load()
class MTIFrame(BaseFrame, UiMomentTensor):

    def __init__(self):
        super(MTIFrame, self).__init__()
        self.setupUi(self)

        super(MTIFrame, self).__init__()
        self.setupUi(self)
        self.__stations_dir = None
        self.__metadata_manager = None
        self.inventory = {}
        self._stations_info = {}
        self.stream = None
        # Binding
        self.root_path_bind = BindPyqtObject(self.rootPathForm)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)
        self.metadata_path_bind = BindPyqtObject(self.datalessPathForm, self.onChange_metadata_path)
        self.earth_path_bind =  BindPyqtObject(self.earth_modelPathForm)

        # Binds
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.datalessBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        self.earthmodelBtn.clicked.connect(lambda: self.on_click_select_file(self.earth_path_bind))

        # Action Buttons
        self.actionSettings.triggered.connect(lambda: self.open_parameters_settings())
        self.plotBtn.clicked.connect(self.plot_seismograms)
        self.actionSettings.triggered.connect(lambda: self.open_parameters_settings())
        self.actionWrite.triggered.connect(self.write)
        self.actionEarth_Model.triggered.connect(lambda: self.open_earth_model())
        self.actionFrom_File.triggered.connect(lambda: self.load_event_from_isolapath())
        self.actionOpen_Help.triggered.connect(lambda: self.open_help())
        self.stationsBtn.clicked.connect(self.stationsInfo)
        self.run_inversionBtn.clicked.connect(lambda: self.run_inversion())
        self.stations_mapBtn.clicked.connect(lambda: self.plot_map_stations())
        self.plot_solutionBtn.clicked.connect(lambda: self.plot_solution())
        #self.earthmodelBtn.clicked.connect(self.read_earth_model)
        # Parameters settings
        self.parameters = ParametersSettings()
        self.earth_model = CrustalModelParametersFrame()
        # help Documentation

        self.help = HelpDoc()

    def open_parameters_settings(self):
        self.parameters.show()

    def open_earth_model(self):
        self.earth_model.show()

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def message_dataless_not_found(self):
        if len(self.dataless_not_found) > 1:
            md = MessageDialog(self)
            md.set_info_message("Metadata not found.")
        else:
            for file in self.dataless_not_found:
                md = MessageDialog(self)
                md.set_info_message("Metadata for {} not found.".format(file))

    def validate_file(self):
        if not MseedUtil.is_valid_mseed(self.file_selector.file_path):
            msg = "The file {} is not a valid mseed. Please, choose a valid format". \
                format(self.file_selector.file_name)
            raise InvalidFile(msg)

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        if dir_path:
            bind.value = dir_path

    def on_click_select_file(self, bind: BindPyqtObject):
        file_path = pw.QFileDialog.getOpenFileName(self, 'Select Directory', bind.value)
        file_path = file_path[0]

        if file_path:
            bind.value = file_path

    @AsycTime.run_async()
    def onChange_metadata_path(self, value):
        try:
            self.__metadata_manager = MetadataManager(value)
            self.inventory = self.__metadata_manager.get_inventory()
        except:
            pass

    # def read_earth_model(self):
    #     model = self.earth_model.getParametersWithFormat()
    #     print(model)

    def plot_seismograms(self):
        parameters = self.get_inversion_parameters()
        lat = float(parameters['latitude'])
        lon = float(parameters['longitude'])
        starttime = convert_qdatetime_utcdatetime(self.starttime_date)
        endtime = convert_qdatetime_utcdatetime(self.endtime_date)
        diff = endtime - starttime

        parameters = self.parameters.getParameters()
        all_traces = []
        obsfiles = MseedUtil.get_mseed_files(self.root_path_bind.value)
        obsfiles.sort()

        for file in obsfiles:
            sd = SeismogramDataAdvanced(file)
            if self.trimCB.isChecked() and diff >= 0:
                tr = sd.get_waveform_advanced(parameters, self.inventory,
                                              filter_error_callback=self.filter_error_message,
                                              start_time=starttime, end_time=endtime)
            else:
                tr = sd.get_waveform_advanced(parameters, self.inventory,
                                              filter_error_callback=self.filter_error_message)

            all_traces.append(tr)

        self.st = Stream(traces=all_traces)
        self.stream_frame = MatplotlibFrame(self.st, type='normal')
        self.stream_frame.show()

        if self.st:
            min_dist = self.min_distCB.value()
            max_dist = self.max_distCB.value()
            mt =  MTIManager(self.st, self.inventory, lat, lon, min_dist, max_dist)
            [self.stream, self.deltas, self.stations_isola_path] = mt.get_stations_index()


    def stationsInfo(self):

        file_path = self.root_path_bind.value
        obsfiles = MseedUtil.get_mseed_files(self.root_path_bind.value)
        obsfiles.sort()
        sd = []

        for file in obsfiles:

            st = SeismogramDataAdvanced(file)

            station = [st.stats.Network,st.stats.Station,st.stats.Location,st.stats.Channel,st.stats.StartTime,
                       st.stats.EndTime, st.stats.Sampling_rate, st.stats.Npts]

            sd.append(station)

        self._stations_info = StationsInfo(sd, check = True)
        self._stations_info.show()



    def write(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', root_path)
        n=len(self.st)
        for j in range(n):
            tr=self.st[j]
            print(tr.id, "Writing data processed")
            path_output =  os.path.join(dir_path, tr.id)
            tr.write(path_output, format="MSEED")

    ##In progress##
    def load_event_from_isolapath(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        file_path = pw.QFileDialog.getOpenFileName(self, 'Select Directory', root_path)
        file =file_path[0]
        frame = pd.read_csv(file, sep='\s+', header=None)
        time = frame.iloc[3][0]
        year = time[0:4]
        mm = time[4:6]
        dd = time[6:8]
        hour = frame.iloc[4][0]
        minute = frame.iloc[5][0]
        sec = frame.iloc[6][0]
        sec = float(sec)
        dec = sec - int(sec)
        dec = int(sec)

        time = UTCDateTime(int(year), int(mm), int(dd), int(hour), int(minute), int(sec), dec)
        event = {'lat': frame.iloc[0][0], 'lon': frame.iloc[0][1], 'depth': frame.iloc[1][0],
                 'mag': frame.iloc[2][0], 'time': time, 'istitution': frame.iloc[7][0]}

        return event

    @AsycTime.run_async()
    def run_inversion(self):
        parameters = self.get_inversion_parameters()
        try:
            stations_map = self._stations_info.get_stations_map()
        except:
            md = MessageDialog(self)
            md.set_info_message("Press Stations info and check your selection")


        if len(self.stream) and len(stations_map)> 0:

            isola = ISOLA(self.stream, self.deltas, location_unc = parameters['location_unc'], depth_unc = parameters['depth_unc'],
                           time_unc = parameters['time_unc'], deviatoric =  parameters['deviatoric'], threads = 8,
                           circle_shape = parameters['circle_shape'], use_precalculated_Green = parameters['GFs'])
            #
            isola.set_event_info(parameters['latitude'], parameters['longitude'], parameters['depth'],
                                 parameters['magnitude'],parameters['origin_time'])
            #
            print(isola.event)
            #
            #
            if self.stations_isola_path:
                isola.read_network_coordinates(self.stations_isola_path)
                isola.set_use_components(stations_map)
                #print(isola.stations)
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
                        isola.covariance_matrix(crosscovariance=True, save_non_inverted=True)
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
                if len(isola.depths) > 1:
                   isola.plot_slices()
                if len(isola.grid) > len(isola.depths) and len(isola.depths) > 1:
                    isola.plot_maps_sum()

                try:

                    isola.plot_MT()
                    isola.plot_uncertainty(n=400)
                    #plot_MT_uncertainty_centroid()
                    isola.plot_seismo('seismo.png')
                    isola.plot_seismo('seismo_sharey.png', sharey=True)
                    isola.plot_seismo('seismo_cova.png', cholesky=True)
                    isola.plot_noise()
                    isola.plot_spectra()
                    isola.plot_stations()
                except:
                    print("Couldn't Plot")
                    isola.plot_covariance_matrix(colorbar=True)
                    #isola.plot_3D()



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

                    print("Couldn't load url")


                self.infoTx.appendPlainText("Moment Tensor Inversion Successfully done !!!, please plot last solution")
            else:
                pass



    def plot_solution(self):
        path = os.path.join(ROOT_DIR, 'mti/output/index.html')
        url = pyc.QUrl.fromLocalFile(path)
        self.widget.load(url)

    def get_inversion_parameters(self):
        parameters = {'latitude': self.latDB.value(), 'longitude':self.lonDB.value(), 'depth':self.depthDB.value(),
                      'origin_time':convert_qdatetime_utcdatetime(self.origin_time),
                      'location_unc':self.location_uncDB.value(),'time_unc':self.timeDB.value(),
                      'magnitude':self.magnitudeDB.value(),
                      'depth_unc':self.depth_uncDB.value(),'freq_min':self.freq_min_DB.value(),
                      'freq_max':self.freq_max_DB.value(),'deviatoric': self.deviatoricCB.isChecked(),
                      'circle_shape':self.circle_shapeCB.isChecked(),'GFs':self.gfCB.isChecked(),
                      'covariance':self.covarianceCB.isChecked()}
        return parameters



    def plot_map_stations(self):
        stations = []
        obsfiles = MseedUtil.get_mseed_files(self.root_path_bind.value)
        obsfiles.sort()
        try:
            if len(self.stream)>0:
                stations = ObspyUtil.get_stations_from_stream(self.stream)
        except:
            pass

        map_dict={}
        sd = []

        for file in obsfiles:
            if len(stations) == 0:
                st = SeismogramDataAdvanced(file)

                name = st.stats.Network+"."+st.stats.Station

                sd.append(name)

                st_coordinates = self.__metadata_manager.extract_coordinates(self.inventory, file)

                map_dict[name] = [st_coordinates.Latitude, st_coordinates.Longitude]
            else:
                st = SeismogramDataAdvanced(file)
                if st.stats.Station in stations:
                    name = st.stats.Network + "." + st.stats.Station

                    sd.append(name)

                    st_coordinates = self.__metadata_manager.extract_coordinates(self.inventory, file)

                    map_dict[name] = [st_coordinates.Latitude, st_coordinates.Longitude]
                else:
                    pass


        self.map_stations = StationsMap(map_dict)
        self.map_stations.plot_stations_map(latitude = self.latDB.value(),longitude=self.lonDB.value())

    def open_help(self):
        self.help.show()






