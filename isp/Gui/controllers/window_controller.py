# Singleton/SingletonDecorator.py
import traceback

from isp.Gui.Frames import MainFrame, TimeFrequencyFrame, EarthquakeAnalysisFrame, ArrayAnalysisFrame, MTIFrame, \
    RecfFrame, EventLocationFrame, SyntheticsAnalisysFrame, DataDownloadFrame, RealTimeFrame, NoiseFrame, MessageDialog
from isp.Gui.Frames.ppsds_frame import PPSDFrame
from isp.Utils import Singleton


@Singleton
class Controller:
    def __init__(self):
        self.main_frame = None
        self.time_frequency_frame = None
        self.earthquake_analysis_frame = None
        self.array_analysis_frame = None
        self.moment_tensor_frame = None
        self.receiver_functions_frame = None
        self.project_frame = None
        self.synthetics_frame = None
        self.data_download_frame = None
        self.ppds_frame = None
        self.realtime_frame = None
        self.noise_frame = None


    def open_main_window(self):
        # Start the ui designer
        self.main_frame = MainFrame()
        # bind clicks
        self.main_frame.seismogramButton.clicked.connect(self.open_seismogram_window)
        self.main_frame.earthquakeButton.clicked.connect(self.open_earthquake_window)
        self.main_frame.arrayAnalysisButton.clicked.connect(self.open_array_window)
        self.main_frame.momentTensorButton.clicked.connect(self.open_momentTensor_window)
        self.main_frame.receiverFunctionsButton.clicked.connect(self.open_receiverFunctions)
        self.main_frame.noiseButton.clicked.connect(self.open_noise)
        self.main_frame.actionReal_Time.triggered.connect(self.open_realtime_window)
        self.main_frame.actionOpen_Project.triggered.connect(self.open_project)
        self.main_frame.actionCreate_new_Project.triggered.connect(self.create_project)
        self.main_frame.actionRetrieve_data.triggered.connect(self.retrieve_data)
        self.main_frame.actionPPSDs.triggered.connect(self.ppsds)
        self.main_frame.actionOpen_Help.triggered.connect(lambda: self.open_help())
        # show frame
        self.main_frame.show()

    def open_seismogram_window(self):
        # Start the ui designer
        if not self.time_frequency_frame:
            self.time_frequency_frame = TimeFrequencyFrame()
        self.time_frequency_frame.show()

    def open_earthquake_window(self):
        # Start the ui designer
        if not self.earthquake_analysis_frame:
            self.earthquake_analysis_frame = EarthquakeAnalysisFrame()
        self.earthquake_analysis_frame.show()

    def open_realtime_window(self):
        # Start the ui designer
        if not self.realtime_frame:
            self.realtime_frame = RealTimeFrame()
        self.realtime_frame.show()

    def open_array_window(self):
        # Start the ui designer
        if not self.array_analysis_frame:
            self.array_analysis_frame = ArrayAnalysisFrame()
        self.array_analysis_frame.show()

    def open_momentTensor_window(self):
        # Start the ui designer
        if not self.moment_tensor_frame:
            self.moment_tensor_frame = MTIFrame()
        self.moment_tensor_frame.show()

    def open_receiverFunctions(self):
        if not self.receiver_functions_frame:
            self.receiver_functions_frame = RecfFrame()
        self.receiver_functions_frame.show()

    def open_noise(self):
        if not self.noise_frame:
            self.noise_frame = NoiseFrame()
        self.noise_frame.show()

    def open_project(self):
        if not self.project_frame:
            self.project_frame = EventLocationFrame()
        if not self.project_frame.isVisible() :
            self.project_frame.refreshLimits()
        self.project_frame.show()

    def create_project(self):
        if not self.synthetics_frame:
            self.synthetics_frame = SyntheticsAnalisysFrame()
        self.synthetics_frame.show()


    def retrieve_data(self):
        if not self.data_download_frame:
            self.data_download_frame = DataDownloadFrame()
        self.data_download_frame.show()

    def ppsds(self):
        if not self.ppds_frame:
            self.ppds_frame = PPSDFrame()
        self.ppds_frame.show()

    def open_help(self):
        self.help.show()

    def exception_parse(self, error_cls, exception, exc_traceback):
        md = MessageDialog(self.main_frame)
        detail_error = "".join(traceback.format_exception(error_cls, exception, exc_traceback))
        md.set_error_message(message="{}:{}".format(error_cls.__name__, exception), detailed_message=detail_error)
        md.show()

