# Singleton/SingletonDecorator.py
from isp.Gui.Frames import MainFrame, TimeFrequencyFrame, EarthquakeAnalysisFrame, ArrayAnalysisFrame, MTIFrame,\
    RecfFrame, EventLocationFrame

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

    def open_main_window(self):
        # Start the ui designer
        self.main_frame = MainFrame()
        # bind clicks
        self.main_frame.seismogramButton.clicked.connect(self.open_seismogram_window)
        self.main_frame.earthquakeButton.clicked.connect(self.open_earthquake_window)
        self.main_frame.arrayAnalysisButton.clicked.connect(self.open_array_window)
        self.main_frame.momentTensorButton.clicked.connect(self.open_momentTensor_window)
        self.main_frame.receiverFunctionsButton.clicked.connect(self.open_receiverFunctions)
        self.main_frame.actionOpen_Project.triggered.connect(self.open_project)
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

    def open_project(self):
        if not self.project_frame:
            self.project_frame = EventLocationFrame()
        if not self.project_frame.isVisible() :
            self.project_frame.refreshLimits()
        self.project_frame.show()


