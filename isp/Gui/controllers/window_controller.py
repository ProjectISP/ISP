# Singleton/SingletonDecorator.py
from isp.Gui.Frames import MainFrame, TimeFrequencyFrame, EarthquakeAnalysisFrame, ArrayAnalysisFrame


class Singleton:
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        if not self.instance:
            self.instance = self.klass(*args, **kwargs)
        return self.instance


@Singleton
class Controller:
    def __init__(self):
        self.main_frame = None
        self.time_frequency_frame = None
        self.earthquake_analysis_frame = None
        self.array_analysis_frame = None

    def open_main_window(self):
        # Start the ui designer
        self.main_frame = MainFrame()
        # bind clicks
        self.main_frame.seismogramButton.clicked.connect(self.open_seismogram_window)
        self.main_frame.earthquakeButton.clicked.connect(self.open_earthquake_window)
        self.main_frame.arrayAnalysisButton.clicked.connect(self.open_array_window)

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
