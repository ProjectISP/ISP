# Singleton/SingletonDecorator.py
from isp.Gui.Frames import MainFrame, SeismogramFrame, EarthquakeAnalysisFrame


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
        self.seismogram_frame = None
        self.mpw = None

    def open_main_window(self):
        # Start the ui designer
        self.main_frame = MainFrame()
        # bind clicks
        self.main_frame.seismogramButton.clicked.connect(self.open_seismogram_window)
        self.main_frame.earthquakeButton.clicked.connect(self.open_earthquake_window)

        # show frame
        self.main_frame.show()

    def open_seismogram_window(self):
        # Start the ui designer
        self.seismogram_frame = SeismogramFrame()
        self.seismogram_frame.show()


    def open_earthquake_window(self):
        # Start the ui designer
        self.earthquake_analysis_frame = EarthquakeAnalysisFrame()
        self.earthquake_analysis_frame.show()
