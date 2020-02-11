import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from obspy import read
from isp.Gui.Frames import MatplotlibFrame, BaseFrame,FilesView, MessageDialog, \
    MatplotlibCanvas, TimeSelectorBox, FilterBox, SpectrumBox, UiTimeAnalysisWidget, UiArrayAnalysisFrame

class ArrayAnalysisFrame(BaseFrame, UiArrayAnalysisFrame):

    def __init__(self):
        super(ArrayAnalysisFrame, self).__init__()
        self.setupUi(self)