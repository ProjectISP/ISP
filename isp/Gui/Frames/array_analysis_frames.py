import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from obspy import read
from isp.Gui.Frames import MatplotlibFrame, BaseFrame,FilesView, MessageDialog, \
    MatplotlibCanvas, TimeSelectorBox, FilterBox, UiArrayAnalysisFrame, CartopyCanvas
from isp.Gui.Utils.pyqt_utils import BindPyqtObject
from isp.Gui import pw
import os
from isp.arrayanalysis import response_array


class ArrayAnalysisFrame(BaseFrame, UiArrayAnalysisFrame):

    def __init__(self):
        super(ArrayAnalysisFrame, self).__init__()
        self.setupUi(self)
        self.__stations_dir = None
        self.canvas = MatplotlibCanvas(self.responseMatWidget)
        self.cartopy_canvas = CartopyCanvas(self.widget_map)
        self.cartopy_canvas.figure.subplots_adjust(left=0.065, bottom=0.1440, right=0.9, top=0.990, wspace=0.2, hspace=0.0)
        self.cartopy_canvas.figure.tight_layout()
        self.canvas.set_new_subplot(1, ncols=1)
        self.root_path_bind = BindPyqtObject(self.rootPathForm)

        self.fmin_bind = BindPyqtObject(self.fminSB)
        self.fmax_bind = BindPyqtObject(self.fmaxSB)
        self.grid_bind = BindPyqtObject(self.gridSB)
        self.smax_bind = BindPyqtObject(self.smaxSB)

        #Qt Components

        #self.time_selector = TimeSelectorBox(self.PlotToolsWidget, 0)
        #self.filter = FilterBox(self.PlotToolsWidget, 1)


        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))

        #Action Buttons
        self.arfBtn.clicked.connect(lambda: self.arf())


    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        if dir_path:
            bind.value = dir_path

    def arf(self):
        coords_path = os.path.join(self.root_path_bind.value, "coords.txt")
        arf,coords = response_array.arf(coords_path, self.fmin_bind.value, self.fmax_bind.value, self.smax_bind.value)
        slim = self.smax_bind.value
        ##Plotting##
        sstep = slim / len(arf)
        x = np.linspace(-1 * slim, slim, (slim - (-1 * slim) / sstep))
        y = np.linspace(-1 * slim, slim, (slim - (-1 * slim) / sstep))
        X, Y = np.meshgrid(x, y)
        self.canvas.plot_contour(x, y, arf, axes_index=0, clabel="Power [dB]", cmap=plt.get_cmap("jet"))
        self.canvas.set_xlabel(0, "Sx (s/km)")
        self.canvas.set_ylabel(0, "Sy (s/km)")
        lon=coords[:,0]
        lat=coords[:,1]
        depth=coords[:,2]
        self.cartopy_canvas.plot_stations(lon, lat,depth, 0)

