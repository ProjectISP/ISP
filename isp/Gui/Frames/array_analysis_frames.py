import matplotlib.pyplot as plt
import numpy as np
from isp.Gui.Frames import  BaseFrame, \
    MatplotlibCanvas,  UiArrayAnalysisFrame, CartopyCanvas
from isp.Gui.Utils.pyqt_utils import BindPyqtObject, convert_qdatetime_utcdatetime
from isp.Gui import pw
import os
import matplotlib.dates as mdates

from isp.arrayanalysis import array_analysis


class ArrayAnalysisFrame(BaseFrame, UiArrayAnalysisFrame):

    def __init__(self):
        super(ArrayAnalysisFrame, self).__init__()
        self.setupUi(self)
        self.__stations_dir = None
        self.canvas = MatplotlibCanvas(self.responseMatWidget)
        self.canvas_fk = MatplotlibCanvas(self.widget_fk,nrows=4)
        self.canvas_slow_map = MatplotlibCanvas(self.widget_slow_map)
        self.canvas_fk.on_double_click(self.on_click_matplotlib)
        self.canvas_stack = MatplotlibCanvas(self.widget_stack)
        self.cartopy_canvas = CartopyCanvas(self.widget_map)
        self.cartopy_canvas.figure.subplots_adjust(left=0.065, bottom=0.1440, right=0.9, top=0.990, wspace=0.2, hspace=0.0)
        self.cartopy_canvas.figure.tight_layout()
        self.canvas.set_new_subplot(1, ncols=1)

        #Binding

        self.root_path_bind = BindPyqtObject(self.rootPathForm)
        self.root_pathFK_bind = BindPyqtObject(self.rootPathFormFK)
        self.dataless_path_bind = BindPyqtObject(self.datalessPathForm)
        self.stationsCoords_bind = BindPyqtObject(self.coordsPathForm)

        self.fmin_bind = BindPyqtObject(self.fminSB)
        self.fmax_bind = BindPyqtObject(self.fmaxSB)
        self.grid_bind = BindPyqtObject(self.gridSB)
        self.smax_bind = BindPyqtObject(self.smaxSB)


        self.fminFK_bind = BindPyqtObject(self.fminFKSB)
        self.fmaxFK_bind = BindPyqtObject(self.fmaxFKSB)
        self.overlap_bind = BindPyqtObject(self.overlapSB)
        self.timewindow_bind = BindPyqtObject(self.timewindowSB)
        self.smaxFK_bind = BindPyqtObject(self.slowFKSB)
        self.slow_grid_bind = BindPyqtObject(self.gridFKSB)
        #Qt Components
        #self.time_selector = TimeSelectorBox(self.PlotToolsWidget, 0)
        #self.filter = FilterBox(self.PlotToolsWidget, 1)

        # Bind buttons
        self.selectDirBtn.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))
        self.selectDirBtnFK.clicked.connect(lambda: self.on_click_select_directory(self.root_pathFK_bind))
        self.datalessBtn.clicked.connect(lambda: self.on_click_select_directory(self.dataless_path_bind))
        self.stationsCoordsBtn.clicked.connect(lambda: self.on_click_select_directory(self.stationsCoords_bind))

        #Action Buttons
        self.arfBtn.clicked.connect(lambda: self.arf())
        self.runFKBtn.clicked.connect(lambda: self.FK_plot())

    def on_click_select_directory(self, bind: BindPyqtObject):
        dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        if dir_path:
            bind.value = dir_path

    def arf(self):

        coords_path = os.path.join(self.root_path_bind.value, "coords.txt")
        arf,coords = array_analysis.array.arf(coords_path, self.fmin_bind.value, self.fmax_bind.value, self.smax_bind.value)

        slim = self.smax_bind.value
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

    def FK_plot(self):
        #xlocator = mdates.AutoDateLocator()
        starttime = convert_qdatetime_utcdatetime(self.starttime_date)
        endtime = convert_qdatetime_utcdatetime(self.endtime_date)
        print(starttime)
        print(endtime)
        wavenumber = array_analysis.array()
        relpower,abspower, AZ, Slowness, T = wavenumber.FK(self.root_pathFK_bind.value, self.stationsCoords_bind.value, starttime, endtime,
        self.fminFK_bind.value, self.fmaxFK_bind.value, self.smaxFK_bind.value, self.slow_grid_bind.value,
        self.timewindow_bind.value, self.overlap_bind.value)
        self.canvas_fk.scatter3d(T,relpower,relpower, axes_index=0, clabel="Power [dB]")
        self.canvas_fk.scatter3d(T, abspower, relpower, axes_index=1, clabel="Power [dB]")
        self.canvas_fk.scatter3d(T, AZ, relpower, axes_index=2, clabel="Power [dB]")
        self.canvas_fk.scatter3d(T, Slowness, relpower, axes_index=3, clabel="Power [dB]")
        self.canvas_fk.set_ylabel(0, " Rel Power ")
        self.canvas_fk.set_ylabel(1, " Absolute Power ")
        self.canvas_fk.set_ylabel(2, " Back Azimuth ")
        self.canvas_fk.set_ylabel(3, " Slowness ")
        self.canvas_fk.set_xlabel(3, "Time [s]")
        #self.canvas_fk.set_major_locator(3,xlocator)
        #self.canvas_fk.set_major_formatter(3, mdates.AutoDateFormatter(xlocator))


    def on_click_matplotlib(self, event, canvas):
        if isinstance(canvas, MatplotlibCanvas):
            wavenumber = array_analysis.array()
            starttime = convert_qdatetime_utcdatetime(self.starttime_date)
            x1, y1 = event.xdata, event.ydata
            DT = x1
            X, Y, Z = wavenumber.FKCoherence(self.root_pathFK_bind.value, self.stationsCoords_bind.value,
            starttime, DT , self.fminFK_bind.value, self.fmaxFK_bind.value, self.smaxFK_bind.value, self.timewindow_bind.value,
                                   self.slow_grid_bind.value, self.methodSB.currentText())
            if self.methodSB.currentText() == "FK":
                clabel="Power"
            elif self.methodSB.currentText() == "MTP.COHERENCE":
                clabel = "Magnitude Coherence"
            self.canvas_slow_map.plot_contour(X, Y, Z, axes_index=0, clabel=clabel, cmap=plt.get_cmap("jet"))
            self.canvas_slow_map.set_xlabel(0, "Sx [s/km]")
            self.canvas_slow_map.set_Ylabel(0, "Sy [s/km]")





