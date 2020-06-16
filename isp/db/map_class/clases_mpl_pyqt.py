from matplotlib import rcParams
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QSize
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from isp.Gui import pw
import cartopy
from matplotlib.transforms import offset_copy
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# class MplCanvas(FigureCanvas):
#     def __init__(self):
#         #fig = Figure(figsize=(width, height), dpi=dpi)
#         #self.axes = fig.add_subplot(111)
#         self.fig = Figure()
#         self.ax = self.fig.add_subplot(111)
#
#         # self.ax.set_aspect(aspect='auto')
#         # divider = make_axes_locatable(self.ax)
#         # self.ax_lon = divider.append_axes("top", 1.2, pad=0.1, sharex=self.ax)
#         # self.ax_lat = divider.append_axes("right", 1.2, pad=0.1, sharey=self.ax)
#         # self.ax_lon.xaxis.set_tick_params(labelbottom=False)
#         # self.ax_lat.yaxis.set_tick_params(labelleft=False)
#
#         FigureCanvas.__init__(self, self.fig)
#
# class MatplotlibWidget(pw.QWidget):
#     def __init__(self, parent = None):
#         super().__init__(parent)
#         self.canvas = MplCanvas()
#         self.vbl = pw.QVBoxLayout()
#         self.vbl.addWidget(self.canvas)
#         self.toolbar = NavigationToolbar(self.canvas, self)
#         self.vbl.addWidget(self.toolbar)
#         self.setLayout(self.vbl)
rcParams["font.family"] = "Ubuntu"
rcParams["font.size"] = 8
rcParams['axes.linewidth'] = 0.4
rcParams['patch.linewidth'] = .25


class MatplotlibWidget(Canvas):
    def __init__(self, parent=None):

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111, projection = ccrs.Mercator())


        self.lat = inset_axes(self.ax, width=0.4,height="100%",loc='center left',bbox_to_anchor=(-0.1, 0.0, 1, 1),
                bbox_transform=self.ax.transAxes, borderpad=0)
        self.lon = inset_axes(self.ax, width="100.0%", height=0.4, loc='upper center', bbox_to_anchor=(0.0, 0.12, 1, 1),
                             bbox_transform=self.ax.transAxes, borderpad=0)

        super(MatplotlibWidget, self).__init__(self.fig)
        self.setParent(parent)
        super(MatplotlibWidget, self).setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        super(MatplotlibWidget, self).updateGeometry()

    def sizeHint(self):
        return QSize(*self.get_width_height())

    def minimumSizeHint(self):
        return QSize(10, 10)
