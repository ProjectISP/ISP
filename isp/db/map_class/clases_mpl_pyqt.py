from matplotlib import rcParams
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
rcParams["font.family"] = "Ubuntu"
rcParams["font.size"] = 8
rcParams['axes.linewidth'] = 0.4
rcParams['patch.linewidth'] = .25


class MatplotlibWidget(Canvas):
    def __init__(self, parent=None):

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111, projection = ccrs.PlateCarree())


        self.lat = inset_axes(self.ax, width=0.4,height="100%",loc='center left',bbox_to_anchor=(-0.08, 0.0, 1, 1),
                bbox_transform=self.ax.transAxes, borderpad=0)
        self.lon = inset_axes(self.ax, width="100.0%", height=0.4, loc='upper center', bbox_to_anchor=(0.0, 0.14, 1, 1),
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

