
from matplotlib import rcParams
from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5 import QtWidgets
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PyQt5.QtWidgets import QVBoxLayout
rcParams["font.family"] = "Ubuntu"
rcParams["font.size"] = 8
rcParams['axes.linewidth'] = 0.4
rcParams['patch.linewidth'] = .25

class MatplotlibWidget(QtWidgets.QWidget):
    clickEventSignal = pyqtSignal(float, float)
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111, projection=ccrs.PlateCarree())

        self.lat = inset_axes(self.ax, width=0.4, height="100%", loc='center left',
                              bbox_to_anchor=(-0.08, 0.0, 1, 1),
                              bbox_transform=self.ax.transAxes, borderpad=0)
        self.lon = inset_axes(self.ax, width="100.0%", height=0.4, loc='upper center',
                              bbox_to_anchor=(0.00, 0.14, 1, 1),
                              bbox_transform=self.ax.transAxes, borderpad=0)
        self.lat.set_ylim(self.ax.get_ylim())
        self.canvas = Canvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)  # Add the toolbar after the canvas to place it at the bottom

        self.setLayout(self.vbl)


        # Connect the mouse click and key press events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_click(self, event):
        if event.inaxes == self.ax and event.dblclick:
            # Check if the double click happened on the map
            latitude = event.ydata
            longitude = event.xdata
            # Emit the custom signal with latitude and longitude
            self.clickEventSignal.emit(latitude, longitude)

    def on_key_press(self, event):
        # Handle the key press event here
        key_press_handler(event, self.canvas, self.toolbar)

        # Check if the pressed key is "t"
        if event.key == 't':
            # Print the latitude and longitude from the axes limits
            self.handle_latitude_longitude(self.ax.get_ylim()[0], self.ax.get_xlim()[0])

    def handle_latitude_longitude(self, latitude, longitude):
        # Common method to handle latitude and longitude
        print(f'Latitude: {latitude}, Longitude: {longitude}')

    def sizeHint(self):
        return self.canvas.sizeHint()
    #
    def minimumSizeHint(self):
         return QSize(10, 10)





