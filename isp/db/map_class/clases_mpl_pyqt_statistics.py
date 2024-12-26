from matplotlib import rcParams
from PyQt5.QtCore import QSize
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from PyQt5.QtWidgets import QVBoxLayout
rcParams["font.family"] = "Ubuntu"
rcParams["font.size"] = 8
rcParams['axes.linewidth'] = 0.4
rcParams['patch.linewidth'] = .25

class MatplotlibWidgetStatistics(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidgetStatistics, self).__init__(parent)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(211)
        self.ax3 = self.fig.add_subplot(212)
        #self.fig, (self.ax1, self.ax3) = plt.subplots(nrows=2, gridspec_kw={'hspace': 0.5})
        self.canvas = Canvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        # Add a second subplot sharing the y-axis
        self.ax2 = self.ax1.twinx()
        #self.ax2.yaxis.set_label_position("right")
        self.ax2.yaxis.tick_right()

        # Add the toolbar after the canvas to place it at the bottom
        self.setLayout(self.vbl)
        self.fig.tight_layout()

    def sizeHint(self):
        return self.canvas.sizeHint()
    #
    def minimumSizeHint(self):
         return QSize(10, 10)