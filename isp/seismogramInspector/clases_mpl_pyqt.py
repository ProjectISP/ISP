import sys

from isp.Gui import pw

sys.path.append(r"/Users/robertocabieces/Documents/obs_array")

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar


class MplCanvas_special(FigureCanvas):
    def __init__(self):
        # self.fig = Figure(facecolor = "0.94")
        self.fig = Figure()

        self.ax1 = self.fig.add_subplot(211)

        self.ax2 = self.fig.add_subplot(212)
        self.ax3 = self.ax1.twinx()
        ##If we want to join x axis
        self.ax2.get_shared_x_axes().join(self.ax1, self.ax2)

        ## if we want to show the colour bar
        self.cax = self.fig.add_axes([0.94, 0.11, 0.025, 0.35])

        FigureCanvas.__init__(self, self.fig)


class MatplotlibWidget_special(pw.QWidget):
    def __init__(self, parent=None):
        pw.QWidget.__init__(self, parent)
        self.canvas = MplCanvas_special()
        self.vbl = pw.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class MplCanvas_1(FigureCanvas):
    def __init__(self):
        # self.fig = Figure(facecolor = "0.94")
        self.fig = Figure()

        self.ax1 = self.fig.add_subplot(111)

        ## if we want to show the colour bar
        self.cax = self.fig.add_axes([0.93, 0.11, 0.02, 0.7])

        FigureCanvas.__init__(self, self.fig)


class MatplotlibWidget_1(pw.QWidget):
    def __init__(self, parent=None):
        pw.QWidget.__init__(self, parent)
        self.canvas = MplCanvas_1()
        self.vbl = pw.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class MplCanvas_2(FigureCanvas):
    def __init__(self):
        # self.fig = Figure(facecolor = "0.94")
        self.fig = Figure()

        self.ax1 = self.fig.add_subplot(211)

        self.ax2 = self.fig.add_subplot(212)

        ## if we want to show the colour bar
        # self.cax = self.fig.add_axes([0.93, 0.11, 0.02, 0.7])

        FigureCanvas.__init__(self, self.fig)


class MatplotlibWidget_2(pw.QWidget):
    def __init__(self, parent=None):
        pw.QWidget.__init__(self, parent)
        self.canvas = MplCanvas_2()
        self.vbl = pw.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class MplCanvas_3(FigureCanvas):
    def __init__(self):
        # self.fig = Figure(facecolor = "0.94")
        self.fig = Figure()

        self.ax1 = self.fig.add_subplot(311)

        self.ax2 = self.fig.add_subplot(312)

        self.ax3 = self.fig.add_subplot(313)
        self.ax1.get_shared_x_axes().join(self.ax1, self.ax2, self.ax3)
        ## if we want to show the colour bar
        self.cax = self.fig.add_axes([0.93, 0.11, 0.02, 0.7])

        FigureCanvas.__init__(self, self.fig)


class MatplotlibWidget_3(pw.QWidget):
    def __init__(self, parent=None):
        pw.QWidget.__init__(self, parent)
        self.canvas = MplCanvas_3()
        self.vbl = pw.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)