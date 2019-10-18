# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale
#               2015 Jens H Nielsen
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

from __future__ import unicode_literals

# Make sure that we are using QT5
import numpy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from obspy import Stream

from isp.Gui import pw, pyc
from isp.Gui.Frames import BaseFrame
from isp.Utils import ObspyUtil


class MatplotlibWidget(pw.QWidget):

    def __init__(self, parent=None, canvas=None):
        super().__init__(parent)
        self.canvas = canvas
        self.vbl = pw.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class MatplotlibCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, obj=None, **kwargs):
        """
        Create a embed matplotlib canvas into pyqt.

        :param parent: A QWidget to be parent of this canvas.

        :param obj: Expected to be a obspy Stream or a matplotlib figure.
        """
        self.button_connection = None
        self.axes = None
        self.callback_on_double_click = None
        self.callback_on_click = None

        if not obj:
            fig = self.__construct_subplot(**kwargs)
        else:
            if isinstance(obj, Stream):
                fig = ObspyUtil.get_figure_from_stream(obj, **kwargs)
            else:
                fig = obj

        super().__init__(fig)

        if parent and isinstance(parent, pw.QWidget):
            if parent.layout():
                parent.layout().itemAt(0).widget().setParent(None)
            else:
                self.layout = pw.QVBoxLayout(parent)
            self.mpw = MatplotlibWidget(parent, self)
            self.layout.addWidget(self.mpw)


        FigureCanvas.setSizePolicy(self,
                                   pw.QSizePolicy.Expanding,
                                   pw.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def __del__(self):
        print("disconnect")
        self.mpl_disconnect(self.button_connection)

    def __construct_subplot(self, **kwargs):
        nrows = kwargs.get("nrows") if "nrows" in kwargs.keys() else 1
        ncols = kwargs.get("ncols") if "ncols" in kwargs.keys() else 1

        fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True)
        # make sure axes are always a np.array
        if type(self.axes) is not numpy.ndarray:
            self.axes = numpy.array([self.axes])
        self.axes = self.axes.flatten()

        return fig

    def register_on_click(self):
        if not self.button_connection:
            self.button_connection = self.mpl_connect('button_press_event', self.__on_click)

    def __on_click(self, event):
        if event.dblclick:
            self.callback_on_double_click(event, self)

    def set_new_subplot(self, nrows, ncols):
        self.figure = self.__construct_subplot(nrows=nrows, ncols=ncols)

    def on_double_click(self, func):
        self.callback_on_double_click = func

    def onclick(self, func):
        self.callback_on_click = func

    def __plot(self, x, y, ax, clear_plot=True, **kwargs):
        if clear_plot:
            ax.cla()
        ax.plot(x, y, **kwargs)
        self.draw()

    def plot(self, x, y, axes_index, clear_plot=True, **kwargs):
        if self.axes is not None:
            ax = self.axes.item(axes_index)
            self.__plot(x, y, ax, clear_plot=clear_plot, **kwargs)

    def draw_arrow(self, x_pos, axe_index=0, arrow_label="Arrow", **kwargs):
        bbox = dict(boxstyle="round", fc="white")
        ax = self.axes.item(axe_index)
        offset = ax.get_ylim()[0]
        ax.annotate(arrow_label, xy=(x_pos, 0), xytext=(x_pos, offset), bbox=bbox,
                    arrowprops=dict(facecolor='red', shrink=0.05))

        self.plot(x_pos, 0, 0, clear_plot=False, marker='|', markersize=1000, color='red')


class MatplotlibFrame(BaseFrame):
    def __init__(self, obj, **kwargs):
        """
        Embed a figure from matplotlib into a pyqt canvas.

        :param obj: Expected to be a obspy Stream or a matplotlib figure.
        """
        super().__init__()
        self.setAttribute(pyc.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Matplotlib Window")

        self.file_menu = pw.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 pyc.Qt.CTRL + pyc.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = pw.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = pw.QWidget(self)

        self.layout = pw.QVBoxLayout(self.main_widget)
        self.mpc = MatplotlibCanvas(self.main_widget, obj, **kwargs)
        self.mpw = MatplotlibWidget(self.main_widget, self.mpc)
        self.layout.addWidget(self.mpw)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("Done!", 2000)

    def set_canvas(self, mpc: MatplotlibCanvas):
        self.mpc = mpc
        self.layout.removeWidget(self.mpw)
        self.mpw = MatplotlibWidget(self.main_widget, self.mpc)
        self.layout.addWidget(self.mpw)

    def fileQuit(self):
        self.close()
        self.mpc = None
        self.mpw = None

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        pw.QMessageBox.about(self, "About",
                                    """
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen

This program is a Qt5 application embedding matplotlib
canvases and Obspy stream.""")

