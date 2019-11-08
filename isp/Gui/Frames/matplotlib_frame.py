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
from matplotlib.axes import Axes
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

    def __init__(self, parent, obj=None, **kwargs):
        """
        Create an embed matplotlib canvas into pyqt.

        :param parent: A QWidget to be parent of this canvas.

        :param obj: Expected to be an obspy Stream or a matplotlib figure. Leave as None if you want
            to construct your own matplotlib figure.

        :param kwargs: If obj is an obspy.Stream you can use valid kwargs for Stream objects. Otherwise, the
            valid kwargs are "nrows" and "ncols" for the subplots.
        """
        self.button_connection = None
        self.axes = None
        self.__callback_on_double_click = None
        self.__callback_on_click = None

        if not obj:
            fig = self.__construct_subplot(**kwargs)
        else:
            if isinstance(obj, Stream):
                fig = ObspyUtil.get_figure_from_stream(obj, **kwargs)
            else:
                fig = obj

        super().__init__(fig)

        if parent and (isinstance(parent, pw.QWidget) or isinstance(parent, pw.QFrame)):
            if parent.layout() is not None:
                layout = parent.layout()
                for child in parent.findChildren(MatplotlibWidget):
                    child.setParent(None)
            else:
                layout = pw.QVBoxLayout(parent)

            mpw = MatplotlibWidget(parent, self)
            layout.addWidget(mpw)

        FigureCanvas.setSizePolicy(self,
                                   pw.QSizePolicy.Expanding,
                                   pw.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.register_on_click()

    def __del__(self):
        print("disconnect")
        self.disconnect_click()

    def __construct_subplot(self, **kwargs):
        nrows = kwargs.get("nrows") if "nrows" in kwargs.keys() else 1
        ncols = kwargs.get("ncols") if "ncols" in kwargs.keys() else 1

        fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all')
        # make sure axes are always a np.array
        if type(self.axes) is not numpy.ndarray:
            self.axes = numpy.array([self.axes])
        self.axes = self.axes.flatten()

        return fig

    def register_on_click(self):
        if not self.button_connection:
            self.button_connection = self.mpl_connect('button_press_event', self.__on_click)

    def disconnect_click(self):
        if self.button_connection:
            self.mpl_disconnect(self.button_connection)

    def __on_click(self, event):
        if event.dblclick and event.button == 1:
            self.__callback_on_double_click(event, self)

    def get_axe(self, index) -> Axes:
        return self.axes.item(index)

    def set_new_subplot(self, nrows, ncols):
        self.figure = self.__construct_subplot(nrows=nrows, ncols=ncols)

    def set_xlabel(self, axe_index, value):
        ax = self.get_axe(axe_index)
        if ax:
            ax.set_xlabel(value)

    def on_double_click(self, func):
        self.__callback_on_double_click = func

    def onclick(self, func):
        self.__callback_on_click = func

    def __plot(self, x, y, ax, clear_plot=True, **kwargs):
        if clear_plot:
            ax.cla()
        artist, = ax.plot(x, y, **kwargs)
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        try:
            # Draw can raise draw error
            self.draw()
            return artist
        except ValueError:
            artist.remove()
            return None

    def plot(self, x, y, axes_index, clear_plot=True, **kwargs):
        if self.axes is not None:
            ax = self.get_axe(axes_index)
            return self.__plot(x, y, ax, clear_plot=clear_plot, **kwargs)

    def plot_contour(self, x, y, z, axes_index, clear_plot=True, **kwargs):
        if self.axes is not None:
            ax = self.get_axe(axes_index)
            cmap = kwargs.pop('cmap', plt.get_cmap('jet'))
            x_label = ax.get_xlabel()
            if clear_plot:
                ax.cla()
            ax.contourf(x, y, z, 100, cmap=cmap, **kwargs)
            ax.set_xlim(*self.get_xlim_from_data(ax, 0))
            ax.set_ylim(*ax.get_ylim())
            if x_label is not None and len(x_label) != 0:
                self.set_xlabel(1, x_label)
        self.draw()

    @staticmethod
    def get_xlim_from_data(ax: Axes, offset=5):
        """
        Compute the limit of the x-axis from the data with a default offset of 5%
        :param ax: The matplotlib axes.
        :param offset: Add an offset to the limit in %.
        :return: A tuple of (x_min, x_max).
        """
        x_max = ax.dataLim.xmax
        x_min = ax.dataLim.xmin - x_max * offset * 0.01
        x_max += x_max * offset * 0.01
        return x_min, x_max

    @staticmethod
    def get_ylim_from_data(ax: Axes, offset=5):
        """
        Compute the limit of the y-axis from the data with a default offset of 5%
        :param ax: The matplotlib axes.
        :param offset: Add an offset to the limit in %.
        :return: A tuple of (y_min, y_max).
        """
        y_max = ax.dataLim.ymax
        y_min = ax.dataLim.ymin - y_max * offset * 0.01
        y_max += y_max * offset * 0.01
        return y_min, y_max

    def draw_arrow(self, x_pos, axe_index=0, arrow_label="Arrow", draw_arrow=False, **kwargs):

        marker = kwargs.pop("marker", '|')
        markersize = kwargs.pop("markersize", 1000)
        color = kwargs.pop("color", 'red')

        bbox = dict(boxstyle="round", fc="white")
        ax = self.axes.item(axe_index)
        arrowprops = None
        if draw_arrow:
            arrowprops = dict(facecolor=color, shrink=0.05)
        annotate = ax.annotate(arrow_label, xy=(x_pos, 0), xytext=(0, -50), bbox=bbox, xycoords='data',
                               textcoords='offset points', annotation_clip=True, arrowprops=arrowprops)

        artist = self.plot(x_pos, 0, 0, clear_plot=False, marker=marker, markersize=markersize, color=color, **kwargs)
        if artist is None:
            print("Error")
            if annotate:
                annotate.remove()
            x_pos += x_pos*0.0001
            self.draw_arrow(x_pos, axe_index=axe_index, arrow_label=arrow_label, marker=marker,


                            markersize=markersize, color=color, **kwargs)


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

