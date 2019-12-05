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

import time

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from obspy import Stream

from isp.Gui import pw, pyc
from isp.Gui.Frames import BaseFrame
from isp.Utils import ObspyUtil, AsycTime


# Make sure that we are using QT5


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
        self.__cbar = None

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

        self.__register_on_click()

    def __del__(self):
        print("disconnect")
        self.disconnect_click()

    def __construct_subplot(self, **kwargs):
        nrows = kwargs.get("nrows") if "nrows" in kwargs.keys() else 1
        ncols = kwargs.get("ncols") if "ncols" in kwargs.keys() else 1

        fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', constrained_layout=True)
        self.__flat_axes()

        return fig

    def __flat_axes(self):
        # make sure axes are always a np.array
        if type(self.axes) is not numpy.ndarray:
            self.axes = numpy.array([self.axes])
        self.axes = self.axes.flatten()

    def __register_on_click(self):
        if not self.button_connection:
            self.button_connection = self.mpl_connect('button_press_event', self.__on_click_event)

    def disconnect_click(self):
        if self.button_connection:
            self.mpl_disconnect(self.button_connection)

    def __on_click_event(self, event: MouseEvent):
        self.is_dlb_click = False
        if event.dblclick and event.button == MouseButton.LEFT:
            # On double click with left button.
            self.is_dlb_click = True
            if self.__callback_on_double_click:
                self.__callback_on_double_click(event, self)

        elif not event.dblclick and event.button == MouseButton.LEFT:
            self.__on_click(event)

    @AsycTime.async_wait(0.5)
    def __on_click(self, event: MouseEvent):
        if not self.is_dlb_click and event.button == MouseButton.LEFT:
            print("Click")
            if self.__callback_on_click:
                self.__callback_on_click(event, self)

    def get_axe(self, index) -> Axes:
        """
        Get a matplotlib Axes of a subplot.

        :param index: The axe index.
        :return: A matplotlib Axes.
        """
        return self.axes.item(index)

    def clear(self):
        for ax in self.axes:
            ax.cla()
            self.draw()

    def set_new_subplot(self, nrows, ncols):
        self.figure.clf()
        plt.close(self.figure)
        self.axes = self.figure.subplots(nrows=nrows, ncols=ncols, sharex='all')
        self.__flat_axes()
        self.draw()

    def set_xlabel(self, axe_index, value):
        ax = self.get_axe(axe_index)
        if ax:
            ax.set_xlabel(value)
            self.draw()  # force to update label

    def set_ylabel(self, axe_index, value):
        ax = self.get_axe(axe_index)
        if ax:
            ax.set_ylabel(value)
            self.draw()  # force to update label

    def on_double_click(self, func):
        """
        Register a callback when double click the matplotlib canvas.

        :param func: The callback function. Expect an event and canvas parameters.
        :return:
        """
        self.__callback_on_double_click = func

    def on_click(self, func):
        """
        Register a callback when click the matplotlib canvas.

        :param func: The callback function. Expect an event and canvas parameters.
        :return:
        """
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
        """
        Wrapper for matplotlib plot.

        :param x: x-axis data.
        :param y: y-axis data.
        :param axes_index: The subplot axes index.
        :param clear_plot: True to clean plot, False to plot over.
        :param kwargs: Valid Matplotlib kwargs for plot.
        :return:
        """
        if self.axes is not None:
            ax = self.get_axe(axes_index)
            return self.__plot(x, y, ax, clear_plot=clear_plot, **kwargs)

    def plot_contour(self, x, y, z, axes_index, clear_plot=True, show_colorbar=True, **kwargs):
        """
        Wrapper for matplotlib contourf.

        :param x: x-axis data.
        :param y: y-axis data.
        :param z: z-axis data.
        :param axes_index: The subplot axes index.
        :param clear_plot: True to clean plot, False to plot over.
        :param show_colorbar: True to show colorbar, false otherwise.
        :param kwargs: Valid Matplotlib kwargs for contourf.
        :return:
        """
        if self.axes is not None:
            ax = self.get_axe(axes_index)
            cmap = kwargs.pop('cmap', plt.get_cmap('jet'))
            levels = kwargs.pop('levels', 100)
            vmin = kwargs.pop('vmin', numpy.amin(z))
            vmax = kwargs.pop('vmax', numpy.amax(z))
            clabel = kwargs.pop('clabel', '')
            x_label = ax.get_xlabel()
            if clear_plot:
                ax.cla()
            cs = ax.contourf(x, y, z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
            cs.set_clim(vmin, vmax)
            self.clear_color_bar()
            if show_colorbar:
                self.__cbar: Colorbar = self.figure.colorbar(cs, ax=ax, extend='both', pad=0.0)
                self.__cbar.ax.set_ylabel(clabel)
            ax.set_xlim(*self.get_xlim_from_data(ax, 0))
            ax.set_ylim(*self.get_ylim_from_data(ax, 0))
            if x_label is not None and len(x_label) != 0:
                self.set_xlabel(1, x_label)
        self.draw()

    def clear_color_bar(self):
        if self.__cbar:
            self.__cbar.remove()

    @staticmethod
    def get_xlim_from_data(ax: Axes, offset=5):
        """
        Compute the limit of the x-axis from the data with a default offset of 5%.

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
        Compute the limit of the y-axis from the data with a default offset of 5%.

        :param ax: The matplotlib axes.
        :param offset: Add an offset to the limit in %.
        :return: A tuple of (y_min, y_max).
        """
        y_max = ax.dataLim.ymax
        y_min = ax.dataLim.ymin - y_max * offset * 0.01
        y_max += y_max * offset * 0.01
        return y_min, y_max

    def draw_arrow(self, x_pos, axe_index=0, arrow_label="Arrow", draw_arrow=False, **kwargs):
        """
        Draw an arrow over the a plot.

        :param x_pos: The position of the arrow
        :param axe_index: The subplot axes index.
        :param arrow_label: The label at the arrow.
        :param draw_arrow: True if you want an arrow, false to draw just a line.
        :param kwargs: Valid Matplotlib kwargs for plot.
        :return:
        """

        marker = kwargs.pop("marker", '|')
        markersize = kwargs.pop("markersize", 1000)
        color = kwargs.pop("color", 'red')

        bbox = dict(boxstyle="round", fc="white")
        ax = self.axes.item(axe_index)
        arrowprops = None
        if draw_arrow:
            arrowprops = dict(facecolor=color, shrink=0.05)
        annotate = ax.annotate(arrow_label, xy=(x_pos, 0), xytext=(0, -30), bbox=bbox, xycoords='data',
                               textcoords='offset points', annotation_clip=True, arrowprops=arrowprops)

        artist = self.plot(x_pos, 0, axe_index, clear_plot=False, marker=marker, markersize=markersize, color=color,
                           **kwargs)
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

