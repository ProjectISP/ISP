from __future__ import unicode_literals

import cartopy.crs as ccrs
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent, PickEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
from obspy import Stream

from isp.Gui import pw, pyc, qt
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


class BasePltPyqtCanvas(FigureCanvas):

    def __init__(self, parent, obj=None, **kwargs):
        """
        Create an embed matplotlib canvas into pyqt.

            Important!! This class is not meant to be used directly only as a parent.
            Instead use  :class:`MatplotlibCanvas` or any of its child classes.

        :param parent: A QWidget to be parent of this canvas.

        :param obj: Expected to be an obspy Stream or a matplotlib figure. Leave as None if you want
            to construct your own matplotlib figure.

        :keyword kwargs: Any valid Matplotlib kwargs for subplots.

        :keyword nrows: default = 1

        :keyword ncols: default = 1

        :keyword sharex: default = all

        :keyword constrained_layout: default = True
        """
        self.button_connection = None
        self.cdi_enter = None
        self.cdi_leave = None
        self.pick_connect = None
        self.axes = None
        self.__callback_on_double_click = None
        self.__callback_on_click = None
        self.__callback_on_pick = None
        self.__cbar = None
        self.pickers = {}

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

        self.register_on_click()  # Register the click event for the canvas
        self.register_on_pick()  # Register the pick events for the draws.

    def __del__(self):
        print("disconnect")
        self.disconnect_click()
        self.disconnect_pick()
        plt.close(self.figure)

    def __construct_subplot(self, **kwargs):

        nrows = kwargs.pop("nrows", 1)
        ncols = kwargs.pop("ncols", 1)
        sharex = kwargs.pop("sharex", "all")
        c_layout = kwargs.pop("constrained_layout", True)

        fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, constrained_layout=c_layout, **kwargs)
        self.__flat_axes()

        return fig

    def __flat_axes(self):
        # make sure axes are always a np.array
        if type(self.axes) is not numpy.ndarray:
            self.axes = numpy.array([self.axes])
        self.axes = self.axes.flatten()

    def register_on_click(self):
        if not self.button_connection:
            self.button_connection = self.mpl_connect('button_press_event', self.__on_click_event)

        if not self.cdi_enter:
            self.cdi_enter = self.mpl_connect('figure_enter_event', self.__figure_enter_event)

        if not self.cdi_leave:
            self.cdi_leave = self.mpl_connect('figure_leave_event', self.__figure_leave_event)

    def __figure_leave_event(self, event):
        """
        Called when mouse leave this figure.

        :param event: 
        :return: 
        """""
        self.clearFocus()

    def __figure_enter_event(self, event):
        """
        Called when mouse enter the figure.

        :param event:
        :return:
        """
        self.setFocusPolicy(qt.ClickFocus)
        self.setFocus()

    def register_on_pick(self):
        if not self.pick_connect:
            self.pick_connect = self.mpl_connect('pick_event', self.__on_pick_event)

    def disconnect_click(self):
        if self.button_connection:
            self.mpl_disconnect(self.button_connection)

        if self.cdi_enter:
            self.mpl_disconnect(self.cdi_enter)

        if self.cdi_leave:
            self.mpl_disconnect(self.cdi_leave)

    def disconnect_pick(self):
        if self.pick_connect:
            self.mpl_disconnect(self.pick_connect)

    def __on_pick_event(self, event: PickEvent):
        if self.__callback_on_pick:
            self.__callback_on_pick(event)

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
        """
        Clear all axes for this object.

        :return:
        """
        for ax in self.axes:
            ax.cla()
            self.draw()

    def set_new_subplot(self, nrows, ncols, **kwargs):
        sharex = kwargs.pop("sharex", "all")
        self.figure.clf()
        plt.close(self.figure)
        self.axes = self.figure.subplots(nrows=nrows, ncols=ncols, sharex=sharex, **kwargs)
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

    def get_ydata(self, ax_index):
        """
        Get y-data at the axe index.

        :param ax_index: The ax index to gte the data from.
        :return: The array or y-data.
        """
        ax = self.get_axe(ax_index)
        return ax.lines[0].get_ydata()

    def get_xdata(self, ax_index):
        """
        Get x-data at the axe index.

        :param ax_index: The ax index to gte the data from.
        :return: The array or x-data.
        """
        ax = self.get_axe(ax_index)
        return ax.lines[0].get_xdata()

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

    def on_pick(self, func):
        """
         Register a callback when pick an artist.

        :param func: The callback function. Expect an event and attached (attached ia a dict of tuple) parameters.
        :return:
        """
        self.__callback_on_pick = func

    def plot(self, x, y, axes_index, **kwargs):
        """
        Implement your own plot.

        :param x: x-axis data
        :param y: y-axis data
        :param axes_index: the index of the axes to plot
        :param kwargs: Any valid matplotlib kwargs for plot.

        :return:
        """
        pass

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


class MatplotlibCanvas(BasePltPyqtCanvas):

    def __init__(self, parent, obj=None, **kwargs):
        """
        Create an embed matplotlib canvas into pyqt.

        :param parent: A QWidget to be parent of this canvas.

        :param obj: Expected to be an obspy Stream or a matplotlib figure. Leave as None if you want
            to construct your own matplotlib figure.

        :keyword kwargs: Any valid Matplotlib kwargs for subplots.

        :keyword nrows: default = 1

        :keyword ncols: default = 1

        :keyword sharex: default = all

        :keyword constrained_layout: default = True
        """
        super().__init__(parent, obj, **kwargs)

    def __plot(self, x, y, ax, clear_plot=True, **kwargs):
        if clear_plot:
            ax.cla()
        artist, = ax.plot(x, y, **kwargs)
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        try:
            # Draw can raise ValueError
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

    def draw_arrow(self, x_pos, axe_index=0, arrow_label="Arrow", draw_arrow=False, amplitude=None, **kwargs):
        """
        Draw an arrow over the a plot. This plot will add a pick event to the line.

        :param x_pos: The position of the arrow
        :param axe_index: The subplot axes index.
        :param arrow_label: The label at the arrow.
        :param draw_arrow: True if you want an arrow, false to draw just a line.
        :param amplitude: (float) The waveform amplitude. If amplitude is given it will plot a dot at the
            x = x_pos, y = amplitude
        :param kwargs: Valid Matplotlib kwargs for plot.
        :return: A line.
        """
        # marker = kwargs.pop("marker", '|')
        # markersize = kwargs.pop("markersize", 1000)
        color = kwargs.pop("color", 'red')
        picker = kwargs.pop("picker", True)

        bbox = dict(boxstyle="round", fc="white")
        ax = self.get_axe(axe_index)
        arrowprops = None
        if draw_arrow:
            arrowprops = dict(facecolor=color, shrink=0.05)
        annotate = ax.annotate(arrow_label, xy=(x_pos, 0), xytext=(0, -30), bbox=bbox, xycoords='data',
                               textcoords='offset points', annotation_clip=True, arrowprops=arrowprops)

        # artist = self.plot(x_pos, 0, axe_index, clear_plot=False, marker=marker, markersize=markersize, color=color,
        #                    **kwargs)
        ymin, ymax = self.get_ylim_from_data(ax, offset=10)
        line = ax.vlines(x_pos, ymin, ymax, color=color, picker=picker, **kwargs)

        point = ax.plot(x_pos, amplitude, marker='o', color="steelblue") if amplitude else [None]

        # Add annotate and point in a dict with a key equal to line signature.
        self.pickers[str(line)] = annotate, point[0]
        self.draw()

        return line

    def remove_arrow(self, line: Line2D):
        """
        Remove arrow line and attached components.

        :param line: The ref of a Line2D.
        :return:
        """
        if line:
            line.remove()
        attached = self.pickers.pop(str(line))  # get the picker
        if attached:
            for item in attached:
                if item:
                    item.remove()
        self.draw()


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


class CartopyCanvas(BasePltPyqtCanvas):

    def __init__(self, parent, **kwargs):
        """
        Create an embed cartopy canvas into pyqt.

        :param parent: A QWidget to be parent of this canvas.

        :keyword kwargs: Any valid Matplotlib kwargs for subplots or Cartopy.

        :keyword nrows: default = 1

        :keyword ncols: default = 1

        :keyword sharex: default = all

        :keyword constrained_layout: default = True

        :keyword projection: default =  ccrs.PlateCarree()
        """
        proj = kwargs.pop("projection", ccrs.PlateCarree())
        super().__init__(parent, subplot_kw=dict(projection=proj), **kwargs)

    def plot(self, x, y, axes_index, clear_plot=True, **kwargs):
        """
        Cartopy plot.

        :param x:
        :param y:
        :param axes_index:
        :param clear_plot:
        :param kwargs:
        :return:
        """
        # TODO implement a useful plot for cartopy this is just a test.
        self.clear()
        ax = self.get_axe(axes_index)
        ax.stock_img()
        self.draw()
