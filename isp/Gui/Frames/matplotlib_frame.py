from __future__ import unicode_literals

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy
import shapely.geometry as sgeom
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent, PickEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
from matplotlib.patheffects import Stroke
from obspy import Stream
from owslib.wms import WebMapService

from isp.Gui import pw, pyc, qt
from isp.Utils import ObspyUtil, AsycTime


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
            self.draw_idle()

    def update_bounds(self):
        """
        Update the bounds of all axes

        :return:
        """
        for ax in self.axes:
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
        self.draw_idle()

    def is_within_xlim(self, x: float, ax_index: int):
        """
        Check whether or not the value x is within the bounds of x_min and x_max of the axe.

        :param x: The value at x-axis.
        :param ax_index: The index of the axe to test the bounds.
        :return: True if within the x-bounds, False otherwise.
        """

        ax = self.get_axe(ax_index)
        x_min, x_max = ax.get_xbound()
        return x_min <= x <= x_max

    def set_new_subplot(self, nrows, ncols, **kwargs):
        sharex = kwargs.pop("sharex", "all")
        self.figure.clf()
        plt.close(self.figure)
        nrows = max(nrows, 1)  # avoid zero rows.
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

    def set_plot_label(self, ax: Axes or int, text: str):
        """
        Sets an label box at the upper right corner of the axes.

        :param ax: The axes or axes_index to add the annotation.
        :param text: The text
        :return:
        """
        if type(ax) == int:
            ax = self.get_axe(ax)
        bbox = dict(boxstyle="round", fc="white")
        return ax.annotate(text, xy=(1, 1), xycoords='axes fraction', xytext=(-20, -20), textcoords='offset points',
                           ha="right", va="top", bbox=bbox)

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

    @staticmethod
    def set_yaxis_color(ax: Axes, color: str, is_left=False):
        """
        Set the color of the y-axis for the given axe.

        :param ax: The matplotlib Axes
        :param color: The color is should be, i.e: 'red', 'blue', 'green', etc..
        :param is_left: If True it will change the left side of the y-axis, otherwise it will change the right side.
        :return:
        """
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis='y', colors=color)
        if is_left:
            ax.spines.get('left').set_color(color)
        else:
            ax.spines.get('right').set_color(color)


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
        self.__cbar = None
        self.__twinx_axes = {}

    def __add_twinx_ax(self, ax_index):
        ax = self.get_axe(ax_index)
        tw_ax = self.get_twinx_ax(ax_index)
        if not tw_ax:
            tw_ax = ax.twinx()
            tw_ax.spines.get('left').set_visible(False)
            self.__twinx_axes[ax_index] = tw_ax
        return tw_ax

    def get_twinx_ax(self, ax_index) -> Axes:
        return self.__twinx_axes.get(ax_index, None)

    def set_ylabel_twinx(self, axe_index, value):
        ax = self.get_twinx_ax(axe_index)
        if ax:
            ax.set_ylabel(value)
            self.draw_idle()  # force to update label

    def __plot(self, x, y, ax, clear_plot=True, **kwargs):
        if clear_plot:
            ax.cla()
        artist, = ax.plot(x, y, **kwargs)
        try:
            # Draw can raise ValueError
            self.draw_idle()
            return artist
        except ValueError:
            artist.remove()
            return None
    ######
    def __plot_date(self, x, y, ax, clear_plot=True, **kwargs):
        if clear_plot:
            ax.cla()
        artist, = ax.plot_date(x, y, **kwargs)
        try:
            # Draw can raise ValueError
            self.draw_idle()
            return artist
        except ValueError:
            artist.remove()
            return None
    #######
    def __plot_3d(self, x, y, z, ax, plot_type, clear_plot=True, show_colorbar=True, **kwargs):
        """
        Wrapper for matplotlib 3d plots.

        :param x: x-axis data.
        :param y: y-axis data.
        :param z: z-axis data.
        :param ax: The subplot ax.
        :param plot_type: The plot type, either contourf or scatter.
        :param clear_plot: True to clean plot, False to plot over.
        :param show_colorbar: True to show colorbar, false otherwise.
        :param kwargs: Valid Matplotlib kwargs for plot_type.
        :return:
        """
        if clear_plot:
            ax.cla()

        cmap = kwargs.pop('cmap', plt.get_cmap('jet'))
        vmin = kwargs.pop('vmin', numpy.amin(z))
        vmax = kwargs.pop('vmax', numpy.amax(z))
        clabel = kwargs.pop('clabel', '')

        x_label = ax.get_xlabel()

        if plot_type == "contourf":
            levels = kwargs.pop('levels', 100)
            cs = ax.contourf(x, y, z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        elif plot_type == "scatter":
            area = 10.*z**2  # points size from 0 to 5
            cs = ax.scatter(x, y, s=area, c=z, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax, marker=".", **kwargs)

        else:
            raise ValueError("Invalid value for plot_type it must be equal to either contourf or scatter.")

        cs.set_clim(vmin, vmax)

        self.clear_color_bar()
        if show_colorbar:
            self.__cbar: Colorbar = self.figure.colorbar(cs, ax=ax, extend='both', pad=0.0)
            self.__cbar.ax.set_ylabel(clabel)
        ax.set_xlim(*self.get_xlim_from_data(ax, 0))
        ax.set_ylim(*self.get_ylim_from_data(ax, 0))
        if x_label is not None and len(x_label) != 0:
            self.set_xlabel(1, x_label)
        self.draw_idle()

    def plot(self, x, y, axes_index, clear_plot=True, is_twinx=False, **kwargs):
        """
        Wrapper for matplotlib plot.

        Import: If the kwarg is_twinx=True, the kwarg clear_plot has no effect and will be always set to True.

        :param x: x-axis data.
        :param y: y-axis data.
        :param axes_index: The subplot axes index.
        :param clear_plot: True to clean plot, False to plot over. Default=True.
        :param is_twinx: True if you want to add a new y-axis scale, False otherwise. Default=False.
        :param kwargs: Valid Matplotlib kwargs for plot.
        :return: The artist plotted.
        """
        if self.axes is not None:
            ax = self.get_axe(axes_index)
            if is_twinx:
                tw_ax = self.__add_twinx_ax(axes_index)
                artist = self.__plot(x, y, tw_ax, clear_plot=True, **kwargs)
                if artist:
                    self.set_yaxis_color(tw_ax, artist.get_color())
                return artist
            else:
                return self.__plot(x, y, ax, clear_plot=clear_plot, **kwargs)

    ############
    def plot_date(self, x, y, axes_index, clear_plot=True, is_twinx=False, **kwargs):
        """
        Wrapper for matplotlib plot.

        Import: If the kwarg is_twinx=True, the kwarg clear_plot has no effect and will be always set to True.

        :param x: x-axis data.
        :param y: y-axis data.
        :param axes_index: The subplot axes index.
        :param clear_plot: True to clean plot, False to plot over. Default=True.
        :param is_twinx: True if you want to add a new y-axis scale, False otherwise. Default=False.
        :param kwargs: Valid Matplotlib kwargs for plot.
        :return: The artist plotted.
        """
        if self.axes is not None:
            ax = self.get_axe(axes_index)
            if is_twinx:
                tw_ax = self.__add_twinx_ax(axes_index)
                artist = self.__plot_date(x, y, tw_ax, clear_plot=True, **kwargs)
                if artist:
                    self.set_yaxis_color(tw_ax, artist.get_color())
                return artist
            else:
                return self.__plot_date(x, y, ax, clear_plot=clear_plot, **kwargs)
    ############

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
            self.__plot_3d(x, y, z, ax, "contourf", clear_plot=clear_plot, show_colorbar=show_colorbar, **kwargs)


    def plot_projection(self, x, y, z, axes_index, clear_plot=True, **kwargs):
        """
        Wrapper for matplotlib scatter3d.

        :param x: x-axis data.
        :param y: y-axis data.
        :param z: z-axis data.
        :param axes_index: The subplot axes index.
        :param clear_plot: True to clean plot, False to plot over.
        :param show_colorbar: True to show colorbar, false otherwise.
        :param kwargs: Valid Matplotlib kwargs for scatter.
        :return:
        """

        if self.axes is not None:
           ax = self.get_axe(axes_index)
           if clear_plot:
                ax.cla()

           ax = self.figure.gca(projection='3d')
           ax.plot(x, y, z, **kwargs)




    def scatter3d(self, x, y, z, axes_index, clear_plot=True, show_colorbar=True, **kwargs):
        """
        Wrapper for matplotlib scatter3d.

        :param x: x-axis data.
        :param y: y-axis data.
        :param z: z-axis data.
        :param axes_index: The subplot axes index.
        :param clear_plot: True to clean plot, False to plot over.
        :param show_colorbar: True to show colorbar, false otherwise.
        :param kwargs: Valid Matplotlib kwargs for scatter.
        :return:
        """
        if self.axes is not None:
            ax = self.get_axe(axes_index)
            self.__plot_3d(x, y, z, ax, "scatter", clear_plot=clear_plot, show_colorbar=show_colorbar, **kwargs)

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
            x = x_pos, y = amplitude.
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

        self.update_bounds()
        ymin, ymax = ax.get_ybound()
        # plot arrows only if x_pos is within the x bounds. Avoid warnings from matplotlib.
        if self.is_within_xlim(x_pos, axe_index):
            annotate = ax.annotate(arrow_label, xy=(x_pos, 0), xytext=(0, -30), bbox=bbox, xycoords='data',
                                   textcoords='offset points', annotation_clip=True, arrowprops=arrowprops)

            line = ax.vlines(x_pos, ymin, ymax, color=color, picker=picker, **kwargs)

            point = ax.plot(x_pos, amplitude, marker='o', color="steelblue") if amplitude else [None]
            # Add annotate and point in a dict with a key equal to line signature.

            self.pickers[str(line)] = annotate, point[0]
            self.draw_idle()

            return line

    def remove_arrow(self, line: Line2D):
        """
        Remove arrow line and attached components.

        :param line: The ref of a Line2D.

        :return:
        """

        if line:
            try:
                line.remove()
            except ValueError as error:
                print(error)
            attached = self.pickers.pop(str(line), None)  # get the picker
            if attached:
                for item in attached:
                    if item:
                        item.remove()
                        del item
            del line
            self.draw_idle()

    def remove_arrows(self, lines: [Line2D]):
        for line in lines:
            self.remove_arrow(line)


class MatplotlibFrame(pw.QMainWindow):
    def __init__(self, obj, **kwargs):
        """
        Embed a figure from matplotlib into a pyqt canvas.

        :param obj: Expected to be a obspy Stream or a matplotlib figure.
        """
        super().__init__()
        # self.setAttribute(pyc.Qt.WA_DeleteOnClose)
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

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("Done!", 2000)

        # used as a callback for closing event in this window.
        self.__close_window_callback = None

    def connect_close(self, func):
        self.__close_window_callback = func

    def set_canvas(self, mpc: MatplotlibCanvas):
        self.mpc = mpc
        self.layout.removeWidget(self.mpw)

    def fileQuit(self):
        self.close()
        self.mpc = None

    def close(self) -> bool:
        if self.__close_window_callback:
            self.__close_window_callback()
        return True

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        pw.QMessageBox.about(self, "About",
                             """ 
                             Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen 
                             
                             This program is a Qt5 application embedding matplotlib 
                             canvases and Obspy stream.""")


class CartopyCanvas(BasePltPyqtCanvas):

    #MAP_SERVICE_URL = 'https://gis.ngdc.noaa.gov/arcgis/services/gebco08_hillshade/MapServer/WMSServer'
    MAP_SERVICE_URL = 'https://www.gebco.net/data_and_products/gebco_web_services/2019/mapserv?'
    #MAP_SERVICE_URL = 'https://gis.ngdc.noaa.gov/arcgis/services/etopo1/MapServer/WMSServer'
    def __init__(self, parent, **kwargs):
        """
        Create an embed cartopy canvas into pyqt.

        :param parent: A QWidget to be parent of this canvas.

        :keyword kwargs: Any valid Matplotlib kwargs for subplots or Cartopy.

        :keyword nrows: default = 1

        :keyword ncols: default = 1

        :keyword sharex: default = all

        :keyword constrained_layout: default = False

        :keyword projection: default =  ccrs.PlateCarree()
        """

        proj = kwargs.pop("projection", ccrs.PlateCarree())
        c_layout = kwargs.pop("constrained_layout", False)
        super().__init__(parent, subplot_kw=dict(projection=proj), constrained_layout=c_layout, **kwargs)

    def plot_map(self, x, y, scatter_x, scatter_y, scatter_z, axes_index, clear_plot=True, resolution = "low", **kwargs):
        """
        Cartopy plot.

        :param x:
        :param y:
        :param scatter_x:
        :param scatter_y:
        :param scatter_z:
        :param axes_index:
        :param clear_plot:
        :param kwargs:
        :return:
        """

        # TODO implement a useful plot for cartopy this is just a test.
        self.clear()
        ax = self.get_axe(axes_index)
        fig = ax.get_figure()
        wms = WebMapService(self.MAP_SERVICE_URL)
        geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
        #layer = 'GEBCO_08 Hillshade'
        layer ='GEBCO_2019_Grid'
        #layer = 'shaded_relief'
        xmin = int(x-8)
        xmax = int(x+8)
        ymin = int(y-8)
        ymax = int(y+8)
        extent = [xmin, xmax, ymin, ymax]

        ax.set_extent(extent, crs=ccrs.PlateCarree())



        if resolution is "high":

            ax.add_wms(wms, layer)

        else:

            coastline_10m = cartopy.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                edgecolor='k', alpha=0.6, linewidth=0.5, facecolor=cartopy.feature.COLORS['land'])
            ax.stock_img()
            ax.add_feature(coastline_10m)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.2, color='gray', alpha=0.2, linestyle='-')

        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = False
        gl.ylines = False

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.plot(x, y, color='red', marker='*',markersize=3)
        ax.scatter(scatter_x, scatter_y, s=10, c=scatter_z/10, marker=".", alpha=0.3,cmap=plt.get_cmap('YlOrBr'))

        # Create an inset GeoAxes showing the Global location
        sub_ax = fig.add_axes([0.70, 0.75, 0.28, 0.28],
                              projection=ccrs.PlateCarree())
        sub_ax.set_extent([-180, 180, -90, 90], geodetic)

        # Make a nice border around the inset axes.
        effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)
        sub_ax.outline_patch.set_path_effects([effect])

        # Add the land, coastlines and the extent of the Solomon Islands.
        sub_ax.add_feature(cfeature.LAND)
        sub_ax.coastlines()
        extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
        sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='none',
                              edgecolor='blue', linewidth=1.0)

        self.draw()

    def plot_stations(self, x, y, depth, axes_index, show_colorbar=True, clear_plot=True, **kwargs):
        self.clear()
        ax = self.get_axe(axes_index)
        # print(self.MAP_SERVICE_URL)
        wms = WebMapService(self.MAP_SERVICE_URL)
        layer = 'GEBCO_08 Hillshade'
        #xmin = int(min(x) - 5)
        #xmax = int(max(x) + 5)
        #ymin = int(min(y) - 5)
        #ymax = int(max(y) + 5)
        xmin = -150
        xmax = -140
        ymin = 60
        ymax = 70
        extent = [xmin, xmax, ymin, ymax]

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        coastline_10m = cartopy.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                                                            edgecolor='k', alpha=0.6, linewidth=0.5,
                                                            facecolor=cartopy.feature.COLORS['land'])
        ax.add_feature(coastline_10m)
        ax.add_wms(wms, layer)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.2, color='gray', alpha=0.2, linestyle='-')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = False
        gl.ylines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        cmap = kwargs.pop('cmap', plt.get_cmap('jet'))
        vmin = kwargs.pop('vmin', numpy.amin(depth))
        vmax = kwargs.pop('vmax', numpy.amax(depth))
        print(x)
        print(y)
        cs = ax.scatter(x, y, s=10, c=depth / 10, marker="^", cmap=plt.get_cmap('YlOrBr'))
        cs.set_clim(vmin, vmax)
        if show_colorbar:
            self.__cbar: Colorbar = self.figure.colorbar(cs, ax=ax, orientation='horizontal', fraction=0.05,
                                                         extend='both', pad=0.15)
            self.__cbar.ax.set_ylabel("Depth [m]")
        self.draw()

class FocCanvas(BasePltPyqtCanvas):

    def __init__(self, parent, **kwargs):
        """
        Create an embed cartopy canvas into pyqt.

        :param parent: A QWidget to be parent of this canvas.

        :keyword kwargs: Any valid Matplotlib kwargs for subplots or Cartopy.

        :keyword nrows: default = 1

        :keyword ncols: default = 1

        :keyword sharex: default = all

        :keyword constrained_layout: default = False

        :keyword projection: default =  ccrs.PlateCarree()
        """

        c_layout = kwargs.pop("constrained_layout", False)
        super().__init__(parent,constrained_layout=c_layout, **kwargs)

    def drawFocMec(self, strike, dip, rake, sta, az, inc,pol, axes_index):
        from obspy.imaging.beachball import beach
        import numpy as np
        azims = []
        incis = []
        polarities = []
        bbox = dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=1.5, alpha=0.7)
        self.clear()
        ax = self.get_axe(axes_index)
        beach2 = beach([strike, dip, rake], facecolor='r', linewidth=1., alpha=0.3, width=2)
        ax.add_collection(beach2)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        N= len(sta)
        for j in range(N):
            station = sta[j]
            azim = az[j]
            inci = inc[j]
            polarity = str(pol[j])
            polarity = polarity[0]
            if inci > 90:
                inci = 180. - inci
                azim = -180. + azim
            plotazim = (np.pi / 2.) - ((azim / 180.) * np.pi)
            azims.append(plotazim)
            incis.append(inci)
            x = (inci * np.cos(plotazim))/90
            y = (inci * np.sin(plotazim))/90
            polarities.append(polarity)
            ax.text(x, y, "  " + station, va="top", bbox=bbox, zorder=2)


        azims = np.array(azims)
        incis = np.array(incis)
        incis=incis/90

        x=incis*np.cos(azims)
        y=incis*np.sin(azims)
        polarities = np.array(polarities, dtype=bool)
        ax.scatter(x, y, marker="o", lw=1, facecolor="w", edgecolor="k", s=50, zorder=3)
        #mask = (polarities == True)
        ax.set_title("Focal Mechanism")
        ax.set_axis_off()
        self.draw()
