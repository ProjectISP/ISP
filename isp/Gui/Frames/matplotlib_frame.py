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
from matplotlib.transforms import offset_copy
from matplotlib.widgets import SpanSelector
from mpl_toolkits.mplot3d import Axes3D
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
        self.cdi_enter_axes = None
        self.pick_connect = None
        self.axes = None
        self.__callback_on_double_click = None
        self.__callback_on_click = None
        self.__callback_on_select = None
        self.__selected_axe_index = None
        self.__callback_on_pick = None
        self.__selector = None
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

        self.disconnect_events()
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

    def register_on_select(self, func, **kwargs):
        """
        Register on_select callback using SpanSelector.

        :param func: A callback method for on_select, func(ax_index, min, max), ax_index is the
            index of the current Matplotlib.Axes and min/max are floats.

        :keyword kwargs:

        :keyword direction: "horizontal" or "vertical". Default = "horizontal".

        :keyword minspan: float, default is 1E-6. If selection is less than minspan, do not
            call on_select callback.

        :keyword useblit: bool, default is True. If True, use the backend-dependent blitting
            features for faster canvas updates.

        :keyword rectprops: dict, default is dict(alpha=0.5, facecolor='red').
            Dictionary of matplotlib.patches.Patch properties.

        :keyword onmove_callback: func(min, max), min/max are floats, default is None.
            Called on mouse move while the span is being selected.

        :keyword span_stays: bool, default is False. If True, the span stays visible after the
            mouse is released.

        :keyword button: int or list of ints. Determines which mouse buttons activate the
            span selector. Default is MouseButton.LEFT. Use MouseButton.RIGHT, MouseButton.CENTER
            or MouseButton.LEFT.

        :return:
        """
        if not self.cdi_enter_axes:
            self.cdi_enter_axes = self.mpl_connect("axes_enter_event", self.__on_enter_axes)

        direction = kwargs.pop("direction", "horizontal")
        useblit = kwargs.pop("useblit", True)
        minspan = kwargs.pop("minspan", 1.E-6)
        rectprops = kwargs.pop("rectprops", dict(alpha=0.5, facecolor='red'))
        button = kwargs.pop("button", MouseButton.LEFT)

        # register callback.
        self.__callback_on_select = func
        self.__selector = SpanSelector(self.get_axe(0), self.__on_select, direction=direction,
                                       useblit=useblit, minspan=minspan, rectprops=rectprops,
                                       button=button, **kwargs)

    def __on_select(self, xmin, xmax):
        if self.__callback_on_select:
            self.__callback_on_select(self.__selected_axe_index, xmin, xmax)

    def __on_enter_axes(self, event):
        if self.__selector and isinstance(self.__selector, SpanSelector):
            # new way to get axes index from event.
            # event.inaxes.get_subplotspec().rowspan.start
            self.__selected_axe_index = self.get_axe_index(event.inaxes)
            self.__selector.new_axes(event.inaxes)
            self.__selector.update_background(event)

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

    def disconnect_events(self):
        if self.button_connection:
            self.mpl_disconnect(self.button_connection)

        if self.cdi_enter:
            self.mpl_disconnect(self.cdi_enter)

        if self.cdi_leave:
            self.mpl_disconnect(self.cdi_leave)

        if self.cdi_enter_axes:
            self.mpl_disconnect(self.cdi_enter_axes)

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

    def get_axe_index(self, ax) -> int:
        """
        Get the index of an axe.

        :param ax: The Matplotlib.Axes

        :return: The axe index.
        """
        return numpy.where(self.axes == ax)[0][0]

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

    def set_new_subplot(self, nrows, ncols, update=True, **kwargs):
        sharex = kwargs.pop("sharex", "all")
        self.figure.clf()
        plt.close(self.figure)
        nrows = max(nrows, 1)  # avoid zero rows.
        self.axes = self.figure.subplots(nrows=nrows, ncols=ncols, sharex=sharex, **kwargs)
        self.__flat_axes()
        if update:
            self.draw()

    def set_xlabel(self, axe_index, value, update=True):
        ax = self.get_axe(axe_index)
        if ax:
            ax.set_xlabel(value)
            if update:
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

    def set_warning_label(self, ax: Axes or int, text: str):
        """
        Sets an label box at the upper right corner of the axes.

        :param ax: The axes or axes_index to add the annotation.
        :param text: The text
        :return:
        """
        if type(ax) == int:
            ax = self.get_axe(ax)
        bbox = dict(boxstyle="round", alpha= 0.5, fc="red")
        return ax.annotate(text, xy=(0.15, 1), xycoords='axes fraction', xytext=(-20, -20), textcoords='offset points',
                           ha="right", va="top",  bbox=bbox)

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

    def __plot_date(self, x, y, ax, clear_plot=True, update=True, **kwargs):
        if clear_plot:
            ax.cla()
        artist, = ax.plot_date(x, y, **kwargs)
        if update:
            try:
                # Draw can raise ValueError
                self.draw_idle()
                return artist
            except ValueError:
                artist.remove()
                return None
        else:
            return artist

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
        yscale = kwargs.pop('yscale', 'linear')
        xscale = kwargs.pop('xscale', 'linear')
        vmin = kwargs.pop('vmin', numpy.amin(z))
        vmax = kwargs.pop('vmax', numpy.amax(z))
        orientation = kwargs.pop('orientation', 'vertical')
        clabel = kwargs.pop('clabel', '')
        x_label = ax.get_xlabel()

        if plot_type == "contourf":
            levels = kwargs.pop('levels', 40)
            cs = ax.contourf(x, y, z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

        elif plot_type == "pcolormesh":
            cs = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

        elif plot_type == "scatter":
            area = 10.*z**2  # points size from 0 to 5
            cs = ax.scatter(x, y, s=area, c=z, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax, marker=".", **kwargs)

        else:
            raise ValueError("Invalid value for plot_type it must be equal to either contourf or scatter.")

        cs.set_clim(vmin, vmax)

        self.clear_color_bar()
        if show_colorbar:
            if orientation == 'horizontal':
                self.__cbar: Colorbar = self.figure.colorbar(cs, ax=ax, extend='both',
                                                             orientation = 'horizontal', pad=0.05)
            else:
                self.__cbar: Colorbar = self.figure.colorbar(cs, ax=ax, extend='both', pad=0.0)
            self.__cbar.ax.set_ylabel(clabel)

        ax.set_xlim(*self.get_xlim_from_data(ax, 0))
        ax.set_ylim(*self.get_ylim_from_data(ax, 0))

        if xscale != "linear":
            ax.set_xscale('log')
        if yscale != "linear":
            ax.set_yscale('log')

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
                artist = self.__plot(x, y, tw_ax, clear_plot=False, **kwargs)
                if artist:
                    self.set_yaxis_color(tw_ax, artist.get_color())
                return artist
            else:
                return self.__plot(x, y, ax, clear_plot=clear_plot, **kwargs)

    def plot_date(self, x, y, axes_index, clear_plot=True, is_twinx=False, update=True, **kwargs):
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
                artist = self.__plot_date(x, y, tw_ax, clear_plot=True, update=update, **kwargs)
                if artist:
                    self.set_yaxis_color(tw_ax, artist.get_color())
                return artist
            else:
                return self.__plot_date(x, y, ax, clear_plot=clear_plot, update=update, **kwargs)

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

            #ax = self.figure.gca(projection='3d')
            ax = Axes3D(self.figure)
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

    def pcolormesh(self, x, y, z, axes_index, clear_plot=True, show_colorbar=True, **kwargs):
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
            self.__plot_3d(x, y, z, ax, "pcolormesh", clear_plot=clear_plot, show_colorbar=show_colorbar, **kwargs)

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

            line = ax.vlines(x_pos, ymin, ymax, color=color, picker=picker,lw=0.75, **kwargs)

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
        self.canvas = MatplotlibCanvas(self.main_widget, obj, **kwargs)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("Done!", 2000)

        # used as a callback for closing event in this window.
        self.__close_window_callback = None

    def connect_close(self, func):
        self.__close_window_callback = func

    def set_canvas(self, mpc: MatplotlibCanvas):
        self.canvas = mpc
        self.layout.removeWidget(self.mpw)

    def fileQuit(self):
        self.close()
        self.canvas = None

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
    MAP_SERVICE_URL = 'https://www.gebco.net/data_and_products/gebco_web_services/2020/mapserv?'
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

    def plot_map(self, x, y, scatter_x, scatter_y, scatter_z, axes_index, clear_plot=True, **kwargs):
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
        from isp import ROOT_DIR
        import os

        resolution = kwargs.pop('resolution')
        stations = kwargs.pop('stations')
        os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join(ROOT_DIR, "maps")
        name_stations = []
        lat = []
        lon = []
        for name, coords in stations.items():
            name_stations.append(name)
            lat.append(float(coords[1]))
            lon.append(float(coords[0]))

        self.clear()
        ax = self.get_axe(axes_index)


        geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
        #layer = 'GEBCO_08 Hillshade'
        layer ='GEBCO_2020_Grid'
        #layer = 'shaded_relief'
        xmin = int(x-6)
        xmax = int(x+6)
        ymin = int(y-4)
        ymax = int(y+4)
        extent = [xmin, xmax, ymin, ymax]

        ax.set_extent(extent, crs=ccrs.PlateCarree())


        if resolution == "high":
            try:

                wms = WebMapService(self.MAP_SERVICE_URL)
                ax.add_wms(wms, layer)

            except:

                ax.background_img(name='ne_shaded', resolution=resolution)

        elif resolution == "low":

                coastline_10m = cartopy.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                    edgecolor='k', alpha=0.6, linewidth=0.5, facecolor=cartopy.feature.COLORS['land'])
                ax.background_img(name='ne_shaded', resolution=resolution)
                ax.add_feature(coastline_10m)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.2, color='gray', alpha=0.2, linestyle='-')

        gl.top_labels = False
        gl.left_labels = False
        gl.xlines = False
        gl.ylines = False

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Plot stations
        geodetic_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        text_transform = offset_copy(geodetic_transform, units='dots', x=-25)
        ax.scatter(lon, lat, s=12, marker="^", color='red', alpha=0.7, transform=ccrs.PlateCarree())
        N = len(name_stations)
        for n in range(N):
            lon1 = lon[n]
            lat1 = lat[n]
            name = name_stations[n]

            ax.text(lon1, lat1, name, verticalalignment='center', horizontalalignment='right', transform=text_transform,
                    bbox=dict(facecolor='sandybrown', alpha=0.5, boxstyle='round'))


        ax.plot(x, y, color='red', marker='*',markersize=3)
        ax.scatter(scatter_x, scatter_y, s=10, c=scatter_z/10, marker=".", alpha=0.3, cmap=plt.get_cmap('YlOrBr'))

        # Create an inset GeoAxes showing the Global location
        sub_ax = ax.figure.add_axes([0.70, 0.73, 0.28, 0.28], projection=ccrs.PlateCarree())
        sub_ax.set_extent([-179.9, 180, -89.9, 90], geodetic)

        # Make a nice border around the inset axes.
        effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)
        sub_ax.outline_patch.set_path_effects([effect])

        # Add the land, coastlines and the extent .
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
        cs = ax.scatter(x, y, s=10, c=depth / 10, marker="^", cmap=plt.get_cmap('YlOrBr'))
        cs.set_clim(vmin, vmax)
        if show_colorbar:
            self.__cbar: Colorbar = self.figure.colorbar(cs, ax=ax, orientation='horizontal', fraction=0.05,
                                                         extend='both', pad=0.15)
            self.__cbar.ax.set_ylabel("Depth [m]")
        self.draw()

    def global_map(self, axes_index, plot_earthquakes = False, update = True,  show_colorbar = False, clear_plot = True,
                   show_stations = False, show_station_names = False, show_distance_circles = False,  **kwargs):
        import numpy as np
        from isp import ROOT_DIR
        import os

        os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join(ROOT_DIR, "maps")
        lon = kwargs.pop('lon', [])
        lat = kwargs.pop('lat', [])
        depth = kwargs.pop('depth', [])
        mag = kwargs.pop('magnitude', [])
        coordinates = kwargs.pop('coordinates', {})
        resolution = kwargs.pop('resolution', 'high')
        color = kwargs.pop('color', 'red')
        size = kwargs.pop('size',8)

        extent = kwargs.pop("extent", [])

        lon30 = kwargs.pop('lon30', [])
        lat30 = kwargs.pop('lat30', [])
        lon90 = kwargs.pop('lon90', [])
        lat90 = kwargs.pop('lat90', [])
        #line1 = []
        #line2 = []
        
        if resolution == "Natural Earth":

            resolution = "low"

        else:

            resolution = "high"

        ax = self.get_axe(axes_index)

        geodetic_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        text_transform = offset_copy(geodetic_transform, units='dots', x=-25)
        depth = np.array(depth) / 1000
        mag = np.array(mag)
        mag = 0.25 * np.exp(mag)

        if clear_plot:
            ax.clear()

        #if len(extent)>=0:
        #    try:
        #        ax.set_extent(extent)
        #    except:
        #        pass
        if update:
            ax.background_img(name='ne_shaded', resolution=resolution)
        else:
            pass


        if show_stations:
            lat = []
            lon = []
            sta_ids = []
            for key in coordinates.keys():

                for j in range(len(coordinates[key][0][:])):

                    sta_ids.append(coordinates[key][1][j])
                    lat.append(coordinates[key][2][j])
                    lon.append(coordinates[key][3][j])
                    if show_station_names:
                        ax.text(coordinates[key][3][j], coordinates[key][2][j], key + "." + coordinates[key][1][j], verticalalignment='center',
                                horizontalalignment='right', transform=text_transform,
                                bbox=dict(facecolor='sandybrown', alpha=0.5, boxstyle='round'))
                    else:
                        pass

            ax.scatter(lon, lat, s=size, marker="^", color=color, alpha=0.7, transform=ccrs.PlateCarree())


        if plot_earthquakes:
            color_map = plt.cm.get_cmap('rainbow')
            reversed_color_map = color_map.reversed()
            cs = ax.scatter(lon, lat, s=mag, c=depth, edgecolors="black", cmap=reversed_color_map, vmin = 0,
                            vmax = 600)

            kw = dict(prop="sizes", num=5, fmt="{x:.0f}", color="red", alpha=0.5, func=lambda s: np.log(s / 0.25))
            ax.legend(*cs.legend_elements(**kw), loc="lower right", title="Magnitudes")

            if show_colorbar:
                try:
                    self.__cbar.ax.clear()
                except:
                    pass
                self.__cbar: Colorbar = self.figure.colorbar(cs, ax=ax, orientation='horizontal', fraction=0.05,
                                                              extend='both', pad=0.08)
                self.__cbar.ax.set_ylabel("Depth [km]")
                # magnitude legend

        if show_distance_circles:
           #if len(line1)>0 and len(line2)>0:
           try:
               l1 = line1.pop(0)
               l2 = line2.pop(0)
               l1.remove()
               l2.remove()
               del l1
               del l2

           except:
                pass

           line1 =  ax.scatter(lon30, lat30, s=8, c="white")
           line2 =  ax.scatter(lon90, lat90, s=8, c="white")
            #ax.plot(lon30, lat30, color='white', linestyle='--',transform=ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                          linewidth=0.2, color='gray', alpha=0.2, linestyle='-')

        gl.top_labels = False
        gl.left_labels = False
        gl.xlines = False
        gl.ylines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        self.draw()


    def clear_color_bar(self):
        try:
            if self.__cbar:
                self.__cbar.remove()
        except:
            pass

    def __flat_axes(self):
        # make sure axes are always a np.array
        if type(self.axes) is not numpy.ndarray:
            self.axes = numpy.array([self.axes])
        self.axes = self.axes.flatten()

    def set_new_subplot_cartopy(self, nrows, ncols, update=True, **kwargs):
        sharex = kwargs.pop("sharex", "all")
        self.figure.clf()
        plt.close(self.figure)
        nrows = max(nrows, 1)  # avoid zero rows.
        self.axes = self.figure.subplots(nrows=nrows, ncols=ncols, subplot_kw = {"projection":ccrs.PlateCarree()},
                                         **kwargs)
        self.__flat_axes()

        if update:
            self.draw()

    def plot_disp_map(self, axes_index, grid, interp, color, plot_global_map =False, show_relief = False):

        import numpy as np
        from isp import ROOT_DIR
        import os

        os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join(ROOT_DIR, "maps")

        #resolution = kwargs.pop('resolution', 'low')
        resolution = "low"
        lats = grid[0]['grid'][:,0][:,1]
        lons = grid[0]['grid'][1,:][:,0]

        ax = self.get_axe(axes_index)
#        geodetic_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
#        text_transform = offset_copy(geodetic_transform, units='dots', x=-25)
        ax.clear()
        self.clear_color_bar()
        if show_relief:
            ax.background_img(name='ne_shaded', resolution=resolution)
        xmin = min(lons)
        xmax = max(lons)
        ymin = min(lats)
        ymax = max(lats)
        extent = [xmin, xmax, ymin, ymax]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.coastlines()

        #map = ax.contourf(lons, lats, grid[0]['m_opt_relative'], transform=ccrs.PlateCarree(), cmap="RdBu",
        #                    vmin=-10, vmax=10, alpha=0.7)
        img_extent = (xmin, xmax, ymin, ymax)
        map = ax.imshow(grid[0]['m_opt_relative'], interpolation=interp, origin='lower', extent=img_extent,
                        transform=ccrs.PlateCarree(), cmap=color, vmin=-10, vmax=10, alpha=0.7)

        self.__cbar: Colorbar = self.figure.colorbar(map, ax=ax, orientation='vertical', fraction=0.05,
                                                     extend='both', pad=0.08)

        self.__cbar.ax.set_ylabel("Velocity [km/s]")

        # Create an inset GeoAxes showing the Global location
        geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
        if plot_global_map:
            sub_ax = ax.figure.add_axes([0.70, 0.73, 0.28, 0.28], projection=ccrs.PlateCarree())
            sub_ax.set_extent([-179.9, 180, -89.9, 90], geodetic)

            # Make a nice border around the inset axes.
            effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)
            sub_ax.outline_patch.set_path_effects([effect])

            # Add the land, coastlines and the extent .
            sub_ax.add_feature(cfeature.LAND)
            sub_ax.coastlines()
            extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
            sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='none',
                                  edgecolor='blue', linewidth=1.0)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.2, color='gray', alpha=0.2, linestyle='-')

        gl.top_labels = False
        gl.left_labels = False
        gl.xlines = False
        gl.ylines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

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

    def drawFocMec(self, strike, dip, rake, sta, az, inc, pol, axes_index):
        from obspy.imaging.beachball import beach
        import numpy as np
        azims_pos = []
        incis_pos = []
        azims_neg = []
        incis_neg = []
        polarities = []
        bbox = dict(boxstyle="round, pad=0.2", fc="w", ec="k", lw=1.5, alpha=0.7)
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
            if polarity == "U":
                azims_pos.append(plotazim)
                incis_pos.append(inci)
                x_pos = (inci * np.cos(plotazim))/90
                y_pos = (inci * np.sin(plotazim))/90
                polarities.append(polarity)
                ax.text(x_pos, y_pos, "  " + station, va="top", bbox=bbox, zorder=2)
            if polarity == "D":
                azims_neg.append(plotazim)
                incis_neg.append(inci)
                x_neg = (inci * np.cos(plotazim)) / 90
                y_neg = (inci * np.sin(plotazim)) / 90
                polarities.append(polarity)
                ax.text(x_neg, y_neg, "  " + station, va="top", bbox=bbox, zorder=2)

        azims_pos = np.array(azims_pos)
        incis_pos = np.array(incis_pos)
        incis_pos=incis_pos/90
        x_pos=incis_pos*np.cos(azims_pos)
        y_pos=incis_pos*np.sin(azims_pos)
        #polarities = np.array(polarities, dtype=bool)
        ax.scatter(x_pos, y_pos, marker="o", lw=1, facecolor="b", edgecolor="k", s=50, zorder=3)

        azims_neg = np.array(azims_neg)
        incis_neg = np.array(incis_neg)
        incis_neg = incis_neg / 90
        x_neg = incis_neg * np.cos(azims_neg)
        y_neg = incis_neg * np.sin(azims_neg)
        ax.scatter(x_neg, y_neg, marker="o", lw=1, facecolor="w", edgecolor="k", s=50, zorder=3)
        #mask = (polarities == True)
        ax.set_title("Focal Mechanism")
        ax.set_axis_off()
        self.draw()

    def drawSynthFocMec(self, axes_index, **kwargs):
        from obspy.imaging.beachball import beach
        first_polarity = kwargs.pop("first_polarity")
        first_polarity = first_polarity[0:3]
        mti = kwargs.pop("mti")

        self.clear()
        ax = self.get_axe(axes_index)
        if len(first_polarity) > 0:
            beach2 = beach(first_polarity, facecolor='r', linewidth=1., alpha=0.3, width=2)
        if len(mti) > 0:
            beach2 = beach(mti, facecolor='b', linewidth=1., alpha=0.3, width=2)

        ax.add_collection(beach2)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)

        ax.set_title("Focal Mechanism")
        ax.set_axis_off()
        self.draw()
