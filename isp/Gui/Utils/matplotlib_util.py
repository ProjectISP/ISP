from typing import Callable, List, Set
from matplotlib.axes import Axes
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np
from matplotlib.widgets import SpanSelector

class Keys:
    NoKey = "None"
    Ctr = "control"
    Shift = "shift"
    Alt = "alt"
    Plus = "+"
    Minus = "-"
    Esc = "escape"
    Up = "up"
    Down = "down"
    # TODO include more keys as necessary


# include more key names here if necessary.
POSITIVE_POLARITY_KEYS = [Keys.Plus, Keys.Shift]

# include more key names here if necessary.
NEGATIVE_POLARITY_KEYS = [Keys.Minus, Keys.Ctr]


def map_polarity_from_pressed_key(key_name: str):
    polarity = "?"
    color = "red"
    if key_name in POSITIVE_POLARITY_KEYS:
        polarity = "+"
        color = "green"
    elif key_name in NEGATIVE_POLARITY_KEYS:
        polarity = "-"
        color = "blue"
    return polarity, color


class CollectionLassoSelector:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.on_select)
        self.ind = []

    def on_select(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


class ExtendSpanSelector(SpanSelector):

    def __init__(self, ax: Axes, onselect: Callable, direction: str, sharex=False, **kwargs):
        # Keep track of all axes
        self.__on_move_callback = kwargs.pop("onmove_callback", None)
        super().__init__(ax, onselect, direction, onmove_callback=self.__on_move, **kwargs)

        self.__kwargs = kwargs
        self.sharex: bool = sharex
        self.__sub_axes: List[Axes] = []
        self.__sub_selectors: Set[SpanSelector] = set()

    def _release(self, event):
        self.clear_subplot()
        return super()._release(event)

    def __on_sub_select(self, xmin, xmax):
        # this is just a placeholder for sub selectors
        pass

    def clear_subplot(self):
        all([ss.clear() for ss in self.__sub_selectors])

    def create_sub_selectors(self):
        self.__sub_selectors = {
            SpanSelector(axe, self.__on_sub_select, self.direction, **self.__kwargs)
            for axe in self.__sub_axes
        }

    def remove_sub_selectors(self):
        all([s.disconnect_events() for s in self.__sub_selectors])
        self.__sub_selectors.clear()

    @staticmethod
    def draw_selector(selector: SpanSelector, vmin, vmax):
        selector._draw_shape(vmin, vmax)
        selector.set_visible(True)
        selector.update()

    def __on_move(self, vmin, vmax):
        all([self.draw_selector(ss, vmin, vmax) for ss in self.__sub_selectors])
        if self.__on_move_callback:
            return self.__on_move_callback(vmin, vmax)

    def set_sub_axes(self, axes: List[Axes]):
        if self.sharex:
            self.__sub_axes = [axe for axe in axes if axe != self.ax]
            self.remove_sub_selectors()
            self.create_sub_selectors()
            self.canvas.draw()
