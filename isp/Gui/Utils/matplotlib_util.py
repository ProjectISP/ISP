from typing import Callable, List

from matplotlib.axes import Axes
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


class ExtendSpanSelector(SpanSelector):

    def __init__(self, ax: Axes, onselect: Callable, direction: str, sharex=False, **kwargs):
        # Keep track of all axes
        self.__on_move_callback = kwargs.pop("onmove_callback", None)
        super().__init__(ax, onselect, direction, onmove_callback=self.__on_move, **kwargs)

        self.__kwargs = kwargs
        self.sharex: bool = sharex
        self.__sub_axes: List[Axes] = []
        self.__sub_selectors: List[SpanSelector] = []

    def _release(self, event):
        self.clear_subplot()
        return super()._release(event)

    def __on_sub_select(self, xmin, xmax):
        # this is just a placeholder for sub selectors
        pass

    def clear_subplot(self):
        all([ss.clear() for ss in self.__sub_selectors])

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
            all([s.disconnect_events() for s in self.__sub_selectors])
            self.__sub_selectors.clear()
            self.__sub_axes = [axe for axe in axes if axe != self.ax]
            self.__sub_selectors = [
                SpanSelector(axe, self.__on_sub_select, self.direction, **self.__kwargs)
                for axe in self.__sub_axes
            ]
