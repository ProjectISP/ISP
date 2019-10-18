from functools import wraps


def on_click_matplot(canvas):
    """
    Bind the decorated method to a click on matplot canvas, and inject the canvas
    and event from matplotlib to the method

    * Example::

        @on_click_matplot(canvas)
        def my_func(event, canvas=None):
             "Call when user click the matplot canvas"
             pass

    :param canvas: Expect to be an instance of MatplotlibCanvas

    :return:
    """
    from isp.Gui.Frames.matplotlib_frame import MatplotlibCanvas

    if not isinstance(canvas, MatplotlibCanvas):
        raise AttributeError("Canvas must be an instance of MatplotlibCanvas")

    def app_decorator(func):
        canvas.button_connection = canvas.mpl_connect('button_press_event',
                                                      lambda event: func(event, canvas))
    return app_decorator


def embed_matplot_canvas(widget_name):
    """
    Embed a matplotlib to the parent widget and inject the canvas to the decorated method.

    * Example::

        @embed_matplot_canvas("my_widget")
        def my_func(self, canvas=None):
             pass

    :param widget_name: The name of the parent widget, QWidget or QFrame.

    :return:
    """
    def app_decorator(func):
        @wraps(func)
        def wrap_func(self, *args, **kwargs):
            from isp.Gui.Frames import BaseFrame
            mpc = None
            if isinstance(self, BaseFrame):
                from isp.Gui.Frames.matplotlib_frame import MatplotlibCanvas
                parent = self.__dict__.get(widget_name)
                mpc = MatplotlibCanvas(parent)
            return func(self, mpc, *args, **kwargs)
        return wrap_func
    return app_decorator



