from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from matplotlib import rcParams

rcParams["font.family"] = "Ubuntu"
rcParams["font.size"] = 8
rcParams['axes.linewidth'] = 0.4
rcParams['patch.linewidth'] = .25


class MatplotlibWidget(Canvas):
    def __init__(self, parent=None):
        self.figure = Figure()

        super(MatplotlibWidget, self).__init__(self.figure)
        self.setParent(parent)
        super(MatplotlibWidget, self).setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        super(MatplotlibWidget, self).updateGeometry()

    def sizeHint(self):
        return QSize(*self.get_width_height())

    def minimumSizeHint(self):
        return QSize(10, 10)
