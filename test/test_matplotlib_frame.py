import os
import sys

import matplotlib
import numpy as np

from isp.Gui.Frames import MatplotlibFrame

matplotlib.use('Qt5Agg')
from PyQt5 import QtWidgets


progname = os.path.basename(sys.argv[0])
progversion = "0.1"

qApp = QtWidgets.QApplication(sys.argv)


def on_select(ax_index, xmin, xmax):
    mpf.canvas.plot(xmax, 2, ax_index, clear_plot=False, marker="o")
    mpf.canvas.draw()


def on_click(event, canvas):
    print(event, canvas)


def on_dlb_click(event, canvas):
    print(event, canvas)


mpf = MatplotlibFrame(None)
mpf.canvas.set_new_subplot(2, 1)
mpf.canvas.on_double_click(on_dlb_click)
mpf.canvas.register_on_select(on_select)
mpf.canvas.plot([1, 2, 3], [1, 2, 3], 0)
mpf.canvas.plot([1, 2, 3], [1, 2, 3], 1)


mpf.setWindowTitle("%s" % progname)
mpf.show()
sys.exit(qApp.exec_())
