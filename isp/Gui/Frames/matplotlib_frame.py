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

import matplotlib
# Make sure that we are using QT5
from obspy import Stream

from isp.Gui import pw, pyc, BaseFrame

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class MatplotlibCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, st: Stream = None, fig=None):
        self.st = st
        if self.st:
            fig = self.__get_figure_from_stream()

        if not fig:
            raise AttributeError("You must give a stream or a plt.figure")

        super().__init__(fig)

        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   pw.QSizePolicy.Expanding,
                                   pw.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def __get_figure_from_stream(self):
        if self.st:
            return self.st.plot(show=False)


class MatplotlibWidget(pw.QWidget):

    def __init__(self, parent=None, canvas=None):
        super().__init__(parent)
        self.canvas = canvas
        self.vbl = pw.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class MatplotlibFrame(BaseFrame):
    def __init__(self, st: Stream = None, fig=None):
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

        l = pw.QVBoxLayout(self.main_widget)
        sc = MatplotlibCanvas(self.main_widget, st, fig)
        mpw = MatplotlibWidget(self.main_widget, sc)
        l.addWidget(mpw)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("Done!", 2000)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        pw.QMessageBox.about(self, "About",
                                    """
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen

This program is a Qt5 application embedding matplotlib
canvases and Obspy stream.""")
