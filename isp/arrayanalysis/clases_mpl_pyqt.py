
import sys
sys.path.append(r"/Users/robertocabieces/Documents/obs_array")

from PyQt4 import QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import Divider, Size
import matplotlib.pyplot as plt

def fix_axes_size_incm(axew, axeh):
    axew = axew/2.54
    axeh = axeh/2.54

    #lets use the tight layout function to get a good padding size for our axes labels.
    fig = plt.gcf()
    ax = plt.gca()
    fig.tight_layout()
    #obtain the current ratio values for padding and fix size
    oldw, oldh = fig.get_size_inches()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    #work out what the new  ratio values for padding are, and the new fig size.
    neww = axew+oldw*(1-r+l)
    newh = axeh+oldh*(1-t+b)
    newr = r*oldw/neww
    newl = l*oldw/neww
    newt = t*oldh/newh
    newb = b*oldh/newh

    #right(top) padding, fixed axes size, left(bottom) pading
    hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
    vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]

    divider = Divider(fig, (0.0, 0.0, 1., 1.), hori, vert, aspect=False)
    # the width and height of the rectangle is ignored.

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    #we need to resize the figure now, as we have may have made our axes bigger than in.
    fig.set_size_inches(neww,newh)
    
class MplCanvas_special(FigureCanvas):
    def __init__(self):
        
        #self.fig = Figure(facecolor = "0.94")
        self.fig = Figure()
        
        self.ax1 = self.fig.add_subplot(211)
        
        self.ax2 = self.fig.add_subplot(212)
        ##If we want to join x axis
        self.ax2.get_shared_x_axes().join(self.ax1,self.ax2)
        
        ## if we want to show the colour bar
        self.cax = self.fig.add_axes([0.94, 0.11, 0.025, 0.35])

        FigureCanvas.__init__(self, self.fig)
        


class MatplotlibWidget_special(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = MplCanvas_special()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        

class MplCanvas_1(FigureCanvas):
    def __init__(self):
        
        #self.fig = Figure(facecolor = "0.94")
        self.fig = Figure()
        
        self.ax1 = self.fig.add_subplot(111)
                
        ## if we want to show the colour bar
        self.cax = self.fig.add_axes([0.93, 0.11, 0.02, 0.7])

        FigureCanvas.__init__(self, self.fig)
        


class MatplotlibWidget_1(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = MplCanvas_1()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        
class MplCanvas_2(FigureCanvas):
    def __init__(self):
        #hspace=0.2
        #wspace=0.2
        #[left, bottom, width, height]
        #self.fig = Figure(facecolor = "0.94")
        self.fig = Figure()
        
        #self.ax1 = self.fig.add_subplot(211)
        
        self.ax1 = self.fig.add_subplot(211,position=[0.3, 0.53,0.45, 0.45])
        self.ax2 = self.fig.add_subplot(212,position=[0.3, 0.03,0.45, 0.45])
        
        #self.ax2 = self.fig.add_subplot(212,position=[0.1, 0.95, 10, 10])
                
        ## if we want to show the colour bar
        #self.cax = self.fig.add_axes([0.93, 0.11, 0.02, 0.7])

        FigureCanvas.__init__(self, self.fig)
        


class MatplotlibWidget_2(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = MplCanvas_2()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
        
        
        
        
class MplCanvas_3(FigureCanvas):
    def __init__(self):
        
        #self.fig = Figure(facecolor = "0.94")
        self.fig = Figure()
        
        self.ax1 = self.fig.add_subplot(311)
        
        self.ax2 = self.fig.add_subplot(312)
        
        self.ax3 = self.fig.add_subplot(313)
        self.ax1.get_shared_x_axes().join(self.ax1,self.ax2,self.ax3)
        ## if we want to show the colour bar
        self.cax = self.fig.add_axes([0.93, 0.11, 0.02, 0.7])

        FigureCanvas.__init__(self, self.fig)
        


class MatplotlibWidget_3(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = MplCanvas_3()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)