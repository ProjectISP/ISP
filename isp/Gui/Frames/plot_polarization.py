from isp.Gui.Frames import MatplotlibCanvas
from isp.Gui.Frames.uis_frames import UitestFrame
from isp.Gui import pw
from mpl_toolkits.mplot3d import Axes3D




class PlotPolarization(pw.QFrame, UitestFrame):
    def __init__(self, z, r, t):
        super(PlotPolarization, self).__init__()
        self.setupUi(self)
        self._z = z
        self._r = r
        self._t = t
        self.canvas = MatplotlibCanvas(self.plotMatWidget_polarization)
        self.canvas.figure.gca(projection='3d')

        self.plotBtn.clicked.connect(lambda: self.plot_particle())
    ##
    @property
    def get_z(self):
        return self._z

    ##
    def getSBvalue(self):
        #print(self.projSB.value())
        return self.projSB.currentText()


    def plot_particle(self):
        self.canvas.plot_projection(self._r,self._t,self._z,axes_index=0)


    #def plot(self):
    #    print(self._t)



        # if self.projSB.currentText() == "R-Z":
        #     self.canvas.plot(self._r, self._z, 0, clear_plot=True, linewidth=0.5)
        #     self.canvas.set_xlabel(0, "Radial")
        #     self.canvas.set_ylabel(0, "Vertical")
        #
        # if self.projSB.currentText() == "T-Z":
        #     self.canvas.plot(self._t, self._z, 0, clear_plot=True, linewidth=0.5)
        #     self.canvas.set_xlabel(0, "Transversal")
        #     self.canvas.set_ylabel(0, "Vertical")
        #
        # if self.projSB.currentText() == "R-T":
        #     self.canvas.plot(self._r, self._t, 0, clear_plot=True, linewidth=0.5)
        #     self.canvas.set_xlabel(0, "Radial")
        #     self.canvas.set_ylabel(0, "Transversal")
