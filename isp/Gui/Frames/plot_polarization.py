from obspy import Stream
from isp.Gui.Frames import MatplotlibCanvas
from isp.Gui.Frames.uis_frames import UiPlotPolarization
from isp.Gui import pw
import numpy as np
from obspy.signal.polarization import flinn


class PlotPolarization(pw.QFrame, UiPlotPolarization ):
    def __init__(self, z, r, t):
        super(PlotPolarization, self).__init__()
        self.setupUi(self)
        self._z = z-np.mean(z)
        self._r = r-np.mean(r)
        self._t = t-np.mean(t)
        all_traces = [self._z, self._r, self._t]
        self.st = Stream(traces=all_traces)
        self._r_max = max(r)
        self._t_max = max(t)
        self._z_max=max(z)
        self.max_value=max([max(abs(z)), max(abs(t)), max(abs(r))])
        self.canvas = MatplotlibCanvas(self.plotMatWidget_polarization)
        self.canvas2D = MatplotlibCanvas(self.plotMatWidget_polarization2, nrows=2, ncols=2, constrained_layout=True)
        #self.canvas.figure.gca(projection='3d')
        self.plot_particle()
        #self.plotBtn.clicked.connect(lambda: self.plot_particle())


    def plot_particle(self):
        azimuth, incidence, rect, plan = flinn(self.st)

        self.canvas.plot_projection(self._r,self._t,self._z, axes_index=0)
        self.canvas.set_ylabel(0, "Radial / North")
        self.canvas.set_xlabel(0, "Transversal / East")
        #self.canvas.set_zlabel(0, "Vertical")

        self.canvas2D.plot(self._r, self._z, 0, clear_plot=True, linewidth=0.5)
        self.canvas2D.set_xlabel(0, "Radial / North")
        self.canvas2D.set_ylabel(0, "Vertical")
        ax1 = self.canvas2D.get_axe(0)
        ax1.set_xlim([-self.max_value,self.max_value])
        ax1.set_ylim([-self.max_value, self.max_value])

        self.canvas2D.plot(self._t, self._z, 1, clear_plot=True, linewidth=0.5)
        self.canvas2D.set_xlabel(1, "Transversal /East")
        self.canvas2D.set_ylabel(1, "Vertical")
        ax2 = self.canvas2D.get_axe(1)
        ax2.set_xlim([-self.max_value, self.max_value])
        ax2.set_ylim([-self.max_value, self.max_value])

        self.canvas2D.plot(self._r, self._t, 2, clear_plot=True, linewidth=0.5)
        self.canvas2D.set_xlabel(2, "Transversal / East")
        self.canvas2D.set_ylabel(2, "Radial / North")
        ax3 = self.canvas2D.get_axe(2)
        ax3.set_xlim([-self.max_value, self.max_value])
        ax3.set_ylim([-self.max_value, self.max_value])

        self.polarizationText.setPlainText("  Azimuth:     {azimuth:.3f} ".format(azimuth=azimuth))
        self.polarizationText.appendPlainText("  Incidence Angle:     {incidence:.3f} ".format(incidence=incidence))
        self.polarizationText.appendPlainText("  Rectilinearity:     {rect:.3f} ".format(rect=rect))
        self.polarizationText.appendPlainText("  Planarity:     {plan:.3f} ".format(plan=plan))