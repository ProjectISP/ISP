from obspy import UTCDateTime, Stream
from isp.Gui import pw
from isp.Gui.Frames import MatplotlibCanvas, MessageDialog
from isp.Gui.Frames.uis_frames import UiVespagram
from isp.arrayanalysis.array_analysis import vespagram_util
import numpy as np
import matplotlib.pyplot as plt
from isp.Gui.Utils.pyqt_utils import add_save_load

@add_save_load()
class Vespagram(pw.QFrame, UiVespagram):
    def __init__(self, st:Stream, inv, t1: UTCDateTime, t2: UTCDateTime, max_values: dict, fmin, fmax, time_win):
        super(Vespagram, self).__init__()
        self.setupUi(self)
        """
        Vespagram, computed a fixed slowness or backazimuth

        :param params required to initialize the class:
            
        
        """

        self.st = st
        self.inv = inv
        self.t1 = t1
        self.t2 = t2
        self.max_values = max_values
        # set max_values
        self.slownessDB.setValue(max_values["max_slowness"])
        self.backazimuthCB.setValue(max_values["max_azimuth"])
        self.freq_min_DB.setValue(fmin)
        self.freq_max_DB.setValue(fmax)
        self.win_len_DB.setValue(time_win)
        self.overlapSB.setValue(int(95))
        ###

        self.canvas_vespagram = MatplotlibCanvas(self.widget_vespagram, nrows=2)
        self.run_vespaBtn.clicked.connect(self.run_vespa)
        self.plotBtn.clicked.connect(self.plot_vespa)

    def closeEvent(self, ce):
        self.save_values()

    def __load__(self):
        self.load_values()

    #@AsycTime.run_async()
    def run_vespa(self):


        vespa = vespagram_util(self.st, self.freq_min_DB.value(), self.freq_max_DB.value(), self.win_len_DB.value(),
                               self.slownessDB.value(), self.backazimuthCB.value(), self.inv, self.t1, self.t2,
                               self.overlapSB.value(), selection=self.selectionCB.currentText(), method="FK")

        self.x, self.y, self.log_vespa_spectrogram = vespa.vespa_deg()
        md = MessageDialog(self)
        md.set_info_message("Vespagram Estimated " + self.selectionCB.currentText() + "!!!")


    def plot_vespa(self):

        if self.selectionCB.currentText() == "Slowness":
            self.__plot_vespa_slow()
        else:
            self.__plot_vespa_az()


    def __plot_vespa_slow(self):
        colour = self.colourCB.currentText()
        vespagram = np.clip(self.log_vespa_spectrogram, a_min=0, a_max=1)
        self.canvas_vespagram.plot_contour(self.x, self.y, vespagram, axes_index=0, clabel="Rel. Power",
                                           cmap=plt.get_cmap(colour))
        self.canvas_vespagram.set_xlabel(0, "Time (s)")
        self.canvas_vespagram.set_ylabel(0, "Azimuth")

    def __plot_vespa_az(self):
        colour = self.colourCB.currentText()
        #vespagram = np.clip(self.log_vespa_spectrogram, a_min=base_line, a_max=0)
        vespagram = np.clip(self.log_vespa_spectrogram, a_min=0, a_max=1)
        self.canvas_vespagram.plot_contour(self.x, self.y, vespagram, axes_index=1, clabel="Rel Power",
                                           cmap=plt.get_cmap(colour))
        self.canvas_vespagram.set_xlabel(1, "Time (s)")
        self.canvas_vespagram.set_ylabel(1, "Slowness (s/km)")