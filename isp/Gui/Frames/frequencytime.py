from scipy.signal import find_peaks
from isp.DataProcessing import SeismogramDataAdvanced, ConvolveWaveletScipy
from isp.Gui import pw
import matplotlib.pyplot as plt
from isp.Gui.Frames import FilesView, MatplotlibCanvas, MessageDialog
from isp.Gui.Frames.parameters import ParametersSettings
from isp.Gui.Frames.stations_info import StationsInfo
from isp.Gui.Frames.uis_frames import UiFrequencyTime
from isp.Gui.Utils.pyqt_utils import add_save_load, BindPyqtObject
from sys import platform
from isp.Utils import ObspyUtil
from isp.seismogramInspector.MTspectrogram import MTspectrogram, hilbert_gauss
from isp.ant.signal_processing_tools import noise_processing
import numpy as np
from obspy import read
import os
from isp.Gui.Utils import CollectionLassoSelector


@add_save_load()
class FrequencyTimeFrame(pw.QWidget, UiFrequencyTime):
    def __init__(self):
        super(FrequencyTimeFrame, self).__init__()
        self.setupUi(self)
        self.solutions = []
        self.periods_now = []
        self.colors = ["white", "green", "black"]
        self._stations_info = {}
        self.parameters = ParametersSettings()
        # Binds
        self.root_path_bind = BindPyqtObject(self.rootPathForm_2, self.onChange_root_path)
        self.canvas_plot1 = MatplotlibCanvas(self.widget_plot_up, ncols=2, sharey  = True)
        top = 0.900
        bottom = 0.180
        left = 0.045
        right = 0.720
        wspace = 0.135
        self.canvas_plot1.figure.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=0.0)

        ax = self.canvas_plot1.get_axe(0)
        left,width = 0.2, 0.55
        bottom, height = 0.180, 0.72
        spacing = 0.02
        coords_ax = [left+width+spacing, bottom, 0.2, height]
        self.fig = ax.get_figure()
        #self.ax_seism1 = self.fig.add_axes(coords_ax, sharey = ax)
        self.ax_seism1 = self.fig.add_axes(coords_ax)
        self.ax_seism1.yaxis.tick_right()


        # Add file selector to the widget
        self.file_selector = FilesView(self.root_path_bind.value, parent=self.fileSelectorWidget_2,
                                       on_change_file_callback=lambda file_path: self.onChange_file(file_path))

        # Binds
        self.selectDirBtn_2.clicked.connect(lambda: self.on_click_select_directory(self.root_path_bind))

        # action
        self.plotBtn.clicked.connect(self.plot_seismogram)
        self.plot2Btn.clicked.connect(self.run_phase_vel)
        self.stationsBtn.clicked.connect(self.stations_info)
        self.macroBtn.clicked.connect(lambda: self.open_parameters_settings())

        # clicks #now is not compatible with collectors
        #self.canvas_plot1.on_double_click(self.on_click_matplotlib)
        #self.canvas_plot1.mpl_connect('key_press_event', self.key_pressed)

    def open_parameters_settings(self):
        self.parameters.show()

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def onChange_root_path(self, value):
        """
        Fired every time the root_path is changed

        :param value: The path of the new directory.

        :return:
        """
        self.file_selector.set_new_rootPath(value)


    def onChange_file(self, file_path):
        # Called every time user select a different file
        pass

    def on_click_select_directory(self, bind: BindPyqtObject):
        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value)
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', bind.value,
                                                           pw.QFileDialog.DontUseNativeDialog)
        if dir_path:
            bind.value = dir_path

    def on_click_select_file(self, bind: BindPyqtObject):
        selected = pw.QFileDialog.getOpenFileName(self, "Select metadata file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            bind.value = selected[0]

    def find_indices(self, lst, condition):

        return [i for i, elem in enumerate(lst) if condition(elem)]

    def find_nearest(self, a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return a.flat[idx], idx

    def stations_info(self):
        sd = []
        st = read(self.file_selector.file_path)
        tr = st[0]
        sd.append([tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel, tr.stats.starttime,
                       tr.stats.endtime, tr.stats.sampling_rate, tr.stats.npts])

        self._stations_info = StationsInfo(sd, check= False)
        self._stations_info.show()



    @property
    def trace(self):
        return ObspyUtil.get_tracer_from_file(self.file_selector.file_path)

    def get_data(self):
        parameters = self.parameters.getParameters()
        file = self.file_selector.file_path
        try:

            sd = SeismogramDataAdvanced(file_path = file)
            tr = sd.get_waveform_advanced(parameters, {}, filter_error_callback=self.filter_error_message)
            #tr = st[0]
            t = tr.times()

            return tr, t
        except:
            return []

    def convert_2_vel(self, tr):

        geodetic = tr.stats.mseed['geodetic']
        dist = geodetic[0]

        return dist


    #@AsycTime.run_async()
    def plot_seismogram(self):

        [tr1, t] = self.get_data()
        tr = tr1.copy()
        fs = tr1.stats.sampling_rate
        selection = self.time_frequencyCB.currentText()
        # take into acount causality

        if self.causalCB.currentText() == "Causal":
            starttime = tr.stats.starttime
            endtime = tr.stats.starttime+int(len(tr.data) / (2*fs))
            tr.trim(starttime=starttime,endtime=endtime)
            data = np.flip(tr.data)
            tr.data = data

        else:
            starttime = tr.stats.starttime +int(len(tr.data) / (2*fs))
            endtime =  tr.stats.endtime
            tr.trim(starttime=starttime, endtime=endtime)

        if self.phase_matchCB.isChecked():
            distance = tr.stats.mseed['geodetic'][0]
            ns = noise_processing(tr)
            tr_filtered = ns.phase_matched_filter(self.typeCB.currentText(),
                  self.phaseMacthmodelCB.currentText(), distance , filter_parameter = self.phaseMatchCB.value())
            tr.data = tr_filtered.data

        if selection == "Continuous Wavelet Transform":

            nf = self.atomsSB.value()
            f_min = 1 / self.period_max_cwtDB.value()
            f_max = 1/  self.period_min_cwtDB.value()
            wmin = self.wminSB.value()
            wmax = self.wminSB.value()
            npts = len(tr.data)
            t = np.linspace(0, tr.stats.delta * npts, npts)
            cw = ConvolveWaveletScipy(tr)
            wavelet=self.wavelet_typeCB.currentText()

            m = self.wavelets_param.value()

            cw.setup_wavelet(wmin=wmin, wmax=wmax, tt=int(fs/f_min), fmin=f_min, fmax=f_max, nf=nf,
                                 use_wavelet = wavelet, m = m, decimate=False)

            scalogram2 = cw.scalogram_in_dbs()
            phase, inst_freq, ins_freq_hz = cw.phase() # phase in radians
            inst_freq = ins_freq_hz
            #delay = cw.get_time_delay()
            x, y = np.meshgrid(t, np.logspace(np.log10(f_min), np.log10(f_max), scalogram2.shape[0]))
            #x = x + delay
            # chop cero division
            dist = self.convert_2_vel(tr)
            vel = (dist / (x[:, 1:] * 1000))
            min_time_idx = fs * (dist / (self.max_velDB.value() * 1000))
            min_time_idx = int(min_time_idx)
            max_time_idx = fs * (dist / (self.min_velDB.value() * 1000))
            max_time_idx = int(max_time_idx)
            period = 1 / y[:, 1:]
            scalogram2 = scalogram2[:, 1:]

            if self.ftCB.isChecked():
                min_vel, idx_min_vel = self.find_nearest(vel[0, :], self.min_velDB.value())
                max_vel, idx_max_vel = self.find_nearest(vel[0, :], self.max_velDB.value())
                self.min_vel = min_vel
                self.max_vel = max_vel
                vel = vel[:, idx_max_vel:idx_min_vel]
                period = period[:, idx_max_vel:idx_min_vel]
                scalogram2 = scalogram2[:, idx_max_vel:idx_min_vel]
                phase = phase[:, idx_max_vel:idx_min_vel]
                inst_freq = inst_freq[:, idx_max_vel:idx_min_vel]


            scalogram2 = np.clip(scalogram2, a_min=self.minlevelCB.value(), a_max=0)
            min_cwt= self.minlevelCB.value()
            max_cwt = 0

            #scalogram2 = scalogram2+0.01

            # flips
            scalogram2 = scalogram2.T
            scalogram2 = np.fliplr(scalogram2)
            scalogram2 = np.flipud(scalogram2)
            self.scalogram2 = scalogram2

            phase = phase.T
            phase = np.fliplr(phase)
            phase = np.flipud(phase)
            self.phase = phase

            inst_freq = inst_freq.T
            inst_freq = np.fliplr(inst_freq)
            inst_freq = np.flipud(inst_freq)
            self.inst_freq = inst_freq

            vel = vel.T
            vel = np.flipud(vel)
            self.vel = vel

            period = period.T
            period = np.fliplr(period)

            # extract ridge

            ridge = np.max(scalogram2, axis = 0)

            distance = self.dist_ridgDB.value()*vel.shape[0]/(max_vel-min_vel)
            height = (self.minlevelCB.value(),0)
            ridges, peaks, group_vel = self.find_ridges(scalogram2, vel, height, distance, self.numridgeSB.value())

            #print(ridges)
            #ridge_vel = []
            # for j in range(len(ridge)):
             #   value, idx = self.find_nearest(scalogram2[:,j],ridge[j])
             #    ridge_vel.append(vel[idx,j])

            self.t = dist/(1000*vel)
            self.dist = dist/1000
            # phase_vel = self.phase_vel(scalogram2, ridge, phase, inst_freq, t, dist/1000, n)
            # phase_vel = np.flipud(phase_vel)

            # Plot
            self.ax_seism1.cla()
            self.ax_seism1.plot(tr.data, tr.times() / tr.stats.sampling_rate, linewidth=0.5)
            self.ax_seism1.plot(tr.data[min_time_idx:max_time_idx],
                                tr.times()[min_time_idx:max_time_idx] / tr.stats.sampling_rate,
                                color='red', linewidth=0.5)

            self.ax_seism1.set_xlabel("Amplitude")
            self.ax_seism1.set_ylabel("Time (s)")
            self.canvas_plot1.clear()
            self.canvas_plot1.plot_contour(period, vel, scalogram2, axes_index=1, levels = 100, clabel="Power [dB]",
                        cmap=plt.get_cmap("jet"), vmin=min_cwt, vmax=max_cwt, antialiased=True, xscale = "log")

            self.canvas_plot1.set_xlabel(1, "Period (s)")
            self.canvas_plot1.set_ylabel(1, "Group Velocity (km/s)")

            # Plot ridges and create lasso selectors

            self.selectors_group_vel = []
            self.group_vel = group_vel
            self.periods = period[0, :]
            ax = self.canvas_plot1.get_axe(1)

            for k in range(self.numridgeSB.value()):
                #self.canvas_plot1.plot(period[0,:], group_vel[k], axes_index=1, marker=".", color = self.colors[k],
                 #                      clear_plot=False)

                pts = ax.scatter(self.periods, self.group_vel[k], c=self.colors[k], marker=".", s=60)
                self.selectors_group_vel.append(CollectionLassoSelector(ax, pts, [0.5, 0., 0.5, 1.]))


        if selection == "Hilbert-Multiband":

            f_min = 1 / self.period_max_mtDB.value()
            f_max = 1/  self.period_min_mtDB.value()

            npts = len(tr.data)
            t = np.linspace(0, tr.stats.delta * npts, npts)
            hg = hilbert_gauss(tr, f_min, f_max, self.freq_resDB.value())
            scalogram2, phase, inst_freq, inst_freq_hz, f = hg.compute_filter_bank()
            inst_freq = inst_freq_hz
            scalogram2 = hg.envelope_db()

            x, y = np.meshgrid(t, f[0:-1])

            # chop cero division
            dist = self.convert_2_vel(tr)
            vel = (dist / (x[:, 1:] * 1000))
            min_time_idx = fs * (dist / (self.max_velDB.value() * 1000))
            min_time_idx = int(min_time_idx)
            max_time_idx = fs * (dist / (self.min_velDB.value() * 1000))
            max_time_idx = int(max_time_idx)
            period = 1 / y[:, 1:]
            scalogram2 = scalogram2[:, 1:]

            if self.ftCB.isChecked():
                min_vel, idx_min_vel = self.find_nearest(vel[0, :], self.min_velDB.value())
                max_vel, idx_max_vel = self.find_nearest(vel[0, :], self.max_velDB.value())
                self.min_vel = min_vel
                self.max_vel = max_vel
                vel = vel[:, idx_max_vel:idx_min_vel]
                period = period[:, idx_max_vel:idx_min_vel]
                scalogram2 = scalogram2[:, idx_max_vel:idx_min_vel]
                phase = phase[:, idx_max_vel:idx_min_vel]
                inst_freq = inst_freq[:, idx_max_vel:idx_min_vel]

            scalogram2 = np.clip(scalogram2, a_min=self.minlevelCB.value(), a_max=0)
            min_cwt = self.minlevelCB.value()
            max_cwt = 0

            # flips
            scalogram2 = scalogram2.T
            scalogram2 = np.fliplr(scalogram2)
            scalogram2 = np.flipud(scalogram2)
            self.scalogram2 = scalogram2

            phase = phase.T
            phase = np.fliplr(phase)
            phase = np.flipud(phase)
            self.phase = phase

            inst_freq = inst_freq.T
            inst_freq = np.fliplr(inst_freq)
            inst_freq = np.flipud(inst_freq)
            self.inst_freq = inst_freq

            vel = vel.T
            vel = np.flipud(vel)
            self.vel = vel

            period = period.T
            period = np.fliplr(period)

            # extract ridge

            ridge = np.max(scalogram2, axis=0)

            distance = self.dist_ridgDB.value() * vel.shape[0] / (max_vel - min_vel)
            height = (self.minlevelCB.value(), 0)
            ridges, peaks, group_vel = self.find_ridges(scalogram2, vel, height, distance, self.numridgeSB.value())

            # print(ridges)
            # ridge_vel = []
            # for j in range(len(ridge)):
            #   value, idx = self.find_nearest(scalogram2[:,j],ridge[j])
            #    ridge_vel.append(vel[idx,j])

            self.t = dist / (1000 * vel)
            self.dist = dist / 1000
            # phase_vel = self.phase_vel(scalogram2, ridge, phase, inst_freq, t, dist/1000, n)
            # phase_vel = np.flipud(phase_vel)

            # Plot
            self.ax_seism1.cla()
            self.ax_seism1.plot(tr.data, tr.times() / tr.stats.sampling_rate, linewidth=0.5)
            self.ax_seism1.plot(tr.data[min_time_idx:max_time_idx],
                                tr.times()[min_time_idx:max_time_idx] / tr.stats.sampling_rate,
                                color='red', linewidth=0.5)

            self.ax_seism1.set_xlabel("Amplitude")
            self.ax_seism1.set_ylabel("Time (s)")
            self.canvas_plot1.clear()
            self.canvas_plot1.plot_contour(period, vel, scalogram2, axes_index=1, levels=100, clabel="Power [dB]",
                                           cmap=plt.get_cmap("jet"), vmin=min_cwt, vmax=max_cwt, antialiased=True,
                                           xscale="log")

            self.canvas_plot1.set_xlabel(1, "Period (s)")
            self.canvas_plot1.set_ylabel(1, "Group Velocity (km/s)")

            # TODO: duplicated with CWT, should be common
            # Plot ridges and create lasso selectors

            self.selectors_group_vel = []
            self.group_vel = group_vel
            self.periods = period[0, :]
            ax = self.canvas_plot1.get_axe(1)

            for k in range(self.numridgeSB.value()):
                # self.canvas_plot1.plot(period[0,:], group_vel[k], axes_index=1, marker=".", color = self.colors[k],
                #                      clear_plot=False)

                pts = ax.scatter(self.periods, self.group_vel[k], c=self.colors[k], marker=".", s=60)
                self.selectors_group_vel.append(CollectionLassoSelector(ax, pts, [0.5, 0., 0.5, 1.]))




    def run_phase_vel(self):

        phase_vel_array = self.phase_velocity()
        test = np.arange(-5, 5, 1) # natural ambiguity
        # Plot phase vel

        ax2 = self.canvas_plot1.get_axe(0)
        ax2.cla()
        self.selectors_phase_vel = []
        self.phase_vel = []
        for k in range(len(test)):
            pts = ax2.scatter(self.periods_now, phase_vel_array[k,:], marker=".", s = 60)
            self.selectors_phase_vel.append(CollectionLassoSelector(ax2, pts, [0.5, 0., 0.5, 1.]))
            ax2.set_xscale('log')


        if self.ftCB.isChecked():
          ax2.set_xlim(self.period_min_cwtDB.value(), self.period_max_cwtDB.value())
          ax2.set_ylim(self.min_vel, self.max_vel)
    #
        self.canvas_plot1.set_xlabel(0, "Period (s)")
        self.canvas_plot1.set_ylabel(0, "Phase Velocity (km/s)")



    def phase_velocity(self):
        self.periods_now = []
        self.solutions = []
        for i, selector in enumerate(self.selectors_group_vel):
            for idx in selector.ind:
                self.periods_now.append(self.periods[idx])
                self.solutions.append(self.group_vel[i, idx])

        landa = -1*np.pi/4
        phase_vel_array = np.zeros([len(np.arange(-5, 5, 1)), len(self.solutions)])
        for k in np.arange(-5, 5, 1):
            for j in range(len(self.solutions)):
                value_period, idx_period = self.find_nearest(self.periods, self.periods_now[j])
                value_group_vel, idx_group_vel = self.find_nearest(self.vel[:,0], self.solutions[j])
                to = self.t[idx_group_vel , 0]
                phase_test = self.phase[idx_group_vel, idx_period]
                inst_freq_test = self.inst_freq[idx_group_vel, idx_period]
                phase_vel_num = self.dist * inst_freq_test
                phase_vel_den = phase_test+inst_freq_test*to-(np.pi/4)-k*2*np.pi+landa
                phase_vel_array[k, j] = phase_vel_num / phase_vel_den


        return phase_vel_array


    def find_ridges(self, scalogram2, vel, height, distance, num_ridges):

        distance = int(distance)
        dim = scalogram2.shape[1]
        ridges = np.zeros([num_ridges, dim])
        peak = np.zeros([num_ridges, dim])
        group_vel = np.zeros([num_ridges, dim])

        for j in range(dim):

            peaks, properties = find_peaks(scalogram2[:,j], height = height, threshold=-5, distance = distance)

            for k in range(num_ridges):

                try:
                    if len(peaks)>0:
                        ridges[k, j] = peaks[k]

                        peak[k, j] =  properties['peak_heights'][k]
                        group_vel[k,j] =vel[int(peaks[k]),0]
                    else:
                        ridges[k, j] = "NaN"
                        peak[k, j] = "NaN"
                        group_vel[k, j] = "NaN"

                except:

                    ridges[k, j] = "NaN"
                    peak[k, j] = "NaN"
                    group_vel[k, j] = "NaN"


        return ridges, peak, group_vel


    def on_click_matplotlib(self, event, canvas):
        if isinstance(canvas, MatplotlibCanvas):

            x1_value, y1_value = event.xdata, event.ydata

            period, pick_vel,_ = self.find_pos(x1_value, y1_value)
            self.solutions.append(pick_vel)
            self.periods_now.append(period)
            self.canvas_plot1.plot(period, pick_vel, color = "purple", axes_index = 1, clear_plot = False, marker = "." )

    def find_pos(self, x1, y1):

        value_period, idx_periods = self.find_nearest(self.periods, x1)
        dim = self.group_vel.shape[0]
        rms = []

        for k in range(dim):
            group_vel_test = self.group_vel[k, :][idx_periods]
            err = abs(group_vel_test - y1)
            if err > 0:
                rms.append(err)
            else:
                err = 100
                rms.append(err)

        rms = np.array(rms)
        idx = np.argmin(rms)
        return value_period, self.group_vel[idx, idx_periods], idx

    def key_pressed(self, event):

        if event.key == 'r':
            x1_value, y1_value = event.xdata, event.ydata
            print(x1_value, y1_value)
            period, pick_vel,idx = self.find_pos(x1_value, y1_value)
            # check if is in solutions
            if period in self.periods_now and pick_vel in self.solutions:
                self.periods_now.remove(period)
                self.solutions.remove(pick_vel)
                self.canvas_plot1.plot(period, pick_vel, color=self.colors[idx], axes_index=1, clear_plot=False, marker=".")


    # if selection == "Multitaper Spectrogram":
    #
    #     win = int(self.mt_window_lengthDB.value() * tr.stats.sampling_rate)
    #     win_half = int(win / (2 * fs))
    #     tbp = self.time_bandwidth_DB.value()
    #     ntapers = self.number_tapers_mtSB.value()
    #     f_min = 1 / self.period_max_mtDB.value()
    #     f_max = 1 / self.period_min_mtDB.value()
    #     mtspectrogram = MTspectrogram(self.file_selector.file_path, win, tbp, ntapers, f_min, f_max)
    #     x, y, log_spectrogram = mtspectrogram.compute_spectrogram(tr)
    #     # x in seconds, y in freqs
    #     x = x + win_half
    #     # chop cero division
    #     dist = self.convert_2_vel(tr)
    #     vel = (dist / (x[:, 1:] * 1000))
    #     min_time_idx = fs * (dist / (self.max_velDB.value() * 1000))
    #     min_time_idx = int(min_time_idx)
    #     max_time_idx = fs * (dist / (self.min_velDB.value() * 1000))
    #     max_time_idx = int(max_time_idx)
    #     period = 1 / y[:, 1:]
    #     log_spectrogram = log_spectrogram[:, 1:]
    #
    #     if self.ftCB.isChecked():
    #         min_vel, idx_min_vel = self.find_nearest(vel[0, :], self.min_velDB.value())
    #         max_vel, idx_max_vel = self.find_nearest(vel[0, :], self.max_velDB.value())
    #         vel = vel[:, idx_max_vel:idx_min_vel]
    #         period = period[:, idx_max_vel:idx_min_vel]
    #         log_spectrogram = log_spectrogram[:, idx_max_vel:idx_min_vel]
    #
    #     log_spectrogram = np.clip(log_spectrogram, a_min=self.minlevelCB.value(), a_max=0.05)
    #     min_log_spectrogram = self.minlevelCB.value()
    #     max_log_spectrogram = 0.05
    #     log_spectrogram = log_spectrogram + 0.05
    #
    #     # flips
    #     log_spectrogram = log_spectrogram.T
    #     log_spectrogram = np.fliplr(log_spectrogram)
    #     log_spectrogram = np.flipud(log_spectrogram)
    #
    #     vel = vel.T
    #     vel = np.flipud(vel)
    #
    #     period = period.T
    #     period = np.fliplr(period)
    #
    #     # Plot
    #     self.ax_seism1.cla()
    #     self.ax_seism1.plot(tr.data, tr.times() / tr.stats.sampling_rate, linewidth=0.5)
    #     self.ax_seism1.plot(tr.data[min_time_idx:max_time_idx],
    #                         tr.times()[min_time_idx:max_time_idx] / tr.stats.sampling_rate,
    #                         color='red', linewidth=0.5)
    #
    #     self.ax_seism1.set_xlabel("Amplitude")
    #     self.ax_seism1.set_ylabel("Time (s)")
    #     self.canvas_plot1.clear()
    #     self.canvas_plot1.plot_contour(period, vel, log_spectrogram, axes_index=0, clabel="Power [dB]",
    #                                    cmap=plt.get_cmap("jet"), vmin=min_log_spectrogram, vmax=max_log_spectrogram,
    #                                    antialiased=True, xscale="log")
    #
    #     # ax = self.canvas_plot1.get_axe(0)
    #     # ax.clabel(cs, levels = levels, fmt = '%2.1', colors = 'k', fontsize = 12)
    #     self.canvas_plot1.set_xlabel(0, "Period (s)")
    #     self.canvas_plot1.set_ylabel(0, "Group Velocity (m/s)")