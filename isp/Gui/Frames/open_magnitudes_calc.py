# import pandas as pd
# from isp.DataProcessing.autoag import Automag
from obspy.core.event import Origin
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy import UTCDateTime
from isp.Gui import pw
from isp.Gui.Frames import MatplotlibCanvas, SettingsLoader, MessageDialog
from isp.Gui.Frames.uis_frames import UiMagnitudeFrame
import nitime.algorithms as tsa
import numpy as np
import matplotlib.dates as mdt
from isp.Gui.Utils.pyqt_utils import add_save_load, set_qdatetime
from isp.Structures.structures import StationCoordinates
from isp import ROOT_DIR, LOCATION_OUTPUT_PATH
import os
from isp.Utils import ObspyUtil
from isp.mti import MTIManager
from scipy.optimize import curve_fit


@add_save_load()
class MagnitudeCalc(pw.QFrame, UiMagnitudeFrame, metaclass=SettingsLoader):
    def __init__(self, option: str, inventory, project, chop):
        super(MagnitudeCalc, self).__init__()
        self.setupUi(self)
        self.spectrum_Widget_Canvas = MatplotlibCanvas(self.spectrumWidget, nrows=2, ncols=1,
                                                       sharex=False, constrained_layout=True)

        self.chop = chop
        self.inventory = inventory
        self.project = project
        self.runBtn.clicked.connect(self.run_magnitudes)
        self.saveBtn.clicked.connect(self.save_results)
        self.plotBtn.clicked.connect(self.plot_comparison)

        #self.set_default_btn.clicked.connect(self.set_default)
        self.Mw = []
        self.Mw_std = []
        self.Ms = []
        self.Ms_std = []
        self.Mb = []
        self.Mb_std = []
        self.ML = []
        self.ML_std = []
        self.Mc = []
        self.Mc_std = []
        self.ML = []
        self.ML_std = []
        self.Magnitude_Ws = []
        self.Mss = []
        self.Mcs = []
        self.Mcs = []
        self.MLs = []
        self.get_hypocenter(option)

    def on_click_select_hyp_file(self):
        file_path = pw.QFileDialog.getOpenFileName(self, 'Select Directory', ROOT_DIR)
        file_path = file_path[0]
        if isinstance(file_path, str):
            return file_path
        else:
            md = MessageDialog(self)
            md.set_info_message("No selected any file, please set hypocenter parameters manually")
            return None

    def get_coordinates_from_metadata(self, inventory, stats):
        selected_inv = inventory.select(network=stats[0], station=stats[1], channel=stats[3],
                                        starttime=stats[4], endtime=stats[5])
        cont = selected_inv.get_contents()
        coords = selected_inv.get_coordinates(cont['channels'][0])
        return StationCoordinates.from_dict(coords)

    def get_hypocenter(self, option):

        # if option == "manually":
        #
        #     md = MessageDialog(self)
        #     md.set_info_message("Loaded Metadata, please set hypocenter parameters by yourself, "
        #                         "then click stations info")

        if option == "last":
            hyp_file = os.path.join(LOCATION_OUTPUT_PATH, "last.hyp")
            origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file, modified=True)
            hyp_values = MTIManager.get_hyp_values(origin[0])
            self.event = origin[0]
            self.__set_hyp(hyp_values)
            md = MessageDialog(self)
            md.set_info_message("Loaded information and set hypocenter parameters, please click stations info")

        elif option == "other":
            hyp_file = self.on_click_select_hyp_file()
            if isinstance(hyp_file, str):
                origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file, modified=True)
                hyp_values = MTIManager.get_hyp_values(origin[0])
                self.event = origin[0]
                self.__set_hyp(hyp_values)
                md = MessageDialog(self)
                md.set_info_message("Loaded information and set hypocenter parameters, please click stations info")

    def __set_hyp(self, hyp_values):
        self.latitudeDB.setValue(hyp_values["latitude"])
        self.longitudeDB.setValue(hyp_values["longitude"])
        self.depthSB.setValue(hyp_values["depth"])
        set_qdatetime(hyp_values['origin_time'], self.origin_time)

    def load_event(self):
        hyp_file = self.on_click_select_hyp_file()
        if isinstance(hyp_file, str):
            origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file, modified=True)
            hyp_values = MTIManager.get_hyp_values(origin[0])
            self.__set_hyp(hyp_values)
            md = MessageDialog(self)
            md.set_info_message("Loaded information and set hypocenter parameters, please click stations info")


    def closeEvent(self, ce):
        self.save_values()
        
    def __load__(self):
        self.load_values()

    def fit_spectrum(self, spectrum, frequencies, traveltime, initial_omega_0, initial_f_c, QUALITY_FACTOR):

        """
            Fit a theoretical source spectrum to a measured source spectrum.

            Uses a Levenburg-Marquardt algorithm.

            :param spectrum: The measured source spectrum.
            :param frequencies: The corresponding frequencies.
            :para traveltime: Event traveltime in [s].
            :param initial_omega_0: Initial guess for Omega_0.
            :param initial_f_c: Initial guess for the corner frequency.


            :returns: Best fits and standard deviations.
                (Omega_0, f_c, Omega_0_std, f_c_std)
                Returns None, if the fit failed.
            """

        # Apply frequency range filtering if specified
        # if min_frequency is not None and max_frequency is not None:
        #     # Find indices corresponding to the desired frequency range
        #     start_idx = np.searchsorted(frequencies, min_frequency, side='left')
        #     end_idx = np.searchsorted(frequencies, max_frequency, side='right')
        #
        #     # Chop the arrays
        #     frequencies = frequencies[start_idx:end_idx]
        #     spectrum = spectrum[start_idx:end_idx]

        def f(frequencies, omega_0, f_c):
            return self.calculate_source_spectrum(frequencies, omega_0, f_c,
                                             QUALITY_FACTOR, traveltime)

        popt, pcov = curve_fit(f, frequencies, spectrum, p0=list([initial_omega_0, initial_f_c]), maxfev=100000)
        if popt is None:
            return None
        return popt[0], popt[1], pcov[0, 0], pcov[1, 1]
    def calculate_source_spectrum(self, frequencies, omega_0, corner_frequency, Q, traveltime):
        """
        After Abercrombie (1995) and Boatwright (1980).

        Abercrombie, R. E. (1995). Earthquake locations using single-station deep
        borehole recordings: Implications for microseismicity on the San Andreas
        fault in southern California. Journal of Geophysical Research, 100,
        24003–24013.

        Boatwright, J. (1980). A spectral theory for circular seismic sources,
        simple estimates of source dimension, dynamic stress drop, and radiated
        energy. Bulletin of the Seismological Society of America, 70(1).

        The used formula is:
            Omega(f) = (Omege(0) * e^(-pi * f * T / Q)) / (1 + (f/f_c)^4) ^ 0.5

        :param frequencies: Input array to perform the calculation on.
        :param omega_0: Low frequency amplitude in [meter x second].
        :param corner_frequency: Corner frequency in [Hz].
        :param Q: Quality factor.
        :param traveltime: Traveltime in [s].
        """
        # Validate inputs

        # if Q <= 0:
        #     raise ValueError("Quality factor Q must be positive.")
        # if omega_0 < 0 or corner_frequency <= 0:
        #     raise ValueError("Omega_0 and corner frequency must be positive.")

        # Compute numerator and denominator for the source spectrum formula
        num = omega_0 * np.exp(-np.pi * frequencies * traveltime / Q)
        denom = np.sqrt(1 + (frequencies / corner_frequency) ** 4)
        return num / denom

    def moment_magnitude(self):
        density = self.densityDB.value()
        vp = self.vpDB.value() * 1000
        vs = self.vsDB.value() * 1000

        if self.phaseCB.currentText() == 'P-Wave':
            radiation_pattern = 0.52
            velocity = vp
            k = 0.32
        elif self.phaseCB.currentText() == "S-Wave":
            radiation_pattern = 0.63
            velocity = vs
            k = 0.21

        Mws = []
        Mws_std = []
        Magnitude_Ws = []
        origin_time = self.event.time
        moments = []
        source_radii = []
        corner_freqs = []

        self.spectrum_Widget_Canvas.plot([], [], 0)
        ax1 = self.spectrum_Widget_Canvas.get_axe(0)
        ax1.cla()

        for key in self.chop['Body waves']:

            magnitude_dict = self.chop['Body waves'][key]
            fs = magnitude_dict[0][6]
            # Calculate distance
            coords = self.get_coordinates_from_metadata(self.inventory, magnitude_dict[0])
            dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event.latitude,self.event.longitude)
            pick_time = UTCDateTime(mdt.num2date(magnitude_dict[1][0]))
            data = np.array(magnitude_dict[2])

            # delta = 1 / magnitude_dict[0][6]

            # Calculate the spectrum.
            data = data - np.mean(data)
            freq, spec, _ = tsa.multi_taper_psd(data, fs, adaptive=True, jackknife=False, low_bias=True)
            spec = spec[1:]
            freq = freq[1:]

            # spec = np.fft.fft(data)
            # amplitude_spectrum = (2 / len(data)) * np.abs(spec[:len(data) // 2])
            # freq = np.fft.fftfreq(len(data), d=1 / fs)
            # freq = freq[:len(data) // 2]
            #
            # # needs to remove division by ceroç
            # amplitude_spectrum = amplitude_spectrum[1:]
            # freq = freq[1:]
            # spec = amplitude_spectrum / (2 * np.pi * freq)
            # go to amplitude and displacement
            spec = np.sqrt(spec)/(2*np.pi*freq)

            # Plot spectrum

            ax1.loglog(freq, spec, '0.1', linewidth=0.5, color='black', alpha=0.5)
            ax1.set_ylim(np.min(spec) / 10.0, np.max(spec) * 100.0)
            #ax1.set_xlim(freq[1], freq[len(freq) - 1])
            ax1.set_ylabel('Amplitude')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.grid(True, which="both", ls="-", color='grey')
            ax1.legend()

            # Finish Plot
            tt = pick_time-origin_time
            #print("Distance",dist,"Vp",vp,"Q",quality,"Density",density, "Travel Time", tt)

            # Q as a function of freq
            Qo = self.qualityDB.value()
            a = self.coeffDB.value()
            quality = (Qo) * (freq) ** a
            #min_freq = 0.5
            #max_freq = int((fs/2)-(0.1*fs/2))
            min_freq = None
            max_freq = None

            try:
                fit = self.fit_spectrum(spec, freq, tt, np.max(spec), 10.0, quality)
            except:
                continue

            if fit is None:
                continue

            Omega_0, f_c, err, _ = fit
            if f_c < 0:
                f_c = -1*f_c
            # Second Plot

            theoretical_spectrum = self.calculate_source_spectrum(freq, Omega_0, f_c, quality, tt)
            ax1.loglog(freq, theoretical_spectrum, color="red", alpha=0.5, lw=0.5)
            #ax1.set_xlim(freq[1], freq[len(freq) - 1])
            ax1.set_ylabel('Amplitude')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.grid(True, which="both", ls="-", color='grey')

            r = 3 * k * vs / f_c

            M_0 = 4.0 * np.pi * density * velocity ** 3 * dist * np.sqrt(Omega_0 ** 2) / radiation_pattern
            Magnitude_Ws.append(2.0 / 3.0 * (np.log10(M_0) - 9.1))

            corner_freqs.append(f_c)
            moments.append(M_0)
            source_radii.append(r)


        # Calculate the seismic moment via basic statistics.
        moments = np.array(moments)

        moment = moments.mean()
        moment_std = moments.std()

        corner_frequency = np.mean(corner_freqs)
        #corner_frequency_std = np.std(corner_freqs - corner_frequency)

        # Calculate the source radius.
        self.source_radius = np.mean(source_radii)
        source_radius_std = np.std(source_radii - self.source_radius)

        # Calculate the stress drop of the event based on the average moment and
        # source radii.
        stress_drop = (7 * moment) / (16 * self.source_radius ** 3)
        stress_drop_std = np.sqrt((stress_drop ** 2) * \
                                  (((moment_std ** 2) / (moment ** 2)) + \
                                   (9 * self.source_radius * source_radius_std ** 2)))
        if self.source_radius > 0 and source_radius_std < self.source_radius:
            print("Source radius:", self.source_radius, " Std:", source_radius_std)
            print("Stress drop:", stress_drop / 1E5, " Std:", stress_drop_std / 1E5)

        self.stress_drop = stress_drop
        self.Magnitude_Ws = Magnitude_Ws
        Mw = 2.0 / 3.0 * (np.log10(moment) - 9.1)
        Mw_std = 2.0 / 3.0 * moment_std / (moment * np.log(10))
        Mws_std.append(Mw_std)
        Mws.append(Mw)
        print("Moment Magnitude", Mws, "Moment Magnitude deviation", Mws_std)
        label = "Mw"
        self.plot_histograms(Magnitude_Ws, label)
        self.Mw = Mw
        self.Mw_std = Mw_std

    def magnitude_surface(self):
        Ms = []
        for key in self.chop['Surf Waves']:
            magnitude_dict = self.chop['Surf Waves'][key]

            coords = self.get_coordinates_from_metadata(self.inventory, magnitude_dict[0])

            dist= locations2degrees(coords.Latitude, coords.Longitude, self.event.latitude,self.event.longitude)
            data = np.array(magnitude_dict[2])*1e6 # convert to nm
            Ms_value=np.log10(np.max(data)/(2*np.pi))+1.66*np.log10(dist)+3.3
            Ms.append(Ms_value)

        Ms=np.array(Ms)
        self.Mss=Ms
        Ms_mean = Ms.mean()
        Ms_deviation =Ms.std()
        print("Surface Magnitude", Ms_mean, "Variance", Ms_deviation)
        label = "Ms"
        self.plot_histograms(Ms,label)
        self.Ms = Ms_mean
        self.Ms_std = Ms_deviation

    def magnitude_body(self):
        Mb = []
        path_to_atenuation_file = os.path.join(ROOT_DIR, "earthquakeAnalysis", "magnitude_atenuation", "atenuation.txt")
        x, y = np.loadtxt(path_to_atenuation_file, skiprows=1, unpack=True)

        for key in self.chop['Body waves']:
            magnitude_dict = self.chop['Body waves'][key]
            coords = self.get_coordinates_from_metadata(self.inventory, magnitude_dict[0])
            dist = locations2degrees(coords.Latitude, coords.Longitude, self.event.latitude, self.event.longitude)
            dist = np.floor(dist)
            if dist < 16:
                print("Epicentral Distance ", dist," is less than 16 degrees, Body-wave Magnitude couldn't be calculated")
            else:
                loc = np.where(x == dist)[0][0]
                atenuation = y[loc]
                data = np.array(magnitude_dict[2]) * 1e6  # convert to nm
                Mb_value = (np.log10(np.max(data)) / (2 * np.pi)) + atenuation
                Mb.append(Mb_value)
                Mbs = np.array(Mb)
                Mb_mean = Mbs.mean()
                Mb_deviation = Mb.std()
                self.Mb = Mb_mean
                self.Mb_std = Mb_deviation
                print("Body Magnitude", Mb_mean, "Variance", Mb_deviation)
        label="Mb"
        self.Mbs = Mbs
        if len(Mb)>0:
            self.plot_histograms(Mb,label)

    def magnitude_coda(self):
        Mc = []
        # values for california
        a = 2.0
        b = 0.0035
        c= -0.87
        # 
        for key in self.chop['Coda']:
            magnitude_dict = self.chop['Coda'][key]
            coords = self.get_coordinates_from_metadata(self.inventory, magnitude_dict[0])
            dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event.latitude, self.event.longitude)
            dist = dist / 1000
            #data = np.array(magnitude_dict[2])
            pick_time = UTCDateTime(mdt.num2date(magnitude_dict[1][0]))
            end_time = UTCDateTime(mdt.num2date(magnitude_dict[1][-1]))
            t_coda = end_time-pick_time
            Mc_value = a*np.log10(t_coda)+b*dist+c
            Mc_value = Mc_value
            Mc.append(Mc_value)
            Mcs=np.array(Mc)
            Mc_mean = Mcs.mean()
            Mc_deviation = Mcs.std()

        print("Coda Magnitude", Mc_mean, "Variance", Mc_deviation)
        label="Mc"
        self.plot_histograms(Mc, label)
        self.Mcs = Mcs
        self.Mc = Mc_mean
        self.Mc_std = Mc_deviation

    def magnitude_local(self):
        ML = []
        for key in self.chop['Body waves']:
            magnitude_dict = self.chop['Body waves'][key]
            coords = self.get_coordinates_from_metadata(self.inventory, magnitude_dict[0])
            dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event.latitude, self.event.longitude)
            dist = dist/1000
            data = np.array(magnitude_dict[2])   # already converted Wood Anderson (Gain in mm 2800 +-60)
            max_amplitude = np.max(data)*1e3 # convert to  mm --> nm
            ML_value = np.log10(max_amplitude)+1.11*np.log10(dist)+0.00189*dist-2.09
            ML.append(ML_value)
            MLs =np.array(ML)
            ML_mean = MLs.mean()
            ML_deviation = MLs.std()
        print("Local Magnitude", ML_mean, "Variance", ML_deviation)
        label="ML"
        self.MLs = MLs
        self.plot_histograms(ML,label)
        self.ML = ML_mean
        self.ML_std = ML_deviation


    def run_magnitudes(self):
        self.spectrum_Widget_Canvas.plot([], [], 1)
        self.ax2 = self.spectrum_Widget_Canvas.get_axe(1)
        self.ax2.cla()

        if self.local_magnitudeRB.isChecked():
            print("Calculating Local Magnitude")
            try:
             self.magnitude_local()
            except:
                pass

        if self.magnitudesRB.isChecked():
            try:
                if self.body_waveCB.isChecked():
                    print("Calculating Body-Wave Magnitude")
                    self.magnitude_body()
            except:
                pass
            try:
                if self.surface_waveCB.isChecked():
                    print("Calculating Surface-Wave Magnitude")
                    self.magnitude_surface()
            except:
                pass
            try:
                if self.moment_magnitudeCB.isChecked():
                    print("Calculating Moment Magnitude")
                    self.moment_magnitude()
            except:
                pass

            try:
                if self.coda_magnitudeCB.isChecked():
                    print("Coda Moment Magnitude")
                    self.magnitude_coda()
            except:
                    pass

        self.print_magnitudes_results()

    def print_magnitudes_results(self):
        self.magnitudesText.setPlainText(" Magnitudes Estimation Results ")
        try:
            if self.Mw:
                self.magnitudesText.appendPlainText("Moment Magnitude: " " Mw {Mw:.3f} " 
                " std {std:.3f} " "Source Radious {source:.3f} " "Stress Drop {stress:.3E} ".format(Mw=self.Mw,
                std=self.Mw_std, source = self.source_radius/1000, stress = self.stress_drop))

        except:
            pass

        try:
            if self.Ms:
                self.magnitudesText.appendPlainText("Surface-Wave Magnitude: " " Ms {Ms:.3f} " 
                                                        " std {std:.3f} ".format(Ms=self.Ms,std=self.Ms_std))
        except:
            pass
        try:
            if self.Mb:
                self.magnitudesText.appendPlainText("Body-Wave Magnitude: " " Mb {Mb:.3f} "
                                                    " std {std:.3f} ".format(Mb=self.Mb, std=self.Mb_std))
        except:
            pass
        try:
            if self.Mc:
                self.magnitudesText.appendPlainText("Coda Magnitude: " " Mc {Mc:.3f} "
                                                    " std {std:.3f} ".format(Mc=self.Mc, std=self.Mc_std))
        except:
            pass
        try:
            if self.ML:
                self.magnitudesText.appendPlainText("Coda Magnitude: " " ML {ML:.3f} "
                                                    " std {std:.3f} ".format(ML=self.ML, std=self.ML_std))
        except:
            pass


    def save_results(self):
        import pandas as pd
        path_output =os.path.join(ROOT_DIR,"earthquakeAnalisysis","location_output","loc","magnitudes_output.mag")
        Magnitude_results = { 'Magnitudes': ["Mw", "Mw_std", "Ms", "Ms_std", "Mb",
          "Mb_std","Mc","Mc_std", "ML","ML_std"],'results':[self.Mw, self.Mw_std,self.Ms,self.Ms_std,self.Mb,
            self.Mb_std, self.Mc,self.Mc_std,self.ML,self.ML_std]}
        df = pd.DataFrame(Magnitude_results, columns=['Magnitudes','results'])
        print(df)
        df.to_csv(path_output, sep=' ', index=False)

    def plot_histograms(self, magnitudes, label):

        self.ax2.hist(magnitudes, bins=4*len(magnitudes), alpha=0.5, label=label)
        self.ax2.set_xlabel("Magnitude", size=12)
        self.ax2.set_ylabel("Count", size=12)
        self.ax2.legend(loc='upper right')

    def plot_comparison(self):
        import matplotlib.pyplot as plt
        from isp.Gui.Frames import MatplotlibFrame
        list_magnitudes = []
        labels = []
        if len(self.Magnitude_Ws)>=0:
            list_magnitudes.append(self.Magnitude_Ws)
            labels.append("Mw")
        if len(self.Mss) >= 0:
            list_magnitudes.append(self.Mss)
            labels.append("Ms")
        if len(self.Mcs) >= 0:
            list_magnitudes.append(self.Mcs)
            labels.append("Mc")
        if len(self.MLs) >= 0:
            labels.append("ML")
            list_magnitudes.append(self.MLs)
        fig, ax1 = plt.subplots(figsize=(6, 6))
        self.mpf = MatplotlibFrame(fig)
        k = 0
        for magnitude in list_magnitudes:
            label = labels[k]
            x = np.arange(len(magnitude))+1
            ax1.scatter(x, magnitude, s=15, alpha=0.5, label=label)
            ax1.tick_params(direction='in', labelsize=10)
            ax1.legend(loc='upper right')
            plt.ylabel('Magnitudes', fontsize=10)
            plt.xlabel('Counts', fontsize=10)
            k = k+1

        self.mpf.show()

########### AutoMag#########

    # def run_automag(self):
    #     magnitude_mw_statistics = ""
    #     magnitude_ml_statistics = ""
    #     self.load_config_automag()
    #     mg = Automag(self.origin, self.event, self.project, self.inventory)
    #     mg.scan_from_origin(self.origin)
    #     magnitude_mw_statistics, magnitude_ml_statistics = mg.estimate_magnitudes(self.config_automag)
    #     self.print_automag_results(magnitude_mw_statistics, magnitude_ml_statistics)
    # def load_config_automag(self):
    #     try:
    #         self.config_automag = pd.read_pickle(self.file_automag_config)
    #         self.modify_pred_config()
    #     except:
    #         md = MessageDialog(self)
    #         md.set_error_message("Coundn't open magnitude file")
    #
    #
    # def modify_pred_config(self):
    #
    #     self.config_automag["max_epi_dist"] = self.mag_max_distDB.value()
    #
    #     if self.mag_max_distDB.value() < 700:
    #         self.config_automag["scale"] = "Regional"
    #     else:
    #         self.config_automag["scale"] = "Teleseism"
    #
    #     self.config_automag["mag_vpweight"] = self.mag_vpweightDB.value()
    #     self.config_automag["rho"] = self.automag_density_DB.value()
    #     self.config_automag["automag_rpp"] = self.automag_rppDB.value()
    #     self.config_automag["automag_rps"] = self.automag_rpsDB.value()
    #
    #     if self.r_power_nRB.isChecked():
    #         self.config_automag["geom_spread_model"] = "r_power_n"
    #     else:
    #         self.config_automag["geom_spread_model"] = "boatwright"
    #     self.config_automag["geom_spread_n_exponent"] = self.geom_spread_n_exponentDB.value()
    #     self.config_automag["geom_spread_cutoff_distance"] = self.geom_spread_cutoff_distanceDB.value()
    #     self.config_automag["a_local_magnitude"] = self.mag_aDB.value()
    #     self.config_automag["b_local_magnitude"] = self.mag_bDB.value()
    #     self.config_automag["c_local_magnitude"] = self.mag_cDB.value()
    #     self.config_automag["win_length"] = self.win_lengthDB.value()
    #
    # def print_automag_results(self, magnitude_mw_statistics, magnitude_ml_statistics):
    #
    #
    #     Mw = magnitude_mw_statistics.summary_spectral_parameters.Mw.weighted_mean.value
    #     Mw_std = magnitude_mw_statistics.summary_spectral_parameters.Mw.weighted_mean.uncertainty
    #
    #     Mo = magnitude_mw_statistics.summary_spectral_parameters.Mo.mean.value
    #     Mo_units = magnitude_mw_statistics.summary_spectral_parameters.Mo.units
    #
    #     fc = magnitude_mw_statistics.summary_spectral_parameters.fc.weighted_mean.value
    #     fc_units = "Hz"
    #
    #     t_star = magnitude_mw_statistics.summary_spectral_parameters.t_star.weighted_mean.value
    #     t_star_std = magnitude_mw_statistics.summary_spectral_parameters.t_star.weighted_mean.uncertainty
    #     t_star_units = magnitude_mw_statistics.summary_spectral_parameters.t_star.units
    #
    #     source_radius = magnitude_mw_statistics.summary_spectral_parameters.radius.mean.value
    #     radius_units = magnitude_mw_statistics.summary_spectral_parameters.radius.units
    #
    #     bsd = magnitude_mw_statistics.summary_spectral_parameters.bsd.mean.value
    #     bsd_units = magnitude_mw_statistics.summary_spectral_parameters.bsd.units
    #
    #     Qo =  magnitude_mw_statistics.summary_spectral_parameters.Qo.mean.value
    #     Qo_std = magnitude_mw_statistics.summary_spectral_parameters.Qo.mean.uncertainty
    #     Qo_units = magnitude_mw_statistics.summary_spectral_parameters.Qo.units
    #
    #     Er = magnitude_mw_statistics.summary_spectral_parameters.Er.mean.value
    #     Er_units = "jul"
    #
    #     ML = magnitude_ml_statistics["ML_mean"]
    #     ML_std = magnitude_ml_statistics["ML_std"]
    #
    #     self.automagnitudesText.clear()
    #     self.automagnitudesText.appendPlainText("Moment Magnitude: " " Mw {Mw:.3f} "
    #                                             " std {std:.3f} ".format(Mw=Mw, std=Mw_std))
    #
    #     self.automagnitudesText.appendPlainText("Seismic Moment and Source radius: " " Mo {Mo:e} Nm"
    #                                        ", R {std:.3f} km".format(Mo=Mo, std=source_radius/1000))
    #
    #     self.automagnitudesText.appendPlainText("Brune stress Drop: " "{bsd:.3f} MPa".format(bsd=bsd))
    #
    #     self.automagnitudesText.appendPlainText("Quality factor: " " Qo {Qo:.3f} " " Q_std {Qo_std:.3f} ".format(Qo=Qo, Qo_std=Qo_std))
    #
    #     self.automagnitudesText.appendPlainText(
    #         "t_star: " "{t_star:.3f} s" " t_star_std {t_star_std:.3f} ".format(t_star=t_star, t_star_std=t_star_std))
    #
    #     self.automagnitudesText.appendPlainText("Local Magnitude: " " ML {ML:.3f} "
    #                                             " ML_std {std:.3f} ".format(ML=ML, std=ML_std))
    #
    # def get_rad_pattern_const(self, enable):
    #     self.automag_rppDB.setEnabled(enable)
    #     self.automag_rpsDB.setEnabled(enable)
    #
    # def get_rad_pattern_focmec(self, enable):
    #     self.automag_strikeDB.setEnabled(enable)
    #     self.automag_dipDB.setEnabled(enable)
    #     self.automag_rakeDB.setEnabled(enable)



