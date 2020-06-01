from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy import UTCDateTime, Trace
from isp.Gui import pw
from isp.Gui.Frames import MatplotlibCanvas
from isp.Gui.Frames.uis_frames import UiMagnitudeFrame
from mtspec import mtspec
import numpy as np
import scipy
import scipy.optimize
import matplotlib.dates as mdt
from isp.Gui.Utils.pyqt_utils import add_save_load
from isp.Structures.structures import StationCoordinates
from isp import ROOT_DIR
import os

def fit_spectrum(spectrum, frequencies, traveltime, initial_omega_0, initial_f_c, QUALITY_FACTOR):
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
    def f(frequencies, omega_0, f_c):
        return calculate_source_spectrum(frequencies, omega_0, f_c,
                QUALITY_FACTOR, traveltime)
    popt, pcov = scipy.optimize.curve_fit(f, frequencies, spectrum, \
        p0=list([initial_omega_0, initial_f_c]), maxfev=100000)
    if popt is None:
        return None
    return popt[0], popt[1], pcov[0, 0], pcov[1, 1]


def calculate_source_spectrum(frequencies, omega_0, corner_frequency, Q, traveltime):
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
    num = omega_0 * np.exp(-np.pi * frequencies * traveltime / Q)
    denom = (1 + (frequencies / corner_frequency) ** 4) ** 0.5
    return num / denom


def get_coordinates_from_metadata(inventory, stats):
    selected_inv = inventory.select(network=stats[0], station=stats[1], channel=stats[3],
                                    starttime=stats[4], endtime=stats[5])
    cont = selected_inv.get_contents()
    coords = selected_inv.get_coordinates(cont['channels'][0])
    return StationCoordinates.from_dict(coords)

@add_save_load()
class MagnitudeCalc(pw.QFrame, UiMagnitudeFrame):
    def __init__(self, origin, inventory, chop):
        super(MagnitudeCalc, self).__init__()
        self.setupUi(self)
        self.spectrum_Widget_Canvas = MatplotlibCanvas(self.spectrumWidget, nrows=2, ncols=1,
                                                       sharex=False, constrained_layout=True)
        self.event = origin
        self.chop = chop
        self.inventory = inventory
        self.runBtn.clicked.connect(self.moment_magnitude)

    def moment_magnitude(self,):
        density = self.densityDB.value()
        vp = self.vpDB.value() * 1000
        vs = vp / 1.73
        quality = self.qualityDB.value()
        radiation_pattern = 0.52
        k = 0.32
        Mws = []
        Mws_std = []
        self.Magnitude_Ws = []
        origin_time =self.event.time
        moments = []
        source_radii = []
        corner_frequencies = []
        corner_freqs = []
        self.spectrum_Widget_Canvas.plot([], [], 0)
        ax1 = self.spectrum_Widget_Canvas.get_axe(0)

        ax1.cla()

        for key in self.chop['Body waves']:

            magnitude_dict = self.chop['Body waves'][key]
            # Calculate distance
            coords = get_coordinates_from_metadata(self.inventory, magnitude_dict[0])
            dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event.latitude,self.event.longitude)
            pick_time = UTCDateTime(mdt.num2date(magnitude_dict[1][0]))
            data = np.array(magnitude_dict[2])

            delta = 1 / magnitude_dict[0][6]

            # Calculate the spectrum.

            spec, freq, jackknife_errors, _, _ = mtspec(data, delta=delta, time_bandwidth=3.5, statistics=True)
            spec = spec[1:]
            freq = freq[1:]
            # go to amplitude and displacement
            spec = np.sqrt(spec)/(2*np.pi*freq)
            # Plot spectrum

            ax1.loglog(freq, spec, '0.1', linewidth=0.5, color='black', alpha=0.5)
            ax1.set_ylim(spec.min() / 10.0, spec.max() * 100.0)
            ax1.set_xlim(freq[1], freq[len(freq) - 1])
            ax1.set_ylabel('Amplitude')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.grid(True, which="both", ls="-", color='grey')
            ax1.legend()


            # Finish Plot
            tt = pick_time-origin_time
            #print("Distance",dist,"Vp",vp,"Q",quality,"Density",density, "Travel Time", tt)

            try:
                fit = fit_spectrum(spec, freq, tt, spec.max(), 10.0, quality)
            except:
                continue

            if fit is None:
                continue

            Omega_0, f_c, err, _ = fit

            # Second Plot

            theoretical_spectrum = calculate_source_spectrum(freq, Omega_0, f_c, quality, tt)
            ax1.loglog(freq, theoretical_spectrum, color="red", alpha=0.5, lw=0.5)
            ax1.set_xlim(freq[1], freq[len(freq) - 1])
            ax1.set_ylabel('Amplitude')
            ax1.set_xlabel('Frequency [Hz]')
            ax1.grid(True, which="both", ls="-", color='grey')

            corner_freqs.append(f_c)
            M_0 = 4.0 * np.pi * density * vp ** 3 * dist * np.sqrt(Omega_0 ** 2) / radiation_pattern
            self.Magnitude_Ws.append(2.0 / 3.0 * (np.log10(M_0) - 9.1))
            r = 3 * k * vs / sum(corner_freqs)
            moments.append(M_0)
            source_radii.append(r)
            corner_frequencies.extend(corner_freqs)

        # Calculate the seismic moment via basic statistics.
        moments = np.array(moments)
        moment = moments.mean()
        moment_std = moments.std()

        corner_frequencies = np.array(corner_frequencies)
        corner_frequency = corner_frequencies.mean()
        corner_frequency_std = corner_frequencies.std()

        # Calculate the source radius.
        source_radii = np.array(source_radii)
        source_radius = source_radii.mean()
        source_radius_std = source_radii.std()

        # Calculate the stress drop of the event based on the average moment and
        # source radii.
        stress_drop = (7 * moment) / (16 * source_radius ** 3)
        stress_drop_std = np.sqrt((stress_drop ** 2) * \
                                  (((moment_std ** 2) / (moment ** 2)) + \
                                   (9 * source_radius * source_radius_std ** 2)))
        if source_radius > 0 and source_radius_std < source_radius:
            print("Source radius:", source_radius, " Std:", source_radius_std)
            print("Stress drop:", stress_drop / 1E5, " Std:", stress_drop_std / 1E5)


        Mw = 2.0 / 3.0 * (np.log10(moment) - 9.1)
        Mw_std = 2.0 / 3.0 * moment_std / (moment * np.log(10))
        Mws_std.append(Mw_std)
        Mws.append(Mw)
        print("Moment Magnitude",Mws)
        print("Moment Magnitude deviation", Mws_std)
        self.plot_histograms()

    def magnitude_surface(self):
        Ms = []
        for key in self.chop['Surf Waves']:
            magnitude_dict = self.chop['Surf Waves'][key]

            coords = get_coordinates_from_metadata(self.inventory, magnitude_dict[0])

            dist= locations2degrees(coords.Latitude, coords.Longitude, self.event.latitude,self.event.longitude)
            data = np.array(magnitude_dict[2])*1e9 # convert to nm
            Ms_value=np.log10(np.max(data)/(2*np.pi))+1.66*np.log10(dist)+3.3
            Ms.append(Ms_value)

        Ms_mean = Ms.mean()
        Ms_deviation =Ms.std()
        print("Surface Magnitude", Ms_mean, "Variance", Ms_deviation)

    def magnitude_body(self):
        Mb = []
        path_to_atenuation_file = os.path.join(ROOT_DIR, "earthquakeAnalisysis", "magnitude_atenuation")
        x, y = np.loadtxt(path_to_atenuation_file, skiprows=1, unpack=True)

        for key in self.chop['Body waves']:
            magnitude_dict = self.chop['Body waves'][key]
            coords = get_coordinates_from_metadata(self.inventory, magnitude_dict[0])
            dist = locations2degrees(coords.Latitude, coords.Longitude, self.event.latitude, self.event.longitude)
            dist = np.floor(dist)
            loc = np.where(x == dist)[0][0]
            atenuation = y[loc]
            data = np.array(magnitude_dict[2]) * 1e6  # convert to nm
            Mb_value = (np.log10(np.max(data)) / (2 * np.pi)) + atenuation
            Mb.append(Mb_value)
            Mb_mean = Mb.mean()
            Mb_deviation = Mb.std()
            print("Body Magnitude", Mb_mean, "Variance", Mb_deviation)

    def magnitude_coda(self):
        Mc = []
        # values for california
        a = 2.0
        b = 0.0035
        c= -0.87
        # 
        for key in self.chop['Coda']:
            magnitude_dict = self.chop['Coda'][key]
            coords = get_coordinates_from_metadata(self.inventory, magnitude_dict[0])
            dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event.latitude, self.event.longitude)
            dist = dist / 1000
            data = np.array(magnitude_dict[2])
            N=len(data)
            pick_time = UTCDateTime(mdt.num2date(magnitude_dict[1][0]))
            end_time = UTCDateTime(mdt.num2date(magnitude_dict[1][N]))
            t_coda = end_time-pick_time
            Mc_value = a*np.log10(t_coda)+b*dist+c
            Mc.append(Mc_value)
            Mc_mean = Mc.mean()
            Mc_deviation = Mc.std()
            print("Local Magnitude", Mc_mean, "Variance", Mc_deviation)

    def magnitude_local(self):

        ML = []
        for key in self.chop['Body waves']:
            magnitude_dict = self.chop['Body waves'][key]
            coords = get_coordinates_from_metadata(self.inventory, magnitude_dict[0])
            dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event.latitude, self.event.longitude)
            dist = dist/1000
            data = np.array(magnitude_dict[2])   # already converted Wood Anderson (Gain in mm 2800 +-60)
            max_amplitude = np.max(data)*1e6 # convert to  mm --> nm
            ML_value = np.log10(max_amplitude)+1.11*np.log10(dist)+0.00189*dist-2.09
            ML.append(ML_value)
            ML_mean = ML.mean()
            ML_deviation = ML.std()
            print("Local Magnitude", ML_mean, "Variance", ML_deviation)



    def plot_histograms(self):
        self.spectrum_Widget_Canvas.plot([], [], 1)
        ax2 = self.spectrum_Widget_Canvas.get_axe(1)
        ax2.cla()
        ax2.hist(self.Magnitude_Ws, bins=4*len(self.Magnitude_Ws), alpha=0.5, label="Mw")
        ax2.set_xlabel("Magnitude", size=12)
        ax2.set_ylabel("Count", size=12)
        ax2.legend(loc='upper right')

