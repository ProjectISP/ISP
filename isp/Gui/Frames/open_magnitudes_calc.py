from obspy.geodetics import gps2dist_azimuth

from isp.Gui import pw
from isp.Gui.Frames.uis_frames import UiMagnitudeFrame
import mtspec
import numpy as np
from obspy import read, Stream
import scipy
import scipy.optimize
import warnings

from isp.Gui.Utils.pyqt_utils import add_save_load


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
    return coords

#def deconv(inventory,tr):
#    pass

@add_save_load()
class MagnitudeCalc(pw.QFrame, UiMagnitudeFrame):
    def __init__(self, origin, inventory, chop):
        super(MagnitudeCalc, self).__init__()
        self.setupUi(self)
        self.event = origin
        self.chop = chop
        self.inventory = inventory

    def moment_magnitude(self, density, vp, Q, radiation_pattern,k):
        Mws = []
        Mws_std = []
        origin_time =self.event.time
        moments = []
        source_radii = []
        corner_frequencies = []
        omegas = []
        corner_freqs = []

        for key, value in self.chop['Body waves']:


            # Calculate distance
            coords = get_coordinates_from_metadata(self.inventory, key[0])

            dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event.lat,
                                          self.event.lon)


            # Deconvolution to Displacement (1. convert to trace, 2. Deconvolution)


            #metadata = [dic_metadata['net'], dic_metadata['station'], dic_metadata['location'], dic_metadata['channel'],
            #            dic_metadata['starttime'], dic_metadata['endtime'], dic_metadata['sampling_rate'],
            #            dic_metadata['npts']]
            #id = {id: [metadata, t, s, xmin_index, xmax_index]}

            stats = {'network': key[0][0], 'station': key[0][1], 'location': key[0][2],
                'channel': key[0][3], 'npts': len(key[2]), 'sampling_rate': key[0][6], 'mseed': {'dataquality': 'D'}}

            stats['starttime'] = key[3]

            #st = Stream([Trace(data=data, header=stats)])

            pick_time = key[3]
            data = key[2]
            delta = key[0][6]
            # Calculate the spectrum.
        #     spec, freq = mtspec.mtspec(data, delta, 2)
        #     spec = np.sqrt(spec)/2*np.pi*freq
        #     tt = pick_time-origin_time
        #
        #     try:
        #         fit = fit_spectrum(spec, freq, tt,spec.max(), 10.0)
        #     except:
        #         continue
        #
        #     if fit is None:
        #         continue
        #
        #     Omega_0, f_c, err, _ = fit
        #     omegas.append(Omega_0)
        #     corner_freqs.append(f_c)
        #
        # M_0 = 4.0 * np.pi * density * vp ** 3 * dist * \
        #       np.sqrt(omegas[0] ** 2 + omegas[1] ** 2 + omegas[2] ** 2) / \
        #       radiation_pattern
        #
        # r = 3 * k * V_S / sum(corner_freqs)
        # moments.append(M_0)
        # source_radii.append(r)
        # corner_frequencies.extend(corner_freqs)
        #
        # # Calculate the seismic moment via basic statistics.
        # moments = np.array(moments)
        # moment = moments.mean()
        # moment_std = moments.std()
        #
        # corner_frequencies = np.array(corner_frequencies)
        # corner_frequency = corner_frequencies.mean()
        # corner_frequency_std = corner_frequencies.std()
        #
        # # Calculate the source radius.
        # source_radii = np.array(source_radii)
        # source_radius = source_radii.mean()
        # source_radius_std = source_radii.std()
        #
        # # Calculate the stress drop of the event based on the average moment and
        # # source radii.
        # stress_drop = (7 * moment) / (16 * source_radius ** 3)
        # stress_drop_std = np.sqrt((stress_drop ** 2) * \
        #                           (((moment_std ** 2) / (moment ** 2)) + \
        #                            (9 * source_radius * source_radius_std ** 2)))
        # if source_radius > 0 and source_radius_std < source_radius:
        #     print
        #     "Source radius:", source_radius, " Std:", source_radius_std
        #     print
        #     "Stress drop:", stress_drop / 1E5, " Std:", stress_drop_std / 1E5
        #
        # Mw = 2.0 / 3.0 * (np.log10(moment) - 9.1)
        # Mw_std = 2.0 / 3.0 * moment_std / (moment * np.log(10))
        # Mws_std.append(Mw_std)
        # Mws.append(Mw)
        # Mw = ("%.3f" % Mw).rjust(7)
