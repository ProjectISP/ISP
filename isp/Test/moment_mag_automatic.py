#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to automatically determine the moment magnitudes of a larger number of
events.

The script will write one output file containing all events with one additional
magnitude.

Some configuration is required. Please edit the uppercase variables right after
all the imports to suit your needs.

All events need to be stored in ONE QuakeML file. Every event has to a larger
number of picks. Furthermore waveform data for all picks is necessary and
station information as (dataless)SEED files for every station.

The script could use some heavy refactoring but its program flow is quite
linear and it works well enough.

Requirements:
    * numpy
    * scipy
    * matplotlib
    * ObsPy
    * colorama
    * progressbar
    * mtspec (https://github.com/krischer/mtspec)

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2012
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import colorama
import glob
import matplotlib.pylab as plt
import mtspec
import numpy as np
from obspy import read, Stream
from obspy.core.event import readEvents, Comment, Magnitude, Catalog
from obspy.xseed import Parser
import progressbar
import scipy
import scipy.optimize
import warnings

# Rock density in km/m^3.
DENSITY = 2700.0
# Velocities in m/s.
V_P = 4800.0
V_S = V_P / 1.73
# How many seconds before and after the pick to choose for calculating the
# spectra.
TIME_BEFORE_PICK = 0.2
TIME_AFTER_PICK = 0.8
PADDING = 20
WATERLEVEL = 10.0
# Fixed quality factor. Very unstable inversion for it. Has almost no influence
# on the final seismic moment estimations but has some influence on the corner
# frequency estimation and therefore on the source radius estimation.
QUALITY_FACTOR = 1000

# Specifiy where to find the files. One large event file contain all events and
# an arbitrary number of waveform and station information files.
EVENT_FILES = glob.glob("events/*")
STATION_FILES = glob.glob("stations/*")
WAVEFORM_FILES = glob.glob("waveforms/*")

# Where to write the output file to.
OUTPUT_FILE = "events_with_moment_magnitudes.xml"


def fit_spectrum(spectrum, frequencies, traveltime, initial_omega_0,
    initial_f_c):
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


def calculate_source_spectrum(frequencies, omega_0, corner_frequency, Q,
    traveltime):
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




def calculate_moment_magnitudes(cat, output_file):
    """
    :param cat: obspy.core.event.Catalog object.
    """

    Mws = []
    Mls = []
    Mws_std = []

    for event in cat:
        if not event.origins:
            print "No origin for event %s" % event.resource_id
            continue
        if not event.magnitudes:
            print "No magnitude for event %s" % event.resource_id
            continue
        origin_time = event.origins[0].time
        local_magnitude = event.magnitudes[0].mag
        #if local_magnitude < 1.0:
            #continue
        moments = []
        source_radii = []
        corner_frequencies = []
        for pick in event.picks:
            # Only p phase picks.
            if pick.phase_hint.lower() == "p":
                radiation_pattern = 0.52
                velocity = V_P
                k = 0.32
            elif pick.phase_hint.lower() == "s":
                radiation_pattern = 0.63
                velocity = V_S
                k = 0.21
            else:
                continue
            distance = (pick.time - origin_time) * velocity
            if distance <= 0.0:
                continue
            stream = get_corresponding_stream(pick.waveform_id, pick.time,
                                              PADDING)
            if stream is None or len(stream) != 3:
                continue
            omegas = []
            corner_freqs = []
            for trace in stream:
                # Get the index of the pick.
                pick_index = int(round((pick.time - trace.stats.starttime) / \
                    trace.stats.delta))
                # Choose date window 0.5 seconds before and 1 second after pick.
                data_window = trace.data[pick_index - \
                    int(TIME_BEFORE_PICK * trace.stats.sampling_rate): \
                    pick_index + int(TIME_AFTER_PICK * trace.stats.sampling_rate)]
                # Calculate the spectrum.
                spec, freq = mtspec.mtspec(data_window, trace.stats.delta, 2)
                try:
                    fit = fit_spectrum(spec, freq, pick.time - origin_time,
                            spec.max(), 10.0)
                except:
                    continue
                if fit is None:
                    continue
                Omega_0, f_c, err, _ = fit
                Omega_0 = np.sqrt(Omega_0)
                omegas.append(Omega_0)
                corner_freqs.append(f_c)
            M_0 = 4.0 * np.pi * DENSITY * velocity ** 3 * distance * \
                np.sqrt(omegas[0] ** 2 + omegas[1] ** 2 + omegas[2] ** 2) / \
                radiation_pattern
            r = 3 * k * V_S / sum(corner_freqs)
            moments.append(M_0)
            source_radii.append(r)
            corner_frequencies.extend(corner_freqs)
        if not len(moments):
            print "No moments could be calculated for event %s" % \
                event.resource_id.resource_id
            continue

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
            print "Source radius:", source_radius, " Std:", source_radius_std
            print "Stress drop:", stress_drop / 1E5, " Std:", stress_drop_std / 1E5

        Mw = 2.0 / 3.0 * (np.log10(moment) - 9.1)
        Mw_std = 2.0 / 3.0 * moment_std / (moment * np.log(10))
        Mws_std.append(Mw_std)
        Mws.append(Mw)
        Mls.append(local_magnitude)
        calc_diff = abs(Mw - local_magnitude)
        Mw = ("%.3f" % Mw).rjust(7)
        Ml = ("%.3f" % local_magnitude).rjust(7)
        diff = ("%.3e" % calc_diff).rjust(7)

        ret_string = colorama.Fore.GREEN + \
            "For event %s: Ml=%s | Mw=%s | " % (event.resource_id.resource_id,
            Ml, Mw)
        if calc_diff >= 1.0:
            ret_string += colorama.Fore.RED
        ret_string += "Diff=%s" % diff
        ret_string += colorama.Fore.GREEN
        ret_string += " | Determined at %i stations" % len(moments)
        ret_string += colorama.Style.RESET_ALL
        print ret_string

        mag = Magnitude()
        mag.mag = Mw
        mag.mag_errors.uncertainty = Mw_std
        mag.magnitude_type = "Mw"
        mag.origin_id = event.origins[0].resource_id
        mag.method_id = "smi:com.github/krischer/moment_magnitude_calculator/automatic/1"
        mag.station_count = len(moments)
        mag.evaluation_mode = "automatic"
        mag.evaluation_status = "preliminary"
        mag.comments.append(Comment( \
            "Seismic Moment=%e Nm; standard deviation=%e" % (moment,
            moment_std)))
        mag.comments.append(Comment("Custom fit to Boatwright spectrum"))
        if source_radius > 0 and source_radius_std < source_radius:
            mag.comments.append(Comment( \
                "Source radius=%.2fm; standard deviation=%.2f" % (source_radius,
                source_radius_std)))
        event.magnitudes.append(mag)

    print "Writing output file..."
    cat.write(output_file, format="quakeml")


def fit_moment_magnitude_relation_curve(Mls, Mws, Mw_stds):
    """
    Fits a quadratic curve to
        Mw = a + b * Ml + c * Ml ** 2
    Returns the best fitting [a, b, c]
    """
    def y(x, a, b, c):
        return a + b * x + c * x ** 2
    # Use a straight line as starting point.
    Mls = np.ma.masked_invalid(Mls)
    Mws = np.ma.masked_invalid(Mws)
    inds = ~(Mls.mask | Mws.mask | np.isnan(Mw_stds) | (Mw_stds <= 0))
    popt, pcov = scipy.optimize.curve_fit(y, Mls[inds], Mws[inds], \
        p0=[0.0, 1.0, 0.0], sigma=Mw_stds[inds], maxfev=100000)
    return popt[0], popt[1], popt[2]


def plot_ml_vs_mw(catalog):
    moment_magnitudes = []
    moment_magnitudes_std = []
    local_magnitudes = []
    local_magnitudes_std = []
    for event in catalog:
        Mw = None
        Mw_std = None
        Ml = None
        Ml_std = None
        for mag in event.magnitudes:
            if Mw is not None and Ml is not None:
                break
            mag_type = mag.magnitude_type.lower()
            if mag_type == "mw":
                if Mw is not None:
                    continue
                Mw = mag.mag
                Mw_std = mag.mag_errors.uncertainty
            elif mag_type == "ml":
                if Ml is not None:
                    continue
                Ml = mag.mag
                Ml_std = mag.mag_errors.uncertainty
        moment_magnitudes.append(Mw)
        moment_magnitudes_std.append(Mw_std)
        local_magnitudes.append(Ml)
        local_magnitudes_std.append(Ml_std)
    moment_magnitudes = np.array(moment_magnitudes, dtype="float64")
    moment_magnitudes_std = np.array(moment_magnitudes_std, dtype="float64")
    local_magnitudes = np.array(local_magnitudes, dtype="float64")
    local_magnitudes_std = np.array(local_magnitudes_std, dtype="float64")

    # Fit a curve through the data.
    a, b, c = fit_moment_magnitude_relation_curve(local_magnitudes,
            moment_magnitudes, moment_magnitudes_std)
    x_values = np.linspace(-2.0, 4.0, 10000)
    fit_curve = a + b * x_values + c * x_values ** 2

    plt.figure(figsize=(10, 8))
    # Show the data values as dots.
    plt.scatter(local_magnitudes, moment_magnitudes, color="blue",
        edgecolor="black")

    # Plot the Ml=Mw line.
    plt.plot(x_values, x_values, label="$Mw=Ml$", color="k", alpha=0.8)
    plt.plot(x_values, 0.67 + 0.56 * x_values + 0.046 * x_values ** 2,
        label="$Mw=0.67 + 0.56Ml + 0.046Ml^2 (gruenthal etal 2003)$", color="green", ls="--")
    plt.plot(x_values, 0.53 + 0.646 * x_values + 0.0376 * x_values ** 2,
        label="$Mw=0.53 + 0.646Ml + 0.0376Ml^2 (gruenthal etal 2009)$", color="green")
    plt.plot(x_values, 0.594 * x_values + 0.985,
        label="$Mw=0.985 + 0.594Ml (goertz-allmann etal 2011)$", color="orange")
    plt.plot(x_values, (x_values + 1.21) / 1.58,
        label="$Mw=(Ml + 1.21) / 1.58 (bethmann etal 2011)$", color="red")
    plt.plot(x_values, fit_curve, color="blue",
        label="$Data$ $fit$ $with$ $Mw=%.2f + %.2fMl + %.3fMl^2$" % (a, b, c))
    # Set limits and labels.
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.xlabel("Ml", fontsize="x-large")
    plt.ylabel("Mw", fontsize="x-large")
    # Show grid and legend.
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig("moment_mag_automatic.pdf")

def plot_source_radius(cat):
    mw = []
    mw_std = []
    source_radius = []
    source_radius_std = []

    plt.figure(figsize=(10, 4.5))

    # Read the source radius.
    for event in cat:
        mag = event.magnitudes[1]
        if len(mag.comments) != 2:
            continue
        mw.append(mag.mag)
        mw_std.append(mag.mag_errors.uncertainty)
        sr, std = mag.comments[1].text.split(";")
        _, sr = sr.split("=")
        _, std = std.split("=")
        sr = float(sr[:-1])
        std = float(std)
        source_radius.append(sr)
        source_radius_std.append(std)
    plt.errorbar(mw, source_radius, yerr=source_radius_std,
        fmt="o", linestyle="None")
    plt.xlabel("Mw", fontsize="x-large")
    plt.ylabel("Source Radius [m]", fontsize="x-large")
    plt.grid()
    plt.savefig("/Users/lion/Desktop/SourceRadius.pdf")


if __name__ == "__main__":
    # Read all instrument responses.
    widgets = ['Parsing instrument responses...', progressbar.Percentage(),
        ' ', progressbar.Bar()]
    pbar = progressbar.ProgressBar(widgets=widgets,
        maxval=len(STATION_FILES)).start()
    parsers = {}
    # Read all waveform files.
    for _i, xseed in enumerate(STATION_FILES):
        pbar.update(_i)
        parser = Parser(xseed)
        channels = [c['channel_id'] for c in parser.getInventory()['channels']]
        parsers_ = dict.fromkeys(channels, parser)
        if any([k in parsers for k in parsers_.keys()]):
            msg = "Channel(s) defined in more than one metadata file."
            warnings.warn(msg)
        parsers.update(parsers_)
    pbar.finish()

    # Parse all waveform files.
    widgets = ['Indexing waveform files...     ', progressbar.Percentage(),
        ' ', progressbar.Bar()]
    pbar = progressbar.ProgressBar(widgets=widgets,
        maxval=len(WAVEFORM_FILES)).start()
    waveform_index = {}
    # Read all waveform files.
    for _i, waveform in enumerate(WAVEFORM_FILES):
        pbar.update(_i)
        st = read(waveform)
        for trace in st:
            if not trace.id in waveform_index:
                waveform_index[trace.id] = []
            waveform_index[trace.id].append( \
                {"filename": waveform,
                 "starttime": trace.stats.starttime,
                 "endtime": trace.stats.endtime})
    pbar.finish()

    # Define it inplace to create a closure for the waveform_index dictionary
    # because I am too lazy to fix the global variable issue right now...
    def get_corresponding_stream(waveform_id, pick_time, padding=1.0):
        """
        Helper function to find a requested waveform in the previously created
        waveform_index file.
        Also performs the instrument correction.

        Returns None if the file could not be found.
        """
        trace_ids = [waveform_id.getSEEDString()[:-1] + comp for comp in "ZNE"]
        st = Stream()
        start = pick_time - padding
        end = pick_time + padding
        for trace_id in trace_ids:
            for waveform in waveform_index.get(trace_id, []):
                if waveform["starttime"] > start:
                    continue
                if waveform["endtime"] < end:
                    continue
                st += read(waveform["filename"]).select(id=trace_id)
        for trace in st:
            paz = parsers[trace.id].getPAZ(trace.id, start)
            # PAZ in SEED correct to m/s. Add a zero to correct to m.
            paz["zeros"].append(0 + 0j)
            trace.detrend()
            trace.simulate(paz_remove=paz, water_level=WATERLEVEL)
        return st

    print "Reading all events."
    cat = Catalog()
    for filename in EVENT_FILES:
        cat += readEvents(filename)
    print "Done reading all events."

    # Will edit the Catalog object inplace.
    calculate_moment_magnitudes(cat, output_file=OUTPUT_FILE)
    # Plot it.
    plot_ml_vs_mw(cat)
