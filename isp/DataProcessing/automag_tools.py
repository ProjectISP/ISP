from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from isp.DataProcessing.automag_processing_tools import signal_preprocess_tools
from isp.Structures.structures import StationCoordinates
import numpy as np
import logging
from obspy.core import Stream
logger = logging.getLogger(__name__.split('.')[-1])

class preprocess_tools:

    def __init__(self, st, pick_info, event_info, arrival, inventory):

        self.st = st.copy()
        self.inventory = inventory
        self.pick_info = pick_info
        self.arrival = arrival
        self.event_info = event_info
        self.model = TauPyModel(model="iasp91")
        self.valid_stream = True
        self.valid_spectrum = True
        self.signal_window_time = None
        self.noise_window_time = None
        # Parameters
        self.scale = "Regional"

    def check_signal_level(self, rmsmin):
        validation = []
        for i, tr in enumerate(self.st):
            rms2 = np.power(tr.data, 2).sum()
            rms = np.sqrt(rms2)
            rms_min = rmsmin
            if rms <= rms_min:
                msg = '{} {}: Trace RMS smaller than {:g}: skipping trace'
                msg = msg.format(tr.id, tr.stats.instrtype, rms_min)
                self.valid_stream = False
                #raise RuntimeError(msg)
            else:
                validation.append(i)
        if len(validation) > 0:
            pass
        else:
            self.valid_stream = False

    def check_clipping(self, st, clipping_sensitivity):

        # before this process needs to be cut the trace between the end of noise window
        # and the end of the signal window
        validation = []
        for i, tr in enumerate(st):
            if self._is_clipped(tr, clipping_sensitivity):
                msg = ('{}: Trace is clipped or significantly distorted: '
                       'skipping trace'.format(tr.id))
                logger.warning(msg)
            else:
                validation.append(i)
        if len(validation) > 0:
            pass
        else:
            self.valid_stream = False

    def _check_sn_ratio(self, st_noise, st_signal, snratio_min):

        for tr_noise, tr_signal in zip(st_noise,st_signal):
            rmsnoise2 = np.power(tr_noise.data, 2).sum()
            rmsnoise = np.sqrt(rmsnoise2)
            rmsS2 = np.power(tr_signal.data, 2).sum()
            rmsS = np.sqrt(rmsS2)

            if rmsnoise == 0:
                msg = '{} {}: empty noise window: skipping trace'
                msg = msg.format(tr_noise.id, tr_noise.stats.instrtype)
                self.valid_stream = False
                raise RuntimeError(msg)

            sn_ratio = rmsS / rmsnoise
            logger.info('{} {}: S/N: {:.1f}'.format(
                tr_noise.id, tr_noise.stats.instrtype, sn_ratio))

            if sn_ratio < snratio_min:
                msg = '{} {}: S/N smaller than {:g}: skipping trace'
                msg = msg.format(tr_signal.id, tr_signal.stats.instrtype, snratio_min)
                logger.warning(msg)
                self.valid_stream = False

    def _is_clipped(self, trace, sensitivity):
        """
        Check if a trace is clipped, based on kernel density estimation.

        Kernel density estimation is used to find the peaks of the histogram of
        the trace data points. The peaks are then weighted by their distance from
        the trace average (which should be the most common value).
        The peaks with the highest weight are then checked for prominence,
        which is a measure of how much higher the peak is than the surrounding
        data. The prominence threshold is determined by the sensitivity parameter.
        If more than one peak is found, the trace is considered clipped or
        distorted.

        Parameters
        ----------
        trace : obspy.core.trace.Trace
            Trace to check.
        sensitivity : int
            Sensitivity level, from 1 (least sensitive) to 5 (most sensitive).
        debug : bool
            If True, plot trace, samples histogram and kernel density.

        Returns
        -------
        bool
            True if trace is clipped, False otherwise.
        """
        try:
            sensitivity = int(sensitivity)
            if sensitivity < 1 or sensitivity > 5:
                raise ValueError('sensitivity must be between 1 and 5')
            trace = trace.copy().detrend('demean')
            npts = len(trace.data)
            # Compute data histogram with a number of bins equal to 0.5% of data points
            nbins = int(npts * 0.005)
            counts, bins = np.histogram(trace.data, bins=nbins)
            counts = counts / np.max(counts)
            # Compute gaussian kernel density
            kde = gaussian_kde(trace.data, bw_method=0.2)
            max_data = np.max(np.abs(trace.data)) * 1.2
            density_points = np.linspace(-max_data, max_data, 100)
            density = kde.pdf(density_points)
            maxdensity = np.max(density)
            density /= maxdensity
            # Distance weight, parabolic, between 1 and 5
            dist_weight = np.abs(density_points) ** 2
            dist_weight *= 4 / dist_weight.max()
            dist_weight += 1
            density_weight = density * dist_weight
            # find peaks with minimum prominence based on clipping sensitivity
            min_prominence = [0.1, 0.05, 0.03, 0.02, 0.01]
            peaks, _ = find_peaks(density_weight, prominence=min_prominence[sensitivity - 1])
            # If more than one peak, then the signal is probably clipped or distorted
            if len(peaks) > 1:
                return True
            else:
                return False
        except:
            return True
    def __get_timespan(self):

        pickP_time = None
        pickS_time = None
        for item in self.pick_info:
            if item[0] == "P":
                pickP_time = item[1]
            elif item[0] == "S":
                pickS_time = item[1]

        self.pickP_time = pickP_time
        self.pickS_time = pickS_time

        #if self.arrival[0].distance_degrees > 1.0:
        if isinstance(pickS_time, UTCDateTime):
            self.signal_window_time = pickP_time + (pickS_time - pickP_time) * 0.95
            self.signal_window_duration = self.signal_window_time - pickP_time
            self.noise_window_duration = self.signal_window_duration / 3
            self.noise_window_time = pickP_time - self.noise_window_duration
        else:
            # Calculate distance in degree
            arrivals = self.model.get_travel_times(source_depth_in_km=self.pick_info[3],
                                                   distance_in_degree=self.arrival["distance_degrees"], phase_list=["S"])

            pickS_time = self.pick_info[3] + arrivals[0].time
            self.signal_window_time = pickP_time + (pickS_time - pickP_time) * 0.95
            self.signal_window_duration = self.signal_window_time - pickP_time
            self.noise_window_duration = self.signal_window_duration / 3
        #else:
            #self.signal_window_time = 6.0
            #self.noise_window_time = 2.0

    def merge_stream(self, gap_max, overlap_max):

        """
        cut earthquake (regional /teleseism), Check for gaps and overlaps; remove mean; merge stream;
        """

        self.__cut_earthquake(scale=self.scale)
        if self.valid_stream:
            traceid = self.st[0].id

            # compute gap/overlap statistics for the first cut trace.
            gaps_olaps = self.st.get_gaps()
            gaps = [g for g in gaps_olaps if g[6] >= 0]
            overlaps = [g for g in gaps_olaps if g[6] < 0]
            gap_duration = sum(g[6] for g in gaps)
            if gap_duration > 0:
                msg = '{}: trace has {:.3f} seconds of gaps.'
                msg = msg.format(traceid, gap_duration)
                logger.info(msg)
                if gap_max is not None and gap_duration > gap_max:
                    msg = '{}: Gap duration larger than gap_max ({:.1f} s): '
                    msg += 'skipping trace'
                    msg = msg.format(traceid, gap_max)
                    self.valid_stream = False
                    raise RuntimeError(msg)
            overlap_duration = -1 * sum(g[6] for g in overlaps)
            if overlap_duration > 0:
                msg = '{}: trace has {:.3f} seconds of overlaps.'
                msg = msg.format(traceid, overlap_duration)
                logger.info(msg)
                if overlap_max is not None and overlap_duration > overlap_max:
                    msg = '{}: Overlap duration larger than overlap_max ({:.1f} s): '
                    msg += 'skipping trace'
                    msg = msg.format(traceid, overlap_max)
                    self.valid_stream = False
                    raise RuntimeError(msg)

            # Merge stream to remove gaps and overlaps
            try:
                # Finally, demean (only if trace has not be already preprocessed)
                # Since the count value is generally huge, we need to demean twice
                # to take into account for the rounding error
                self.st.detrend(type='constant')
                self.st.detrend(type='constant')
                self.st.merge(fill_value=0)
                # st.merge raises a generic Exception if traces have
                # different sampling rates
            except Exception:
                self.valid_stream = False
                msg = '{}: unable to fill gaps: skipping trace'.format(traceid)
                raise RuntimeError(msg)

    def deconv_waveform(self, gap_max, overlap_max, rmsmin, clipping_sensitivity):
        self.st_deconv = Stream([])
        self.st_wood = Stream([])
        self.__get_timespan()
        self.merge_stream(gap_max, overlap_max) #this process includes cut around earthquake
        self.check_signal_level(rmsmin=rmsmin)
        st = self.__cut_waveform(cutstream=False) #this process is just to check that it is not clipped
        self.check_clipping(st, clipping_sensitivity=clipping_sensitivity)
        paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
                  'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

        if self.valid_stream:

            self.st.detrend(type="simple")
            self.st.taper(type="blackman", max_percentage=0.05)
            f1 = 0.05
            f2 = 0.08
            f3 = 0.35 * self.st[0].stats.sampling_rate
            f4 = 0.40 * self.st[0].stats.sampling_rate
            pre_filt = (f1, f2, f3, f4)

            st_deconv = []
            st_wood = []


            for tr in self.st:

                tr_deconv = tr.copy()
                tr_wood = tr.copy()

                try:
                    tr_deconv.remove_response(inventory=self.inventory, pre_filt=pre_filt, output="DISP", water_level=90)
                    st_deconv.append(tr_deconv)
                except:
                    tr.data = np.array([])

                try:
                    resp = self.inventory.get_response(tr.id, tr.stats.starttime)
                    resp = resp.response_stages[0]
                    paz_mine = {'sensitivity': resp.stage_gain * resp.normalization_factor, 'zeros': resp.zeros,
                                'gain': resp.stage_gain, 'poles': resp.poles}
                    tr_wood.simulate(paz_remove=paz_mine, paz_simulate=paz_wa, water_level=90)
                    st_wood.append(tr_wood)
                except:
                    tr.data = np.array([])

            self.st_deconv = Stream(traces=st_deconv)
            self.st_wood = Stream(traces=st_wood)

    def __cut_waveform(self, cutstream=True):

        # Cut waveform taking as reference the P wave and S wave if is picked, otherwise estimates the theoretical S.
        # If distance is inside 1º, hardcoded to 10 seconds time window

        if cutstream:
            self.st_deconv.trim(starttime=self.pickP_time - self.noise_window_duration,
                                endtime=self.pickP_time + self.signal_window_duration)
        else:
            st = self.st.copy()
            st.trim(starttime=self.pickP_time - self.noise_window_duration,
                    endtime=self.pickP_time + self.signal_window_duration)
            return st

    def _check_noise_level(self):

        for tr_signal, tr_noise in zip(self.st_cut_signal, self.st_cut_noise):
            traceId = tr_signal.get_id()
            trace_signal_rms = ((tr_signal.data ** 2).sum()) ** 0.5
            # Scale trace_noise_rms to length of signal window,
            # based on length of non-zero noise window
            try:
                scale_factor = float(len(tr_signal)) / len(tr_noise.data != 0)
            except ZeroDivisionError:
                scale_factor = 1
            trace_noise_rms = ((tr_noise.data ** 2 * scale_factor).sum()) ** 0.5
            if trace_noise_rms / trace_signal_rms < 1e-6:
                # Skip trace if noise level is too low and if noise weighting is used
                msg = \
                    '{}: Noise level is too low or zero: station will be skipped'
                msg = msg.format(traceId)
                self.valid_spectrum = False
                raise RuntimeError(msg)
    def __split_noise2signal(self):
        st1 = self.st_deconv.copy()
        st2 = self.st_deconv.copy()

        self.st_cut_signal = st1.trim(starttime=(self.pickP_time - 0.05*self.signal_window_duration),
                            endtime=self.pickP_time + self.signal_window_duration)
        self.st_cut_noise = st2.trim(starttime=self.pickP_time - self.noise_window_duration,
                                                 endtime=self.pickP_time+self.noise_window_duration)

    def __cut_earthquake(self, scale="Regional"):
        # check minimum span

        maxstart = np.max([tr.stats.starttime for tr in self.st])
        minend = np.min([tr.stats.endtime for tr in self.st])

        if scale == "Regional":
            start_diff = (self.pick_info[0][1] - 60) - maxstart
            end_diff = minend - (self.pick_info[0][1] + 3 * 60)
            if start_diff > 0 and end_diff > 0:
                self.st.trim(starttime=self.pick_info[0][1] - 60, endtime=self.pick_info[0][1] + 3 * 60)
            else:
                self.valid_stream =False

        else:
            start_diff = (self.pick_info[0][1] - 1300) - maxstart
            end_diff = minend  - (self.pick_info[0][1] + 3600)
            if start_diff > 0 and end_diff > 0:
                self.st.trim(starttime=self.pick_info[0][1] - 1300, endtime=self.pick_info[0][1] + 3600)
            else:
                self.valid_stream = False
    def compute_spectrum(self, geom_spread_model, geom_spread_n_exponent,
                         geom_spread_cutoff_distance, rho, spectral_smooth_width_decades, spectral_sn_min,
                         spectral_sn_freq_range):
         spectrum = None
         if isinstance(self.st_deconv, Stream) and self.valid_stream:
            self.__cut_waveform()
            self.__split_noise2signal()
            spt = signal_preprocess_tools(self.st_cut_noise, self.st_cut_signal, self.arrival["distance_km"],
                geom_spread_model, geom_spread_n_exponent, geom_spread_cutoff_distance, self.event_info, rho,
                                          spectral_smooth_width_decades, spectral_sn_min, spectral_sn_freq_range)

            # convert the spectral amplitudes to moment magnitude
            spectrum = spt.do_spectrum()

         return spectrum

    def extract_coordinates_from_station_name(self, inventory, name):
         selected_inv = inventory.select(station=name)
         cont = selected_inv.get_contents()
         coords = selected_inv.get_coordinates(cont['channels'][0])
         return StationCoordinates.from_dict(coords)
    #
    def magnitude_local(self):
        #print("Calculating Local Magnitude")
        tr_E = self.st_wood.select(component="E")
        tr_E = tr_E[0]
        tr_N = self.st_wood.select(component="N")
        tr_N = tr_N[0]
        coords = self.extract_coordinates_from_station_name(self.inventory, self.st_wood[0].stats.station)
        dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event_info[1], self.event_info[2])
        dist = dist / 1000
        max_amplitude_N = np.max(tr_N.data)*1e3 # convert to  mm --> nm
        max_amplitude_E = np.max(tr_E.data) * 1e3  # convert to  mm --> nm
        max_amplitude = max([max_amplitude_E, max_amplitude_N])
        ML_value = np.log10(max_amplitude)+1.11*np.log10(dist)+0.00189*dist-2.09
        #print(ML_value)
        return ML_value




