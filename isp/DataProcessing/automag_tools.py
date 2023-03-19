from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel

from isp.Structures.structures import StationCoordinates
from isp.DataProcessing.automag_additional_tools import AddMagTools
import numpy as np
import logging
from obspy.core import Stream
import re
logger = logging.getLogger(__name__.split('.')[-1])
class Indmag:
    def __init__(self, st_deconv, st_wood, pick_info, event_info, inventory):
        self.st_deconv = st_deconv
        self.st_wood = st_wood
        self.inventory = inventory
        self.pick_info = pick_info
        self.event_info = event_info
        self.ML = []

    def cut_waveform(self):
        pickP_time = None
        for item in self.pick_info:
            if item[0] == "P":
                pickP_time = item[1]

        self.st_deconv.trim(starttime=pickP_time-5, endtime=pickP_time+300)
        self.st_wood.trim(starttime=pickP_time-5, endtime=pickP_time + 300)

    def extract_coordinates_from_station_name(self, inventory, name):
        selected_inv = inventory.select(station=name)
        cont = selected_inv.get_contents()
        coords = selected_inv.get_coordinates(cont['channels'][0])
        return StationCoordinates.from_dict(coords)

    def magnitude_local(self):
        print("Calculating Local Magnitude")
        tr_E = self.st_wood.select(component="E")
        tr_E = tr_E[0]
        tr_N = self.st_deconv.select(component="N")
        tr_N = tr_N[0]
        coords = self.extrac_coordinates_from_station_name(self.inventory, self.st_deconv[0].stats.station)
        dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event_info[1], self.event_info[2])
        dist = dist / 1000
        max_amplitude_N = np.max(tr_N.data)*1e3 # convert to  mm --> nm
        max_amplitude_E = np.max(tr_E.data) * 1e3  # convert to  mm --> nm
        max_amplitude = max([max_amplitude_E, max_amplitude_N])
        ML_value = np.log10(max_amplitude)+1.11*np.log10(dist)+0.00189*dist-2.09
        print(ML_value)
        self.ML.append(ML_value)

    def process_traces(self, config, st):
        """Remove mean, deconvolve and ignore unwanted components."""
        logger.info('Processing traces...')
        out_st = Stream()
        for id in sorted(set(tr.id for tr in st)):
            # We still use a stream, since the trace can have gaps or overlaps
            st_sel = st.select(id=id)
            try:
                self._skip_ignored(config, id)
                self.add_hypo_dist_and_arrivals(config, st_sel)
                trace = AddMagTools.merge_stream(config, st_sel)
                trace.stats.ignore = False
                trace_process = AddMagTools.process_trace(config, trace)
                out_st.append(trace_process)
            except (ValueError, RuntimeError) as msg:
                logger.warning(msg)
                continue

        if len(out_st) == 0:
            logger.error('No traces left! Exiting.')
            #ssp_exit()

        # Rotate traces, if SH or SV is requested
        if config.wave_type in ['SH', 'SV']:
            for id in sorted(set(tr.id[:-1] for tr in out_st)):
                net, sta, loc, chan = id.split('.')
                st_sel = out_st.select(network=net, station=sta,
                                       location=loc, channel=chan + '?')
                t0 = max(tr.stats.starttime for tr in st_sel)
                t1 = min(tr.stats.endtime for tr in st_sel)
                st_sel.trim(t0, t1)
                st_sel.rotate('NE->RT')

        logger.info('Processing traces: done')
        return out_st

    @staticmethod
    def _skip_ignored(config, id):
        """Skip traces ignored from config."""
        network, station, location, channel = id.split('.')
        # build a list of all possible ids, from station only
        # to full net.sta.loc.chan
        ss = [station, ]
        ss.append('.'.join((network, station)))
        ss.append('.'.join((network, station, location)))
        ss.append('.'.join((network, station, location, channel)))
        if config.use_traceids is not None:
            combined = (
                    "(" + ")|(".join(config.use_traceids) + ")"
            ).replace('.', r'\.')
            if not any(re.match(combined, s) for s in ss):
                msg = '{}: ignored from config file'.format(id)
                raise RuntimeError(msg)
        if config.ignore_traceids is not None:
            combined = (
                    "(" + ")|(".join(config.ignore_traceids) + ")"
            ).replace('.', r'\.')
            if any(re.match(combined, s) for s in ss):
                msg = '{}: ignored from config file'.format(id)
                raise RuntimeError(msg)

    def add_hypo_dist_and_arrivals(self, config, st):
        for trace in st:
            if AddMagTools.hypo_dist(trace) is None:
                msg = '{}: Unable to compute hypocentral distance: skipping trace'
                msg = msg.format(trace.id)
                raise RuntimeError(msg)
            if config.max_epi_dist is not None and \
                    trace.stats.epi_dist > config.max_epi_dist:
                msg = '{}: Epicentral distance ({:.1f} km) '
                msg += 'larger than max_epi_dist ({:.1f} km): skipping trace'
                msg = msg.format(
                    trace.id, trace.stats.epi_dist, config.max_epi_dist)
                raise RuntimeError(msg)
            AddMagTools.add_arrivals_to_trace(trace, config)
            try:
                p_arrival_time = trace.stats.arrivals['P'][1]
            except KeyError:
                msg = '{}: Unable to get P arrival time: skipping trace'
                msg = msg.format(trace.id)
                raise RuntimeError(msg)
            if config.wave_type[0] == 'P' and p_arrival_time < trace.stats.starttime:
                msg = '{}: P-window incomplete: skipping trace'
                msg = msg.format(trace.id)
                raise RuntimeError(msg)
            try:
                s_arrival_time = trace.stats.arrivals['S'][1]
            except KeyError:
                msg = '{}: Unable to get S arrival time: skipping trace'
                msg = msg.format(trace.id)
                raise RuntimeError(msg)
            if config.wave_type[0] == 'S' and s_arrival_time < trace.stats.starttime:
                msg = '{}: S-window incomplete: skipping trace'
                msg = msg.format(trace.id)
                raise RuntimeError(msg)
            # Signal window for spectral analysis (S phase)
            s_minus_p = s_arrival_time - p_arrival_time
            s_pre_time = config.signal_pre_time
            if s_minus_p / 2 < s_pre_time:
                # use (Ts-Tp)/2 if it is smaller than signal_pre_time
                # (for short-distance records with short S-P interval)
                s_pre_time = s_minus_p / 2
                msg = '{}: signal_pre_time is larger than (Ts-Tp)/2.'
                msg += 'Using (Ts-Tp)/2 instead'
                msg = msg.format(trace.id)
                logger.warning(msg)
            t1 = s_arrival_time - s_pre_time
            t1 = max(trace.stats.starttime, t1)
            t2 = t1 + config.win_length
            trace.stats.arrivals['S1'] = ('S1', t1)
            trace.stats.arrivals['S2'] = ('S2', t2)
            # Signal window for spectral analysis (P phase)
            t1 = p_arrival_time - config.signal_pre_time
            t1 = max(trace.stats.starttime, t1)
            t2 = t1 + min(config.win_length, s_minus_p)
            trace.stats.arrivals['P1'] = ('P1', t1)
            trace.stats.arrivals['P2'] = ('P2', t2)
            # Noise window for spectral analysis
            t1 = max(trace.stats.starttime, p_arrival_time - config.noise_pre_time)
            t2 = t1 + config.win_length
            if t2 >= p_arrival_time:
                logger.warning(
                    '{}: noise window ends after P-wave arrival'.format(trace.id))
                # Note: maybe we should also take into account signal_pre_time here
                t2 = p_arrival_time
                t1 = min(t1, t2)
            trace.stats.arrivals['N1'] = ('N1', t1)
            trace.stats.arrivals['N2'] = ('N2', t2)

class preprocess_tools:

    pick_info = None
    model = None
    arrival = None
    st = None

    def __init__(self, st, pick_info, event_info, arrival, inventory):

        self.st = st
        self.inventory = inventory
        self.pick_info = pick_info
        self.arrival = arrival
        self.event_info = event_info
        self.model = TauPyModel(model="iasp91")

    @classmethod
    def cut_waveform(cls):

        # Cut waveform taking as reference the P wave and S wave if is picked, otherwise estimates the theoretical S.
        # If distance is inside 1º, hardcoded to 10 seconds time window

        pickP_time = None
        pickS_time = None
        for item in cls.pick_info:
            if item[0] == "P":
                pickP_time = item[1]
            elif item[0] == "S":
                pickS_time = item[1]

        if cls.arrival[0].distance > 1.0:
            if isinstance(pickS_time, UTCDateTime):
                signal_window_time = pickP_time + (pickS_time-pickP_time)*0.95
                noise_window_time = signal_window_time/3
            else:
                # Calculate distance in degree
                arrivals = cls.model.get_travel_times(source_depth_in_km=cls.pick_info[3],
                            distance_in_degree=cls.arrival[0].distance, phase_list=["S"])

                pickS_time = cls.pick_info[3]+arrivals[0].time

                signal_window_time = pickP_time + (pickS_time-pickP_time)*0.95
                noise_window_time = signal_window_time / 3
        else:
            signal_window_time = pickP_time + 10
            noise_window_time = signal_window_time / 3

        cls.st.trim(starttime=pickP_time - noise_window_time, endtime=pickP_time + signal_window_time)

        return cls.st