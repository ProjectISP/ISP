import math
import logging
import os
from glob import glob
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
#from isp.earthquakeAnalisysis.NLLGrid import NLLGrid
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
logger = logging.getLogger(__name__.split('.')[-1])

class AddMagTools:

    @classmethod
    def hypo_dist(cls, trace):
        """Compute hypocentral and epicentral distance (in km) for a trace."""
        try:
            coords = trace.stats.coords
            hypo = trace.stats.hypo
        except (KeyError, AttributeError):
            return None
        if None in (coords, hypo):
            return None
        stla = coords.latitude
        stlo = coords.longitude
        stel = coords.elevation
        evla = hypo.latitude
        evlo = hypo.longitude
        evdp = hypo.depth
        if None in (stla, stlo, stel, evla, evlo, evdp):
            return None
        epi_dist, az, baz = gps2dist_azimuth(
            hypo.latitude, hypo.longitude,
            trace.stats.coords.latitude, trace.stats.coords.longitude)
        epi_dist /= 1e3   # in km
        gcarc = kilometers2degrees(epi_dist)
        hypo_dist = math.sqrt(epi_dist**2 + (stel+evdp)**2)
        trace.stats.azimuth = az
        trace.stats.back_azimuth = baz
        trace.stats.epi_dist = epi_dist
        trace.stats.hypo_dist = hypo_dist
        trace.stats.gcarc = gcarc
        return hypo_dist

    @classmethod
    def add_arrivals_to_trace(cls, trace, config):
        """
        Add P and S arrival times and takeoff angles to trace.

        Uses the theoretical arrival time if no pick is available
        or if the pick is too different from the theoretical arrival.
        """
        cls.add_arrivals_to_trace = {}
        cls.add_arrivals_to_trace["pick_cache"] = {}
        cls.add_arrivals_to_trace["travel_time_cache"] = {}
        cls.add_arrivals_to_trace["angle_cache"] = {}

        tolerance = config.p_arrival_tolerance
        for phase in 'P', 'S':
            key = '{}_{}'.format(trace.id, phase)
            # First, see if there are cached values
            try:
                trace.stats.arrivals[phase] = \
                    cls.add_arrivals_to_trace.pick_cache[key]
                trace.stats.travel_times[phase] = \
                    cls.add_arrivals_to_trace.travel_time_cache[key]
                trace.stats.takeoff_angles[phase] = \
                    cls.add_arrivals_to_trace.angle_cache[key]
                continue
            except KeyError:
                pass
            # If no cache is available, compute travel_time and takeoff_angle
            try:
                travel_time, takeoff_angle, method = \
                    cls._wave_arrival(trace, phase, config)
                theo_pick_time = cls._get_theo_pick_time(trace, travel_time)
                pick_time = cls._find_picks(trace, phase, theo_pick_time, tolerance)
            except RuntimeError:
                continue
            if pick_time is not None:
                logger.info('{}: found {} pick'.format(trace.id, phase))
                travel_time = \
                    cls._travel_time_from_pick(trace, pick_time) or travel_time
                pick_phase = phase
            elif theo_pick_time is not None:
                logger.info('{}: using theoretical {} pick from {}'.format(
                    trace.id, phase, method))
                pick_time = theo_pick_time
                pick_phase = phase + 'theo'
            else:
                continue
            if config.rp_from_focal_mechanism:
                logger.info(
                    '{}: {} takeoff angle: {:.1f} computed from {}'.format(
                        trace.id, phase, takeoff_angle, method
                    ))
            cls.add_arrivals_to_trace.pick_cache[key] = \
                trace.stats.arrivals[phase] = (pick_phase, pick_time)
            cls.add_arrivals_to_trace.travel_time_cache[key] = \
                trace.stats.travel_times[phase] = travel_time
            cls.add_arrivals_to_trace.angle_cache[key] = \
                trace.stats.takeoff_angles[phase] = takeoff_angle

    @classmethod
    def add_arrivals_to_trace(cls, trace, config):
        """
        Add P and S arrival times and takeoff angles to trace.

        Uses the theoretical arrival time if no pick is available
        or if the pick is too different from the theoretical arrival.
        """
        tolerance = config.p_arrival_tolerance
        for phase in 'P', 'S':
            key = '{}_{}'.format(trace.id, phase)
            # First, see if there are cached values
            try:
                trace.stats.arrivals[phase] = \
                    cls.add_arrivals_to_trace.pick_cache[key]
                trace.stats.travel_times[phase] = \
                    cls.add_arrivals_to_trace.travel_time_cache[key]
                trace.stats.takeoff_angles[phase] = \
                    cls.add_arrivals_to_trace.angle_cache[key]
                continue
            except KeyError:
                pass
            # If no cache is available, compute travel_time and takeoff_angle
            try:
                travel_time, takeoff_angle, method = \
                    cls._wave_arrival(trace, phase, config)
                theo_pick_time = cls._get_theo_pick_time(trace, travel_time)
                pick_time = cls._find_picks(trace, phase, theo_pick_time, tolerance)
            except RuntimeError:
                continue
            if pick_time is not None:
                logger.info('{}: found {} pick'.format(trace.id, phase))
                travel_time = \
                    cls._travel_time_from_pick(trace, pick_time) or travel_time
                pick_phase = phase
            elif theo_pick_time is not None:
                logger.info('{}: using theoretical {} pick from {}'.format(
                    trace.id, phase, method))
                pick_time = theo_pick_time
                pick_phase = phase + 'theo'
            else:
                continue
            if config.rp_from_focal_mechanism:
                logger.info(
                    '{}: {} takeoff angle: {:.1f} computed from {}'.format(
                        trace.id, phase, takeoff_angle, method
                    ))
            cls.add_arrivals_to_trace.pick_cache[key] = \
                trace.stats.arrivals[phase] = (pick_phase, pick_time)
            cls.add_arrivals_to_trace.travel_time_cache[key] = \
                trace.stats.travel_times[phase] = travel_time
            cls.add_arrivals_to_trace.angle_cache[key] = \
                trace.stats.takeoff_angles[phase] = takeoff_angle

    @classmethod
    def _wave_arrival(cls, trace, phase, config):
        """Get travel time and takeoff angle."""
        NLL_time_dir = config.NLL_time_dir
        focmec = config.rp_from_focal_mechanism
        vel = {'P': config.vp_tt, 'S': config.vs_tt}
        try:
            travel_time, takeoff_angle = \
                cls._wave_arrival_nll(trace, phase, NLL_time_dir, focmec)
            method = 'NonLinLoc grid'
            return travel_time, takeoff_angle, method
        except RuntimeError:
            pass
        try:
            travel_time, takeoff_angle = \
                cls._wave_arrival_vel(trace, vel[phase])
            method = 'constant V{}: {:.1f} km/s'.format(phase.lower(), vel[phase])
            return travel_time, takeoff_angle, method
        except RuntimeError:
            pass
        try:
            travel_time, takeoff_angle = cls._wave_arrival_taup(trace, phase)
            method = 'global velocity model (iasp91)'
            return travel_time, takeoff_angle, method
        except RuntimeError:
            raise

    @classmethod
    def _wave_arrival_nll(cls, trace, phase, NLL_time_dir, focmec):
        """Travel time and takeoff angle using a NLL grid."""
        if NLL_time_dir is None:
            raise RuntimeError
        station = trace.stats.station
        travel_time = takeoff_angle = None
        grdtypes = ['time']
        if focmec:
            grdtypes.append('angle')
        for type in grdtypes:
            try:
                grd = cls._get_nll_grd(phase, station, type, NLL_time_dir)
            except RuntimeError:
                logger.warning(
                    '{}: Cannot find NLL {} grid. '
                    'Falling back to another method'.format(trace.id, type))
                raise RuntimeError
            if grd.station == 'DEFAULT':
                sta_x, sta_y = grd.project(
                    trace.stats.coords.longitude, trace.stats.coords.latitude)
                grd.sta_x, grd.sta_y = sta_x, sta_y
            hypo_x, hypo_y = grd.project(
                trace.stats.hypo.longitude, trace.stats.hypo.latitude)
            hypo_z = trace.stats.hypo.depth
            if type == 'time':
                travel_time = grd.get_value(hypo_x, hypo_y, hypo_z)
            elif type == 'angle':
                azimuth, takeoff_angle, quality = grd.get_value(
                    hypo_x, hypo_y, hypo_z)
        return travel_time, takeoff_angle

    def _get_nll_grd(cls, phase, station, type, NLL_time_dir):
        # Lazy-import here, since nllgrid is not an installation requirement
        for _station in station, 'DEFAULT':
            key = '{}_{}_{}'.format(phase, _station, type)
            try:
                # TODO SOLVE CIRCULAR IMPORT PROBLEM
                # first try to lookup in cache
                #grd = cls._get_nll_grd.grds[key]
                grd = None
                return grd
            except KeyError:
                pass
            try:
                grdfile = '*.{}.{}.{}.hdr'.format(phase, _station, type)
                grdfile = os.path.join(NLL_time_dir, grdfile)
                grdfile = glob(grdfile)[0]
                #grd = NLLGrid(grdfile)
                # TODO SOLVE CIRCULAR IMPORT PROBLEM
                grd = None
                # cache NLL grid
                cls._get_nll_grd.grds[key] = grd
                return grd
            except IndexError:
                # IndexError from glob()[0]
                pass
        raise RuntimeError

    @classmethod
    def _wave_arrival_vel(cls, trace, vel):
        """Travel time and takeoff angle using a constant velocity (in km/s)."""
        if vel is None:
            raise RuntimeError
        travel_time = trace.stats.hypo_dist / vel
        takeoff_angle = math.degrees(math.asin(trace.stats.epi_dist / trace.stats.hypo_dist))
        # takeoff angle is 180° upwards and 0° downwards
        takeoff_angle = 180. - takeoff_angle
        return travel_time, takeoff_angle

    def _find_picks(cls, trace, phase, theo_pick_time, tolerance):
        """Search for valid picks in trace stats. Return pick time if found."""
        for pick in (p for p in trace.stats.picks if p.phase == phase):
            if cls._validate_pick(pick, theo_pick_time, tolerance, trace.id):
                trace.stats.arrivals[phase] = (phase, pick.time)
                return pick.time
        return None

    def _validate_pick(cls, pick, theo_pick_time, tolerance, trace_id):
        """Check if a pick is valid, i.e., close enough to theoretical one."""
        if theo_pick_time is None:
            return True
        delta_t = pick.time - theo_pick_time
        if abs(delta_t) > tolerance:  # seconds
            logger.warning(
                '{}: measured {} pick time - theoretical time = {:.1f} s'.format(
                    trace_id, pick.phase, delta_t
                ))
            return False
        return True

    @classmethod
    def _get_theo_pick_time(cls, trace, travel_time):
        try:
            theo_pick_time = trace.stats.hypo.origin_time + travel_time
        except TypeError:
            theo_pick_time = None
        return theo_pick_time
    @classmethod
    def _travel_time_from_pick(cls, trace, pick_time):
        try:
            travel_time = pick_time - trace.stats.hypo.origin_time
        except TypeError:
            travel_time = None
        return travel_time

    @classmethod
    def merge_stream(cls, config, st):
        """
        Check for gaps and overlaps; remove mean; merge stream.
        """
        traceid = st[0].id
        # First, compute gap/overlap statistics for the whole trace.
        gaps_olaps = st.get_gaps()
        gaps = [g for g in gaps_olaps if g[6] >= 0]
        overlaps = [g for g in gaps_olaps if g[6] < 0]
        gap_duration = sum(g[6] for g in gaps)
        if gap_duration > 0:
            msg = '{}: trace has {:.3f} seconds of gaps.'
            msg = msg.format(traceid, gap_duration)
            logger.info(msg)
            gap_max = config.gap_max
            if gap_max is not None and gap_duration > gap_max:
                msg = '{}: Gap duration larger than gap_max ({:.1f} s): '
                msg += 'skipping trace'
                msg = msg.format(traceid, gap_max)
                raise RuntimeError(msg)
        overlap_duration = -1 * sum(g[6] for g in overlaps)
        if overlap_duration > 0:
            msg = '{}: trace has {:.3f} seconds of overlaps.'
            msg = msg.format(traceid, overlap_duration)
            logger.info(msg)
            overlap_max = config.overlap_max
            if overlap_max is not None and overlap_duration > overlap_max:
                msg = '{}: Overlap duration larger than overlap_max ({:.1f} s): '
                msg += 'skipping trace'
                msg = msg.format(traceid, overlap_max)
                raise RuntimeError(msg)
        # Then, compute the same statisics for the signal window.
        st_cut = st.copy()
        if config.wave_type[0] == 'S':
            t1 = st[0].stats.arrivals['S1'][1]
            t2 = st[0].stats.arrivals['S2'][1]
        elif config.wave_type[0] == 'P':
            t1 = st[0].stats.arrivals['P1'][1]
            t2 = st[0].stats.arrivals['P2'][1]
        st_cut.trim(starttime=t1, endtime=t2)
        if not st_cut:
            msg = '{}: No signal for the selected {}-wave cut interval: '
            msg += 'skipping trace >\n'
            msg += '> Cut interval: {} - {}'
            msg = msg.format(traceid, config.wave_type[0], t1, t2)
            raise RuntimeError(msg)
        gaps_olaps = st_cut.get_gaps()
        gaps = [g for g in gaps_olaps if g[6] >= 0]
        overlaps = [g for g in gaps_olaps if g[6] < 0]
        duration = st_cut[-1].stats.endtime - st_cut[0].stats.starttime
        gap_duration = sum(g[6] for g in gaps)
        if gap_duration > duration / 4:
            msg = '{}: Too many gaps for the selected cut interval: skipping trace'
            msg = msg.format(traceid)
            raise RuntimeError(msg)
        overlap_duration = -1 * sum(g[6] for g in overlaps)
        if overlap_duration > 0:
            msg = '{}: Signal window has {:.3f} seconds of overlaps.'
            msg = msg.format(traceid, overlap_duration)
            logger.info(msg)
        # Finally, demean (only if trace has not be already preprocessed)
        if config.trace_units == 'auto':
            # Since the count value is generally huge, we need to demean twice
            # to take into account for the rounding error
            st.detrend(type='constant')
            st.detrend(type='constant')
        # Merge stream to remove gaps and overlaps
        try:
            st.merge(fill_value=0)
            # st.merge raises a generic Exception if traces have
            # different sampling rates
        except Exception:
            msg = '{}: unable to fill gaps: skipping trace'.format(traceid)
            raise RuntimeError(msg)
        return st[0]

    @classmethod
    def process_trace(cls, config, trace):
        # copy trace for manipulation
        trace_process = trace.copy()
        comp = trace_process.stats.channel
        instrtype = trace_process.stats.instrtype
        if config.ignore_vertical and comp[-1] in config.vertical_channel_codes:
            if config.wave_type == 'P':
                msg = '{} {}: cannot ignore vertical trace, '
                msg += 'since "wave_type" is set to "P"'
                msg = msg.format(trace.id, trace.stats.instrtype)
                logger.warning(msg)
            else:
                msg = '{} {}: ignoring vertical trace as requested'.format(
                    trace.id, trace.stats.instrtype
                )
                logger.info(msg)
                trace_process.stats.ignore = True
                trace_process.stats.ignore_reason = 'vertical'
        # check if the trace has (significant) signal
        cls._check_signal_level(config, trace_process)
        # check if trace is clipped
        cls._check_clipping(config, trace_process)
        # Remove instrument response
        if not config.options.no_response:
            correct = config.correct_instrumental_response
            bp_freqmin, bp_freqmax = cls.get_bandpass_frequencies(config, trace)
            pre_filt = (bp_freqmin, bp_freqmin*1.1, bp_freqmax*0.9, bp_freqmax)
            if cls.remove_instr_response(trace_process, correct, pre_filt) is None:
                msg = '{} {}: Unable to remove instrument response: '
                msg += 'skipping trace'
                msg = msg.format(trace_process.id, instrtype)
                raise RuntimeError(msg)
        cls.filter_trace(config, trace_process)
        # Check if the trace has significant signal_to_noise ratio
        cls.check_sn_ratio(config, trace_process)
        return trace_process

    def _check_signal_level(cls, config, trace):
        rms2 = np.power(trace.data, 2).sum()
        rms = np.sqrt(rms2)
        rms_min = config.rmsmin
        if rms <= rms_min:
            msg = '{} {}: Trace RMS smaller than {:g}: skipping trace'
            msg = msg.format(trace.id, trace.stats.instrtype, rms_min)
            raise RuntimeError(msg)

    @classmethod
    def _check_clipping(cls, config, trace):
        trace.stats.clipped = False
        if config.clipping_sensitivity == 0:
            return
        # cut the trace between the end of noise window
        # and the end of the signal window
        t1 = trace.stats.arrivals['N2'][1]
        if config.wave_type[0] == 'S':
            t2 = trace.stats.arrivals['S2'][1]
        elif config.wave_type[0] == 'P':
            t2 = trace.stats.arrivals['P2'][1]
        t2 = (trace.stats.arrivals['S'][1] + config.win_length)
        tr = trace.copy().trim(t1, t2).detrend('demean')
        if cls.is_clipped(tr, config.clipping_sensitivity):
            trace.stats.clipped = True
            trace.stats.ignore = True
            trace.stats.ignore_reason = 'distorted'
            msg = (
                '{} {}: Trace is clipped or significantly distorted: '
                'skipping trace'.format(tr.id, tr.stats.instrtype)
            )
            logger.warning(msg)

    @classmethod
    def get_bandpass_frequencies(cls, config, trace):
        """Get frequencies for bandpass filter."""
        # see if there is a station-specific filter
        station = trace.stats.station
        try:
            bp_freqmin = float(config['bp_freqmin_' + station])
            bp_freqmax = float(config['bp_freqmax_' + station])
        except KeyError:
            instrtype = trace.stats.instrtype
            try:
                bp_freqmin = float(config['bp_freqmin_' + instrtype])
                bp_freqmax = float(config['bp_freqmax_' + instrtype])
            except KeyError:
                msg = '{}: Unknown instrument type: {}: skipping trace'
                msg = msg.format(trace.id, instrtype)
                raise ValueError(msg)
        return bp_freqmin, bp_freqmax

    @classmethod
    def remove_instr_response(cls, trace, correct='True', pre_filt=(0.5, 0.6, 40., 45.)):
        if correct == 'False':
            return trace
        traceId = trace.get_id()
        paz = trace.stats.paz
        if paz is None:
            logger.warning('%s: no poles and zeros for trace' % traceId)
            return None

        # remove the mean...
        trace.detrend(type='constant')
        # ...and the linear trend
        trace.detrend(type='linear')

    @classmethod
    def filter_trace(cls, config, trace):
        bp_freqmin, bp_freqmax = cls.get_bandpass_frequencies(config, trace)
        nyquist = 1. / (2. * trace.stats.delta)
        if bp_freqmax >= nyquist:
            bp_freqmax = nyquist * 0.999
            msg = '{}: maximum frequency for bandpass filtering '
            msg += 'is larger or equal to Nyquist. Setting it to {} Hz'
            msg = msg.format(trace.id, bp_freqmax)
            logger.warning(msg)
        filter = dict(type='bandpass', freqmin=bp_freqmin, freqmax=bp_freqmax)
        trace.filter(**filter)
        # save filter info to trace stats
        trace.stats.filter = AttribDict(filter)

    def get_bandpass_frequencies(cls, config, trace):
        """Get frequencies for bandpass filter."""
        # see if there is a station-specific filter
        station = trace.stats.station
        try:
            bp_freqmin = float(config['bp_freqmin_' + station])
            bp_freqmax = float(config['bp_freqmax_' + station])
        except KeyError:
            instrtype = trace.stats.instrtype
            try:
                bp_freqmin = float(config['bp_freqmin_' + instrtype])
                bp_freqmax = float(config['bp_freqmax_' + instrtype])
            except KeyError:
                msg = '{}: Unknown instrument type: {}: skipping trace'
                msg = msg.format(trace.id, instrtype)
                raise ValueError(msg)
        return bp_freqmin, bp_freqmax

    def check_sn_ratio(cls, config, trace):
        # noise time window for s/n ratio
        trace_noise = trace.copy()
        # remove the mean...
        trace_noise.detrend(type='constant')
        # ...and the linear trend...
        trace_noise.detrend(type='linear')
        t1 = trace_noise.stats.arrivals['N1'][1]
        t2 = trace_noise.stats.arrivals['N2'][1]
        trace_noise.trim(starttime=t1, endtime=t2, pad=True, fill_value=0)
        # signal window for s/n ratio
        trace_signal = trace.copy()
        # remove the mean...
        trace_signal.detrend(type='constant')
        # ...and the linear trend...
        trace_signal.detrend(type='linear')
        if config.wave_type[0] == 'S':
            t1 = trace_signal.stats.arrivals['S1'][1]
            t2 = trace_signal.stats.arrivals['S2'][1]
        elif config.wave_type[0] == 'P':
            t1 = trace_signal.stats.arrivals['P1'][1]
            t2 = trace_signal.stats.arrivals['P2'][1]
        trace_signal.trim(starttime=t1, endtime=t2, pad=True, fill_value=0)
        rmsnoise2 = np.power(trace_noise.data, 2).sum()
        rmsnoise = np.sqrt(rmsnoise2)
        rmsS2 = np.power(trace_signal.data, 2).sum()
        rmsS = np.sqrt(rmsS2)
        if rmsnoise == 0:
            if config.weighting == 'noise':
                msg = '{} {}: empty noise window: skipping trace'
                msg = msg.format(trace.id, trace.stats.instrtype)
                raise RuntimeError(msg)
            else:
                msg = '{} {}: empty noise window!'
                msg = msg.format(trace.id, trace.stats.instrtype)
                logger.warning(msg)
                rmsnoise = 1.
        sn_ratio = rmsS / rmsnoise
        logger.info('{} {}: S/N: {:.1f}'.format(
            trace.id, trace.stats.instrtype, sn_ratio))
        trace.stats.sn_ratio = sn_ratio
        snratio_min = config.sn_min
        if sn_ratio < snratio_min:
            msg = '{} {}: S/N smaller than {:g}: skipping trace'
            msg = msg.format(trace.id, trace.stats.instrtype, snratio_min)
            logger.warning(msg)
            trace.stats.ignore = True
            trace.stats.ignore_reason = 'low S/N'

    @classmethod
    def is_clipped(trace, sensitivity, debug=False):
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
        peaks, _ = find_peaks(
            density_weight,
            prominence=min_prominence[sensitivity - 1]
        )
        # if debug:
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        #     fig.suptitle(trace.id)
        #     ax[0].plot(trace.times(), trace.data)
        #     ax[0].set_ylim(-max_data, max_data)
        #     ax[0].set_xlabel('Time (s)')
        #     ax[0].set_ylabel('Amplitude')
        #     ax[1].hist(
        #         bins[:-1], bins=len(counts), weights=counts,
        #         orientation='horizontal')
        #     ax[1].plot(density, density_points, label='kernel density')
        #     ax[1].plot(
        #         density_weight, density_points, label='weighted\nkernel density')
        #     ax[1].scatter(
        #         density_weight[peaks], density_points[peaks],
        #         s=100, marker='x', color='red')
        #     ax[1].set_xlabel('Density')
        #     ax[1].legend()
        #     plt.show()
        # If more than one peak, then the signal is probably clipped or distorted
        if len(peaks) > 1:
            return True
        else:
            return False