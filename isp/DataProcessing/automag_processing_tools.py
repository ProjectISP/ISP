import math
import numpy as np
from obspy.taup import TauPyModel
from scipy.interpolate import interp1d
from obspy.signal.invsim import cosine_taper as _cos_taper
from obspy.core import Stream
from scipy.signal import argrelmax
from scipy.optimize import curve_fit, minimize, basinhopping
import logging
from scipy.signal import peak_widths
from scipy.interpolate import griddata
from scipy.signal._peak_finding_utils import PeakPropertyWarning
import itertools
import warnings
from automag_statistics import SpectralParameter, StationParameters, SourceSpecOutput
logger = logging.getLogger(__name__.split('.')[-1])

class signal_preprocess_tools:

        def __init__(self, tr_noise, tr_signal, hypo_dist_in_km, geom_spread_model, geom_spread_n_exponent,
                                geom_spread_cutoff_distance, event_info, rho, spectral_smooth_width_decades,
                     spectral_sn_min, spectral_sn_freq_range):

            self.tr_noise = tr_noise
            self.tr_signal = tr_signal
            self.hypo_dist_in_km = hypo_dist_in_km
            self.geom_spread_model = geom_spread_model
            self.geom_spread_n_exponent = geom_spread_n_exponent
            self.geom_spread_cutoff_distance = geom_spread_cutoff_distance
            self.event_info = event_info
            model = TauPyModel(model='iasp91')
            self.v_model = model.model.s_mod.v_mod
            # Density (kg/m3):
            # warning for now is hardcoded
            self.rho = rho
            #self.rho = 2500
            self.spectral_smooth_width_decades = spectral_smooth_width_decades
            self.spectral_sn_min = spectral_sn_min
            self.spectral_sn_freq_range = spectral_sn_freq_range
            self.spectral_sn = True
            self.weight = None
            self.weight_log = None
            self.spectral_snratio = None

        def do_spectrum(self):
            """Compute the spectrum of an ObsPy Trace object."""
            spectrum = {}
            Id_list = []
            #print(self.tr_signal)
            for tr_signal, tr_noise in zip(self.tr_signal, self.tr_noise):

                N = max([len(tr_signal.data), len(tr_noise.data)])
                DD = 2 ** math.ceil(math.log2(N))

                #process before fft
                tr_signal.detrend(type="simple")
                tr_signal.taper(type="blackman", max_percentage=0.05)
                tr_noise.detrend(type="simple")
                tr_noise.taper(type="blackman", max_percentage=0.05)

                signal = tr_signal.data
                noise = tr_noise.data
                delta = tr_signal.stats.delta

                full_period_signal = 1/(delta*len(signal))
                full_period_noise = 1 / (delta*len(noise))

                amp_signal, freq_signal = self.__dofft(signal, delta, DD)
                amp_noise, freq_noise = self.__dofft(noise, delta, DD)
                # remove DC component (freq=0)
                amp_signal = abs(amp_signal)[1:]
                freq_signal = freq_signal[1:]
                amp_noise = abs(amp_noise)[1:]
                freq_noise = freq_noise[1:]
                geom_spread_signal = self.geometrical_spreading_coefficient(freq_signal)
                geom_spread_noise = self.geometrical_spreading_coefficient(freq_noise)
                amp_signal *= geom_spread_signal
                amp_noise *= geom_spread_noise
                amp_signal_moment = amp_signal
                amp_noise_moment = amp_noise
                coeff_signal_rp, coeff_signal_rp, vs = self.__displacement_to_moment()
                coeff_noise_rp, coeff_noise_rs, vs = self.__displacement_to_moment()
                amp_signal *= coeff_signal_rp
                amp_noise *= coeff_noise_rp
                amp_signal_log, freq_signal_log = self.__smooth_spectrum(amp_signal, freq_signal,
                                                        smooth_width_decades=self.spectral_smooth_width_decades)
                amp_noise_log, freq_noise_log = self.__smooth_spectrum(amp_noise, freq_noise,
                                                        smooth_width_decades=self.spectral_smooth_width_decades)

                mag_signal = magnitude_aux_tools.moment_to_mag(amp_signal)
                mag_signal_log = magnitude_aux_tools.moment_to_mag(amp_signal_log)
                mag_noise = magnitude_aux_tools.moment_to_mag(amp_noise)
                Id_list.append(tr_signal.id)
                if self.__check_spectral_sn_ratio(amp_signal, amp_noise, freq_signal, freq_signal_log, delta):

                    spectrum[tr_signal.id] = {"amp_signal": amp_signal, "freq_signal": freq_signal, "amp_noise": amp_noise,
                    "freq_noise": freq_noise, "amp_signal_log": amp_signal_log, "freq_signal_log": freq_signal_log,
                    "amp_noise_log": amp_noise_log, "freq_noise_log": freq_noise_log, "weights": self.weight, "weigh_log": self.weight_log,
                    "spectral_snratio": self.spectral_snratio, "mag_signal": mag_signal, "mag_noise": mag_noise,
                    "mag_signal_log": mag_signal_log, "vs": vs, "amp_signal_moment": amp_signal_moment,
                    "amp_noise_moment": amp_noise_moment, "full_period_signal":full_period_signal,
                                              "full_period_noise":full_period_noise}

                else:
                    spectrum[tr_signal.id] = None

            return spectrum

        def geometrical_spreading_coefficient(self, freq_signal):

            if self.geom_spread_model == 'r_power_n':
                exponent = self.geom_spread_n_exponent
                return self.__geom_spread_r_power_n(self.hypo_dist_in_km, exponent)
            elif self.geom_spread_model == 'boatwright':
                cutoff_dist_in_km = self.geom_spread_cutoff_distance
                return self.__geom_spread_boatwright(self.hypo_dist_in_km, cutoff_dist_in_km, freq_signal)

        def __geom_spread_r_power_n(self, hypo_dist_in_km, exponent):
            """rⁿ geometrical spreading coefficient."""
            dist = hypo_dist_in_km * 1e3
            coeff = dist ** exponent
            return coeff

        def __geom_spread_boatwright(self, hypo_dist_in_km, cutoff_dist_in_km, freqs):
            """"
            Geometrical spreading coefficient from Boatwright et al. (2002), eq. 8.

            Except that we take the square root of eq. 8, since we correct amplitude
            and not energy.
            """
            dist = hypo_dist_in_km * 1e3
            cutoff_dist = cutoff_dist_in_km * 1e3
            if dist <= cutoff_dist:
                coeff = dist
            else:
                exponent = np.ones_like(freqs)
                low_freq = freqs <= 0.2
                mid_freq = np.logical_and(freqs > 0.2, freqs <= 0.25)
                high_freq = freqs >= 0.25
                exponent[low_freq] = 0.5
                exponent[mid_freq] = 0.5 + 2 * np.log(5 * freqs[mid_freq])
                exponent[high_freq] = 0.7
                coeff = cutoff_dist * (dist / cutoff_dist) ** exponent
            return coeff

        def __dofft(self, signal, delta, npts):
            """Compute the complex Fourier transform of a signal."""

            # TODO: needs to ask to Claudio, why multiply by delta the fft?
            fft = np.fft.rfft(signal, n=npts) * delta
            fftfreq = np.fft.rfftfreq(npts, d=delta)

            return fft, fftfreq

        def __displacement_to_moment(self):
            """
            Return the coefficient for converting displacement to seismic moment.

            From Aki&Richards,1980
            """

            v_hypo, vs = self.get_vel_from_taup(depth_event_km=self.event_info[3])
            v_station, vs = self.get_vel_from_taup()

            v_hypo *= 1000.
            v_station *= 1000.
            v3 = v_hypo ** (5. / 2) * v_station ** (1. / 2)
            rpp, rps = self.__get_radiation_pattern_coefficient()
            coeff_rp = 4 * math.pi * v3 * self.rho / (2 * rpp)
            coeff_rs = 4 * math.pi * v3 * self.rho / (2 * rps)

            return coeff_rp, coeff_rs, vs

        def __smooth_spectrum(self, amplitude, freq, smooth_width_decades=0.2):
            """Smooth spectrum in a log10-freq space."""
            # 1. Generate log10-spaced frequencies
            _log_freq = np.log10(freq)
            # frequencies in logarithmic spacing
            log_df = _log_freq[-1] - _log_freq[-2]
            freq_logspace = \
                10 ** (np.arange(_log_freq[0], _log_freq[-1] + log_df, log_df))
            # 2. Reinterpolate data using log10 frequencies
            # make sure that extrapolation does not create negative values
            f = interp1d(freq, amplitude, fill_value='extrapolate')
            data_logspace = f(freq_logspace)
            data_logspace[data_logspace <= 0] = np.min(amplitude)
            # 3. Smooth log10-spaced data points
            npts = max(1, int(round(smooth_width_decades / log_df)))
            data_logspace = self.__smooth(data_logspace, window_len=npts)
            # 4. Reinterpolate to linear frequencies
            # make sure that extrapolation does not create negative values
            f = interp1d(freq_logspace, data_logspace, fill_value='extrapolate')
            data = f(freq)
            data[data <= 0] = np.min(amplitude)

            # 5. Optimize the sampling rate of log spectrum,
            #    based on the width of the smoothing window
            # make sure that extrapolation does not create negative values
            log_df = smooth_width_decades / 5
            freq_logspace = \
                10 ** (np.arange(_log_freq[0], _log_freq[-1] + log_df, log_df))
            freq_log = freq_logspace
            data_logspace = f(freq_logspace)
            data_logspace[data_logspace <= 0] = np.min(data)
            return data_logspace, freq_log

        def __check_spectral_sn_ratio(self, spec, specnoise, freqs, freqs_log, delta):
            self.weight, self.weight_log = self._build_weight_from_noise(spec, specnoise, freqs, freqs_log, delta)
            if self.spectral_sn_freq_range is not None:
                sn_fmin, sn_fmax = self.spectral_sn_freq_range
                idx = np.where((sn_fmin <= freqs) * (freqs <= sn_fmax))
            else:
                idx = range(len(spec))
            self.spectral_snratio = spec[idx].sum() / len(spec[idx])
            ssnmin = self.spectral_sn_min
            if self.spectral_snratio < ssnmin:
                return False
            else:
                return True

        def _build_weight_from_noise(self, spec, specnoise, freq, freq_log, delta):

            weight_data = spec.copy()
            if specnoise is None or np.all(specnoise == 0):
                weight_data = np.ones(len(spec))
                weight_data_raw = np.ones(len(spec))
            else:
                weight_data /= specnoise

                # save data to raw_data
                weight_data_raw = weight_data.copy()
                # The inversion is done in magnitude units,
                # so let's take log10 of weight
                weight_data = np.log10(weight_data)
                # Weight spectrum is smoothed once more
                weight_data_log, freq_log = self.__smooth_spectrum(weight_data, freq, self.spectral_smooth_width_decades)
                weight_data /= np.max(weight_data)
                # slightly taper weight at low frequencies, to avoid overestimating
                # weight at low frequencies, in cases where noise is underestimated
                weight_data = self.__cosine_taper(weight_data, delta / 4, left_taper=True)
                # Make sure weight is positive
                weight_data[weight_data <= 0] = 0.001
            # interpolate to log-frequencies
            f = interp1d(freq, weight_data, fill_value='extrapolate')
            weight_data_log = f(freq_log)
            weight_data_log /= np.max(weight_data_log)
            # Make sure weight is positive
            weight_data_log[weight_data_log <= 0] = 0.001

            return weight_data, weight_data_log

        def _smooth(self, x, window_len=11, window='hanning'):
            if x.ndim != 1:
                raise ValueError('smooth only accepts 1 dimension arrays.')
            if x.size < window_len:
                raise ValueError('Input vector needs to be bigger than window size.')
            if window_len < 3:
                return x
            if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is one of 'flat', 'hanning', 'hamming', "
                                 "'bartlett', 'blackman'")
            s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
            if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
            else:
                w = eval('np.' + window + '(window_len)')
            y = np.convolve(w / w.sum(), s, mode='same')

            yy = y[window_len:-window_len + 1]
            # check if there are NaN values
            nanindexes = np.where(np.isnan(yy))
            yy[nanindexes] = x[nanindexes]
            return yy

        def __cosine_taper(self, signal, width, left_taper=False):
            # TODO: this taper looks more like a hanning...
            npts = len(signal)
            p = 2 * width
            tap = _cos_taper(npts, p)
            if left_taper:
                tap[int(npts / 2):] = 1.
            signal *= tap
            return signal

        def get_vel_from_taup(self, **kwargs):

            depth_event_km = kwargs.pop('depth_event_km', 1e-3)
            vp = self.v_model.evaluate_above(depth_event_km, "P")[0]
            vs = self.v_model.evaluate_above(depth_event_km, "S")[0]

            return vp, vs

        def __get_radiation_pattern_coefficient(self):

            # TODO FROM FOCAL MECHANISM

            #rp = abs(radiation_pattern(strike, dip, rake, takeoff_angle, azimuth))
            # we are interested only in amplitude
            # (P, SV and SH radiation patterns have a sign)


            # P-wave average radiation pattern coefficient:
            rpp = 0.52
            # S-wave average radiation pattern coefficient:
            rps = 0.62

            return rpp,rps

        # modified from: http://stackoverflow.com/q/5515720
        def __smooth(self, x, window_len=11, window='hanning'):
            if x.ndim != 1:
                raise ValueError('smooth only accepts 1 dimension arrays.')
            if x.size < window_len:
                raise ValueError('Input vector needs to be bigger than window size.')
            if window_len < 3:
                return x
            if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is one of 'flat', 'hanning', 'hamming', "
                                 "'bartlett', 'blackman'")
            s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
            if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
            else:
                w = eval('np.' + window + '(window_len)')
            y = np.convolve(w / w.sum(), s, mode='same')

            yy = y[window_len:-window_len + 1]
            # check if there are NaN values
            nanindexes = np.where(np.isnan(yy))
            yy[nanindexes] = x[nanindexes]
            return yy

        #
        # def __select_spectra(spec_st, specid):
        #     """Select spectra from stream, based on specid."""
        #     network, station, location, channel = specid.split('.')
        #     channel = channel + '?' * (3 - len(channel))
        #     spec_st_sel = spec_st.select(
        #         network=network, station=station, location=location, channel=channel)
        #     spec_st_sel = Stream(sp for sp in spec_st_sel if not sp.stats.ignore)
        #     return spec_st_sel


class ssp_inversion:

        def __init__(self, spectrum_dict, t_star_0_variability, invert_t_star_0, t_star_0, event_info, arrival,
                     inv_selected, bound_config, inv_algorithm, pi_misfit_max, pi_t_star_min_max, pi_fc_min_max,
                     pi_bsd_min_max):

            self.spectrum_dict = spectrum_dict
            self.t_star_0_variability = t_star_0_variability
            self.invert_t_star_0 = invert_t_star_0
            self.t_star_0 = t_star_0
            self.arrival = arrival
            self.event_info = event_info
            self.inv = inv_selected
            self.bound_config = bound_config
            self.inv_algorithm = inv_algorithm
            self.t_star_0 = t_star_0
            self.pi_misfit_max = pi_misfit_max
            self.pi_t_star_min_max = pi_t_star_min_max
            self.pi_fc_min_max = pi_fc_min_max
            self.pi_bsd_min_max = pi_bsd_min_max

            """
            spectrum[tr_signal.id] = {"amp_signal": amp_signal, "freq_signal": freq_signal, "amp_noise": amp_noise,
                    "freq_noise":freq_noise, "amp_signal_log": amp_signal_log, "freq_signal_log":freq_signal_log,
                    "amp_noise_log":amp_noise_log, "freq_noise_log": freq_noise_log, "weights": self.weights, "weigh_log": self.weigh_log,
                    "spectral_snratio": self.spectral_snratio, "mag_signal": mag_signal, "mag_noise": mag_noise}
            """


        def run_estimate_all_traces(self):

            all_channels_station_pars = []
            for keyId, trace_dict in self.spectrum_dict.items():

                spec = trace_dict["amp_signal"]
                spec_log_mag = trace_dict["mag_signal_log"]
                weight = trace_dict["weights"]
                weight_log = trace_dict["weigh_log"]
                freq_log = trace_dict["freq_signal_log"]
                vs = trace_dict["vs"]
                try:
                    station_pars = self.run_estimate(keyId, self.inv_algorithm, spec, spec_log_mag, weight, weight_log,
                            self.bound_config, freq_log, self.t_star_0, self.pi_misfit_max, self.pi_t_star_min_max,
                            self.pi_fc_min_max, self.pi_bsd_min_max, vs, invert_t_star_0=True)
                    all_channels_station_pars.append(station_pars)
                except:
                    pass

            return all_channels_station_pars


        def _freq_ranges_for_Mw0_and_tstar0(self, weight, freq_log, keyId):

            fc_0 = 3.0
            # we start where signal-to-noise becomes strong
            idx0 = np.where(weight > 0.5)[0][0]
            # we stop at the first max of signal-to-noise (proxy for fc)
            idx_max = argrelmax(weight)[0]
            # just keep the indexes for maxima > 0.5
            idx_max = [idx for idx in idx_max if weight[idx] > 0.5]
            if not idx_max:
                # if idx_max is empty, then the source and/or noise spectrum
                # is most certainly "strange". In this case, we simply give up.
                msg = '{}: unable to find a frequency range to compute Mw_0. '
                msg += 'This is possibly due to an uncommon spectrum '
                msg += '(e.g., a resonance).'
                msg = msg.format(keyId)
                raise RuntimeError(msg)
            idx1 = idx_max[0]
            if idx1 == idx0:
                try:
                    idx1 = idx_max[1]
                except IndexError:
                    # if there are no other maxima, just take 5 points
                    idx1 = idx0 + 5

            fc_0 = freq_log[idx1]

            return idx0, idx1, fc_0

        def _set_parameters(self, weight):
            yerr = 1. / np.sqrt(weight)
            return yerr

        def run_estimate(self, keyId, inv_algorithm, spec, spec_log_mag, weight, weight_log, bounds_config,
                                    freq_log, t_star_0, pi_misfit_max, pi_t_star_min_max, pi_fc_min_max, pi_bsd_min_max, vs,
                                    invert_t_star_0=True):

            hyp_dist = self.arrival["distance_km"]
            epi_dist = self.arrival["distance_km"]
            stla = self.event_info[1]
            stlo = self.event_info[2]
            az = self.arrival["azimuth"]
            travel_time = self.arrival["travel_time"]

            yerr = self._set_parameters(weight_log)
            idx0, idx1, fc_0 = self._freq_ranges_for_Mw0_and_tstar0(weight_log, freq_log, keyId)
            t_star_min = t_star_max = None
            # if invert_t_star_0:
            #     # fit t_star_0 and Mw on the initial part of the spectrum,
            #     # corrected for the effect of fc
            #     ydata_corr = spec_log_mag - magnitude_aux_tools.spectral_model(freq_log, Mw=0, fc=fc_0, t_star=0)
            #     ydata_corr = magnitude_aux_tools.smooth(ydata_corr, window_len=18)
            #     slope, Mw_0 = np.polyfit(freq_log[idx0: idx1], ydata_corr[idx0: idx1], deg=1)
            #     t_star_0 = -3. / 2 * slope / (np.pi * np.log10(np.e))
            #     t_star_min = t_star_0 * (1 - self.t_star_0_variability)
            #     t_star_max = t_star_0 * (1 + self.t_star_0_variability)

            #if not invert_t_star_0 or t_star_0 < 0:
                 # we calculate the initial value for Mw as an average
            Mw_0 = np.nanmean(spec_log_mag[idx0: idx1])
            t_star_0 = self.t_star_0

            initial_values = InitialValues(Mw_0, fc_0, t_star_0)
            logger.info('{}: initial values: {}'.format(keyId, str(initial_values)))
            bounds = Bounds(keyId, bounds_config, spec, hyp_dist, initial_values)
            bounds.Mw_min = np.nanmin(spec_log_mag[idx0: idx1]) * 0.9
            bounds.Mw_max = np.nanmax(spec_log_mag[idx0: idx1]) * 1.1
            if t_star_min is not None:
                bounds.t_star_min = t_star_min
            if t_star_max is not None:
                bounds.t_star_max = t_star_max
            logger.info('{}: bounds: {}'.format(keyId, str(bounds)))
            try:

                params_opt, params_err, misfit = magnitude_aux_tools._curve_fit(inv_algorithm, freq_log, spec_log_mag,
                                                    weight_log, yerr, initial_values, bounds)
            except (RuntimeError, ValueError) as m:
                msg = str(m) + '\n'
                msg += '{}: unable to fit spectral model'.format(keyId)
                raise RuntimeError(msg)

            Mw, fc, t_star = params_opt
            Mw_err, fc_err, t_star_err = params_err

            inverted_par_str = 'Mw: {:.4f}; fc: {:.4f}; t_star: {:.4f}'.format(
                Mw, fc, t_star)
            logger.info('{}: optimal values: {}'.format(keyId, inverted_par_str))
            logger.info('{}: misfit: {:.3f}'.format(keyId, misfit))

            if np.isclose(fc, bounds.fc_min, rtol=0.1):
                msg = '{}: optimal fc within 10% of fc_min: {:.3f} ~= {:.3f}: '
                msg += 'ignoring inversion results'
                msg = msg.format(keyId, fc, bounds.fc_min)
                raise ValueError(msg)

            if np.isclose(fc, bounds.fc_max, rtol=1e-4):
                msg = '{}: optimal fc within 10% of fc_max: {:.3f} ~= {:.3f}: '
                msg += 'ignoring inversion results'
                msg = msg.format(keyId, fc, bounds.fc_max)
                raise ValueError(msg)

            misfit_max = pi_misfit_max or np.inf
            if misfit > misfit_max:
                msg = '{}: misfit larger than pi_misfit_max: {:.3f} > {:.3f}: '
                msg += 'ignoring inversion results'
                msg = msg.format(keyId, misfit, misfit_max)
                raise ValueError(msg)

            # Check post-inversion bounds for t_star and fc
            pi_t_star_min, pi_t_star_max = \
                pi_t_star_min_max or (-np.inf, np.inf)
            if not (pi_t_star_min <= t_star <= pi_t_star_max):
                msg = '{}: t_star: {:.3f} not in allowed range [{:.3f}, {:.3f}]: '
                msg += 'ignoring inversion results'
                msg = msg.format(keyId, t_star, pi_t_star_min, pi_t_star_max)
                raise ValueError(msg)
            pi_fc_min, pi_fc_max = pi_fc_min_max or (-np.inf, np.inf)
            if not (pi_fc_min <= fc <= pi_fc_max):
                msg = '{}: fc: {:.3f} not in allowed range [{:.3f}, {:.3f}]: '
                msg += 'ignoring inversion results'
                msg = msg.format(keyId, fc, pi_fc_min, pi_fc_max)
                raise ValueError(msg)


            station_pars = StationParameters(id=keyId, instrument_type="N/A",
                                             latitude=stla, longitude=stlo, hypo_dist_in_km=hyp_dist,
                                             epi_dist_in_km=epi_dist, azimuth=az)

            station_pars.Mw = SpectralParameter(
                id='Mw', value=Mw,
                lower_uncertainty=Mw_err[0], upper_uncertainty=Mw_err[1],
                confidence_level=68.2, format='{:.2f}')

            station_pars.fc = SpectralParameter(
                id='fc', value=fc,
                lower_uncertainty=fc_err[0], upper_uncertainty=fc_err[1],
                confidence_level=68.2, format='{:.3f}')

            station_pars.t_star = SpectralParameter(
                id='t_star', value=t_star,
                lower_uncertainty=t_star_err[0], upper_uncertainty=t_star_err[1],
                confidence_level=68.2, format='{:.3f}')

            # seismic moment
            station_pars.Mo = SpectralParameter(
                id='Mw', value=moment_tools.mag_to_moment(Mw), format='{:.3e}')
            # source radius in meters
            station_pars.radius = SpectralParameter(
                id='radius', value=moment_tools.source_radius(fc, vs * 1e3), format='{:.3f}')
            # Brune stress drop in MPa
            station_pars.bsd = SpectralParameter(
                id='bsd', value=moment_tools.bsd(station_pars.Mo.value, station_pars.radius.value),
                format='{:.3e}')
            # quality factor

            station_pars.Qo = SpectralParameter(
                id='Qo', value=moment_tools.quality_factor(travel_time, t_star), format='{:.1f}')

            # Check post-inversion bounds for bsd
            pi_bsd_min, pi_bsd_max = pi_bsd_min_max or (-np.inf, np.inf)
            if not (pi_bsd_min <= station_pars.bsd.value <= pi_bsd_max):
                msg = '{}: bsd: {:.3e} not in allowed range [{:.3e}, {:.3e}]: '
                msg += 'ignoring inversion results'
                msg = msg.format(keyId, station_pars.bsd.value, pi_bsd_min, pi_bsd_max)
                raise ValueError(msg)

                # additional parameter errors, computed from fc, Mw and t_star
                # seismic moment
                Mw_min = Mw - Mw_err[0]
                Mw_max = Mw + Mw_err[1]
                Mo_min = mag_to_moment(Mw_min)
                Mo_max = mag_to_moment(Mw_max)
                station_pars.Mo.lower_uncertainty = station_pars.Mo.value - Mo_min
                station_pars.Mo.upper_uncertainty = Mo_max - station_pars.Mo.value
                station_pars.Mo.confidence_level = 68.2
                # source radius in meters
                fc_min = fc - fc_err[0]
                if fc_min <= 0:
                    fc_min = freq_log[0]
                fc_max = fc + fc_err[1]
                radius_min = source_radius(fc_max, vs * 1e3)
                radius_max = source_radius(fc_min, vs * 1e3)
                station_pars.radius.lower_uncertainty = \
                    station_pars.radius.value - radius_min
                station_pars.radius.upper_uncertainty = \
                    radius_max - station_pars.radius.value
                station_pars.radius.confidence_level = 68.2
                # Brune stress drop in MPa
                bsd_min = bsd(Mo_min, radius_max)
                bsd_max = bsd(Mo_max, radius_min)
                station_pars.bsd.lower_uncertainty = station_pars.bsd.value - bsd_min
                station_pars.bsd.upper_uncertainty = bsd_max - station_pars.bsd.value
                station_pars.bsd.confidence_level = 68.2
                # quality factor
                t_star_min = t_star - t_star_err[0]
                if t_star_min <= 0:
                    t_star_min = 0.001
                t_star_max = t_star + t_star_err[1]
                Qo_min = quality_factor(travel_time, t_star_max)
                Qo_max = quality_factor(travel_time, t_star_min)
                station_pars.Qo.lower_uncertainty = station_pars.Qo.value - Qo_min
                station_pars.Qo.upper_uncertainty = Qo_max - station_pars.Qo.value
                station_pars.Qo.confidence_level = 68.2

            return station_pars


class magnitude_aux_tools:

        def __init__(self):
            self.moment = None

        @classmethod
        def moment_to_mag(cls, moment):
            """Convert moment to magnitude."""
            return (np.log10(moment) - 9.1) / 1.5

        @classmethod
        def mag_to_moment(cls, magnitude):
            """Convert magnitude to moment."""
            return np.power(10, (1.5 * magnitude + 9.1))

        @classmethod
        def source_radius(cls, fc_in_hz, vs_in_m_per_s):
            """
            Compute source radius in meters.

            Madariaga (2009), doi:10.1007/978-1-4419-7695-6_22, eq. 31
            """
            return 0.3724 * vs_in_m_per_s / fc_in_hz

        @classmethod
        def bsd(cls, Mo_in_N_m, ra_in_m):
            """
            Compute Brune stress drop in MPa.

            Madariaga (2009), doi:10.1007/978-1-4419-7695-6_22, eq. 27
            """
            return 7. / 16 * Mo_in_N_m / ra_in_m ** 3 * 1e-6

        @classmethod
        def quality_factor(cls, travel_time_in_s, t_star_in_s):
            """Compute quality factor from travel time and t_star."""
            if t_star_in_s == 0:
                return np.inf
            return travel_time_in_s / t_star_in_s

        @classmethod
        def smooth(cls, x, window_len=11, window='hanning'):
            if x.ndim != 1:
                raise ValueError('smooth only accepts 1 dimension arrays.')
            if x.size < window_len:
                raise ValueError('Input vector needs to be bigger than window size.')
            if window_len < 3:
                return x
            if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is one of 'flat', 'hanning', 'hamming', "
                                 "'bartlett', 'blackman'")
            s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
            if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
            else:
                w = eval('np.' + window + '(window_len)')
            y = np.convolve(w / w.sum(), s, mode='same')

            yy = y[window_len:-window_len + 1]
            # check if there are NaN values
            nanindexes = np.where(np.isnan(yy))
            yy[nanindexes] = x[nanindexes]
            return yy

        @classmethod
        def spectral_model(cls, freq, Mw, fc, t_star, alpha=1.):
            r"""
            Spectral model.

            .. math::

               Y_{data} = M_w + \frac{2}{3} \left[ - \log_{10} \left(
                                1+\left(\frac{f}{f_c}\right)^2 \right) -
                                \pi \, f t^* \log_{10} e \right]

            see :ref:`Theoretical Background <theoretical_background>`
            for a detailed derivation of this model.
            """
            # log S(w)= log(coeff*Mo) + log((1/(1+(w/wc)^2)) + \
            #           log (exp (- w *t_star/2))
            # attenuation model: exp[-pi t* f] with t*=T /Q
            loge = math.log10(math.e)
            return (Mw -
                    (2. / 3.) * np.log10(1. + np.power((freq / fc), 2)) -
                    (2. / 3.) * loge * (math.pi * np.power(freq, alpha) * t_star))

        @classmethod
        def objective_func(cls, xdata, ydata, weight):
            """Objective function generator for bounded inversion."""
            errsum = np.sum(weight)

            def _objective_func(params):
                # params components should be np.float
                if len(params) == 4:
                    model = cls.spectral_model(xdata, params[0], params[1],
                                           params[2], params[3])
                else:
                    model = cls.spectral_model(xdata, params[0], params[1], params[2])
                res = np.array(ydata) - np.array(model)
                res2 = np.power(res, 2)
                wres = np.array(weight) * np.array(res2)
                return np.sqrt(np.sum(wres) / errsum)

            return _objective_func

        @classmethod
        def spectral_model(cls, freq, Mw, fc, t_star, alpha=1.):
            r"""
            Spectral model.

            .. math::

               Y_{data} = M_w + \frac{2}{3} \left[ - \log_{10} \left(
                                1+\left(\frac{f}{f_c}\right)^2 \right) -
                                \pi \, f t^* \log_{10} e \right]

            see :ref:`Theoretical Background <theoretical_background>`
            for a detailed derivation of this model.
            """
            # log S(w)= log(coeff*Mo) + log((1/(1+(w/wc)^2)) + \
            #           log (exp (- w *t_star/2))
            # attenuation model: exp[-pi t* f] with t*=T /Q
            loge = math.log10(math.e)
            return (Mw -
                    (2. / 3.) * np.log10(1. + np.power((freq / fc), 2)) -
                    (2. / 3.) * loge * (math.pi * np.power(freq, alpha) * t_star))

        @classmethod
        def callback(cls, x):
            pass

        @classmethod
        def _curve_fit(cls, inv_algorithm, freq_log, data_log_mag, weight, yerr, initial_values, bounds):

            """
            Curve fitting.

            Available algorithms:
              - Levenberg-Marquardt (LM, via `curve_fit()`). Automatically switches to
                Trust Region Reflective algorithm if bounds are provided.
              - Truncated Newton algorithm (TNC) with bounds.
              - Basin-hopping (BH)
              - Grid search (GS)
            """

            ydata = data_log_mag
            minimize_func = cls.objective_func(freq_log, ydata, weight)
            x0 = initial_values.get_params0()
            if inv_algorithm == 'TNC':
                res = minimize(minimize_func, x0=x0, method='TNC', callback=cls.callback,
                               bounds=bounds.bounds)
                params_opt = res.x
                # trick: use curve_fit() bounded to params_opt
                # to get the covariance
                #, xdata, ydata, p0 = None, sigma = None, absolute_sigma = False,
                #check_finite = True, bounds = (-np.inf, np.inf), method = None,
                #jac = None, ** kwargs
                _, params_cov = curve_fit(cls.spectral_model, freq_log, ydata, p0=params_opt, sigma=yerr,
                   bounds=(params_opt - (1e-10), params_opt + (1e-10)))
                # _, params_cov = curve_fit(cls.spectral_model, freq_log, ydata, p0=params_opt, sigma=yerr,
                #     bounds=(params_opt - (0.01*params_opt), params_opt + (0.01*params_opt)))
                err = np.sqrt(params_cov.diagonal())
                # symmetric error
                params_err = ((e, e) for e in err)
            elif inv_algorithm == 'LM':
                bnds = bounds.get_bounds_curve_fit()
                if bnds is not None:
                    logger.info(
                        'Trying to use using Levenberg-Marquardt '
                        'algorithm with bounds. Switching to the '
                        'Trust Region Reflective algorithm.'
                    )
                params_opt, params_cov = curve_fit(cls.spectral_model, freq_log, ydata, p0=initial_values.get_params0(), sigma=yerr,
                    bounds=bnds)
                err = np.sqrt(params_cov.diagonal())
                # symmetric error
                params_err = ((e, e) for e in err)
            elif inv_algorithm == 'BH':
                res = basinhopping(minimize_func, x0=initial_values.get_params0(), niter=100, accept_test=bounds)
                params_opt = res.x
                # trick: use curve_fit() bounded to params_opt
                # to get the covariance
                _, params_cov = curve_fit(cls.spectral_model, freq_log, ydata, p0=params_opt, sigma=yerr,
                    bounds=(params_opt - (1e-10), params_opt + (1e-10)))
                err = np.sqrt(params_cov.diagonal())
                # symmetric error
                params_err = ((e, e) for e in err)
            elif inv_algorithm in ['GS', 'IS']:
                nsteps = (20, 150, 150)  # we do fewer steps in magnitude
                sampling_mode = ('lin', 'log', 'lin')
                params_name = ('Mw', 'fc', 't_star')
                params_unit = ('', 'Hz', 's')
                grid_sampling = GridSampling(minimize_func, bounds.bounds, nsteps, sampling_mode, params_name,
                                             params_unit)
                if inv_algorithm == 'GS':
                    grid_sampling.grid_search()
                elif inv_algorithm == 'IS':
                    grid_sampling.kdtree_search()
                params_opt = grid_sampling.params_opt
                params_err = grid_sampling.params_err

            misfit = minimize_func(params_opt)
            return params_opt, params_err, misfit


class InitialValues():
    """Initial values for spectral inversion."""

    def __init__(self, Mw_0=None, fc_0=None, t_star_0=None):
        self.Mw_0 = Mw_0
        self.fc_0 = fc_0
        self.t_star_0 = t_star_0

    def __str__(self):
        """String representation."""
        s = 'Mw_0: %s; ' % round(self.Mw_0, 4)
        s += 'fc_0: %s; ' % round(self.fc_0, 4)
        s += 't_star_0: %s' % round(self.t_star_0, 4)
        return s

    def get_params0(self):

        return (self.Mw_0, self.fc_0, self.t_star_0)


class Bounds(object):
    """Bounds for bounded spectral inversion."""

    def __init__(self, keyId, config, spec, hypo_dist, initial_values):
        self.config = config
        self.keyId = keyId
        self.spec = spec
        self.hd = hypo_dist
        self.ini_values = initial_values
        self.Mw_min = self.Mw_max = None
        self._set_fc_min_max()
        if config["Qo_min_max"] is None:
            self.t_star_min, self.t_star_max =\
                self._check_minmax(config["t_star_min_max"])
        else:
            self.t_star_min, self.t_star_max = self._Qo_to_t_star()
        self._fix_initial_values_t_star()

    def __str__(self):
        """String representation."""
        s = 'Mw: {}, {}; '.format(
            *[round(x, 4) if x is not None else x for x in self.bounds[0]])
        s += 'fc: {}, {}; '.format(
            *[round(x, 4) if x is not None else x for x in self.bounds[1]])
        s += 't_star: {}, {}'.format(
            *[round(x, 4) if x is not None else x for x in self.bounds[2]])
        return s

    def _set_fc_min_max(self):
        fc_0 = self.ini_values.fc_0
        if self.config["fc_min_max"] is None:
            # If no bound is given, set it to fc_0 +/- a decade
            scale = 10.  # a decade
            self.fc_min = fc_0/scale
            self.fc_max = fc_0*scale
        else:
            self.fc_min, self.fc_max = self.config["fc_min_max"]
        if self.fc_min > fc_0:
            logger.warning(
                '{} : fc_min ({}) larger than fc_0 ({}). '
                'Using fc_0 instead.'.format(self.keyId, self.fc_min, round(fc_0, 4)))
            self.fc_min = fc_0
        if self.fc_max < fc_0:
            logger.warning(
                '{}: fc_max ({}) smaller than fc_0 ({}). ''Using fc_0 instead.'.format(self.spec.id, self.fc_max,
                                                                                       round(fc_0, 4)))
            self.fc_max = fc_0

    def _check_minmax(self, minmax):
        if minmax is None:
            return (None, None)
        else:
            return minmax

    def _Qo_to_t_star(self):
        phase = self.config.wave_type[0]
        travel_time = self.spec.stats.travel_times[phase]
        t_star_bounds = travel_time/self.config.Qo_min_max
        return sorted(t_star_bounds)

    def _fix_initial_values_t_star(self):
        if self.ini_values.t_star_0 is not None:
            return
        if None in self.bounds[2]:
            return
        if self.t_star_min < self.ini_values.t_star_0 < self.t_star_max:
            return
        t_star_0 = (self.t_star_max + self.t_star_min) / 2.
        logger.warning(
            '{} {}: initial t_star value ({}) outside '
            'bounds. Using bound average ({})'.format(
                self.spec.id, self.spec.stats.instrtype,
                self.ini_values.t_star_0, round(t_star_0, 4))
        )
        self.ini_values.t_star_0 = t_star_0

    def __call__(self, **kwargs):
        """Interface for basin-hopping."""
        params = kwargs['x_new']
        params_min = np.array(
            (self.Mw_min, self.fc_min, self.t_star_min)).astype(float)
        params_max = np.array(
            (self.Mw_max, self.fc_max, self.t_star_max)).astype(float)
        params_min[np.isnan(params_min)] = 1e-99
        params_max[np.isnan(params_min)] = 1e+99
        tmin = bool(np.all(params >= params_min))
        tmax = bool(np.all(params <= params_max))
        return tmin and tmax

    @property
    def bounds(self):
        """Get bounds for minimize() as sequence of (min, max) pairs."""
        self._bounds = ((self.Mw_min, self.Mw_max),
                        (self.fc_min, self.fc_max),
                        (self.t_star_min, self.t_star_max))
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        """Set bounds from a sequence of three (min, max) pairs."""
        self._bounds = value
        self.Mw_min, self.Mw_max = value[0]
        self.fc_min, self.fc_max = value[1]
        self.t_star_min, self.t_star_max = value[2]

    def get_bounds_curve_fit(self):
        """Get bounds for curve-fit()."""
        bnds = np.array(self.bounds, dtype=float).T
        if np.all(np.isnan(bnds)):
            return None
        bnds[0, np.isnan(bnds[0])] = -1e100
        bnds[1, np.isnan(bnds[1])] = 1e100
        return bnds


class GridSampling():
    """
    A class for sampling a parameter space over a grid.

    Sampling can be performed by several approaches.
    The class provides optimal solutions, uncertainties and plotting methods.
    """

    def __init__(self, misfit_func, bounds, nsteps, sampling_mode,
                 params_name, params_unit):
        """
        Init grid sampling.

        bounds : sequence of (min, max) pairs for each dimension.
        nsteps : number of grid steps for each dimension.
        sampling_mode : sequence of 'lin' or 'log' for each diemesion.
        params_name : sequence of parameter names (str).
        params_name : sequence of parameter units (str).
        """
        self.misfit_func = misfit_func
        self.bounds = bounds
        self.nsteps = nsteps
        self.sampling_mode = sampling_mode
        self.params_name = params_name
        self.params_unit = params_unit
        self.misfit = None
        self._conditional_misfit = None
        self._conditional_peak_widths = None
        self._values = None
        self._min_idx = None
        self.truebounds = []
        for bds, ns, mode in zip(self.bounds, self.nsteps, self.sampling_mode):
            if None in bds:
                msg = 'All parameters must be bounded for grid sampling'
                raise RuntimeError(msg)
            if mode == 'log':
                if bds[0] == 0:
                    bds = (bds[1]/ns, bds[1])
                bds = tuple(np.log10(bds))
            self.truebounds.append(bds)
        self.kdt = None

    @property
    def values(self):
        if self._values is not None:
            return self._values
        values = []
        for bds, ns, mode in zip(
                self.truebounds, self.nsteps, self.sampling_mode):
            if mode == 'log':
                values.append(np.logspace(*bds, ns))
            else:
                values.append(np.linspace(*bds, ns))
        self._values = np.meshgrid(*values, indexing='ij')
        return self._values

    @property
    def min_idx(self):
        if self.misfit is None:
            return None
        if self._min_idx is None:
            return np.unravel_index(
                np.nanargmin(self.misfit), self.misfit.shape)
        else:
            return self._min_idx

    @property
    def values_1d(self):
        """Extract a 1D array of parameter values along one dimension."""
        # same thing for values: we extract a 1d array of values along dim
        ndim = len(self.values)
        values_1d = []
        for dim in range(ndim):
            v = np.moveaxis(self.values[dim], dim, -1)
            values_1d.append(v[0, 0])
        return tuple(values_1d)

    @property
    def conditional_misfit(self):
        """
        Compute conditional misfit along each dimension.

        Conditional misfit is computed by fixing the other parameters to
        their optimal value.
        """
        if self.misfit is None:
            return None
        if self._conditional_misfit is not None:
            return self._conditional_misfit
        ndim = self.misfit.ndim
        cond_misfit = []
        for dim in range(ndim):
            # `dim` is the dimension to keep
            # we move `dim` to the last axis
            mm = np.moveaxis(self.misfit, dim, -1)
            # we fix ndim-1 coordinates of the minimum
            idx = tuple(v for n, v in enumerate(self.min_idx) if n != dim)
            # we extract from mm a 1-d profile along dim,
            # by fixing all the other dimensions (conditional misfit)
            mm = mm[idx]
            cond_misfit.append(mm)
        self._conditional_misfit = tuple(cond_misfit)
        return self._conditional_misfit

    @property
    def params_opt(self):
        if self.misfit is None:
            return None
        return np.array([v[self.min_idx] for v in self.values])

    @property
    def params_err(self):
        if self.misfit is None:
            return None
        error = []
        for p, w in zip(self.params_opt, self.conditional_peak_widths):
            err_left = p-w[1]
            err_right = w[2]-p
            error.append((err_left, err_right))
        return tuple(error)

    @property
    def conditional_peak_widths(self):
        """Find width of conditional misfit around its minimum."""
        if self.misfit is None:
            return None
        if self._conditional_peak_widths is not None:
            return self._conditional_peak_widths
        peak_widths = []
        rel_height = np.exp(-0.5)  # height of a gaussian for x=sigma
        for mm, idx, values in zip(
                self.conditional_misfit, self.min_idx, self.values_1d):
            width_height, idx_left, idx_right = GridSampling.peak_width(
                mm, idx, rel_height, negative=True)
            peak_widths.append(
                (width_height, values[idx_left], values[idx_right]))
        self._conditional_peak_widths = tuple(peak_widths)
        return self._conditional_peak_widths

    @staticmethod
    def peak_width(x, peak_idx, rel_height, negative=False):
        """
        Find width of a single peak at a given relative height.

        rel_height: float parameter between 0 and 1
                    0 means the base of the curve and 1 the peak value
                    (Note: this is the opposite of scipy.peak_widths)
        """
        if rel_height < 0 or rel_height > 1:
            msg = 'rel_height must be between 0 and 1'
            raise ValueError(msg)
        if negative:
            sign = -1
        else:
            sign = 1
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=PeakPropertyWarning)
            _, width_height, idx_left, idx_right = peak_widths(
                sign * x, [peak_idx, ], 1 - rel_height)
        idx_left = int(idx_left)
        idx_right = int(idx_right)
        width_height = sign * width_height[0]
        # fall back approach if the previous one fails
        if idx_left == idx_right:
            height = x.max() - x.min()
            if not negative:
                rel_height = 1 - rel_height
            width_height = x.max() - rel_height * height
            # Search for the indexes of the misfit curve points which are
            # closest to width_height, on the left and on the right
            #   Note: This assumes that the misfit function is monotonic.
            #         A safer but less precise approach is the commented one,
            #         based on "iii"
            # iii = np.where(np.isclose(x, width_height, rtol=0.1))
            try:
                # idx_left = np.min(iii[iii < peak_idx])
                x2 = x.copy()
                x2[peak_idx:] = np.inf
                idx_left = np.argmin(np.abs(x2 - width_height))
            except ValueError:
                idx_left = 0
            try:
                # idx_right = np.max(iii[iii > peak_idx])
                x2 = x.copy()
                x2[:peak_idx] = np.inf
                idx_right = np.argmin(np.abs(x2 - width_height))
            except ValueError:
                idx_right = -1
        return width_height, idx_left, idx_right

    def grid_search(self):
        """Sample the misfit function by simple grid search."""
        # small helper function to transform args into a tuple
        def mf(*args):
            return self.misfit_func(args)
        mf = np.vectorize(mf)
        self.misfit = mf(*self.values)

    def kdtree_search(self):
        # small helper function to transform misfit to pdf and manage logscale
        def mf(args):
            newargs = []
            for a, mode in zip(args, self.sampling_mode):
                if mode == 'log':
                    a = 10**a
                newargs.append(a)
            return np.exp(-self.misfit_func(newargs))
        extent = sum(self.truebounds, ())
        maxdiv = (20, 2000, 200)
        kdt = KDTree(extent, 2, mf, maxdiv=maxdiv)
        while kdt.ncells <= np.prod(maxdiv):
            oldn = kdt.ncells
            kdt.divide()
            if kdt.ncells == oldn:
                break
        deltas = []
        for bds, ns in zip(self.truebounds, self.nsteps):
            deltas.append((bds[1] - bds[0])/ns)
        pdf, extent = kdt.get_pdf(deltas)
        self.kdt = kdt
        self.misfit = -np.log(pdf)
        self.nsteps = self.misfit.shape
        self.extent = extent


class KDTree():
    def __init__(self, extent, init_parts, calc_pdf, min_cell_prob=0.,
                 maxdiv=None):
        # extent defines the size of search hypervolume
        # reshape extent to (dim, 2), where dim is the
        # arbitrary dimension of the parameter space
        self.extent = np.array(extent).reshape(-1, 2)
        # create the first cell, with the same size
        # of the search hypervolume
        cell0 = KDTCell(self.extent, calc_pdf, maxdiv=maxdiv)
        self.init_prob = cell0.prob
        cell0.min_cell_prob = cell0.prob*min_cell_prob
        self.cells = cell0.divide(init_parts)
        self.ncells = len(self.cells)

    def divide(self):
        # find the cell with highest probability and
        # divide it in 2 parts along every dimension
        # self.cells.sort(key=lambda c: c.prob)
        self.cells.sort(key=lambda c: c.prob_divisible)
        cell0 = self.cells.pop()
        # print(self.init_prob, cell0.prob, cell0.prob/self.init_prob)
        self.cells += cell0.divide(2)
        self.ncells = len(self.cells)

    def get_pdf(self, deltas):
        deltas = np.array(deltas).reshape(-1, 1)
        extent = self.extent.copy()
        ranges = []
        extent_new = []
        for v in np.hstack((extent, deltas)):
            start, stop, step = v
            # add a small number to make sure end value is included
            stop += stop/1e5
            rng = np.arange(start, stop, step)
            ranges.append(rng)
            extent_new += [rng[0], rng[-1]]
        xi = np.meshgrid(*ranges, indexing='ij')
        coords = np.array([cell.coords for cell in self.cells])
        pdf_values = np.array([cell.pdf for cell in self.cells])
        pdf = griddata(coords, pdf_values, tuple(xi), method='linear')
        return pdf, extent_new

class KDTCell():
    def __init__(self, extent, calc_pdf, min_cell_prob=0,
                 ndiv=None, maxdiv=None):
        self.extent = extent
        self.coords = np.mean(extent, axis=1)
        self.delta = np.diff(extent)
        self.calc_pdf = calc_pdf
        self.pdf = calc_pdf(self.coords)
        self.volume = np.prod(self.delta)
        self.prob = self.pdf * self.volume
        self.min_cell_prob = min_cell_prob
        if self.prob <= min_cell_prob:
            self.is_divisible = False
            self.prob_divisible = 0.
        else:
            self.is_divisible = True
            self.prob_divisible = self.prob
        self.ndiv = ndiv
        if self.ndiv is None:
            self.ndiv = np.zeros(self.extent.shape[0])
        self.maxdiv = maxdiv

    def divide(self, parts):
        if not self.is_divisible:
            return [self, ]
        # dim is the dimension of the parameter space
        dim = self.extent.shape[0]
        # define a set of increments for all the spatial directions
        # the number of increments is "parts"
        # The code "np.mgrid...[1]" produces a 2D array of the type:
        #  [[0 1 ... parts-1]
        #   ... (dim) ...
        #   [0 1 ... parts-1]]
        # increments = np.mgrid[0:dim, 0:parts][1] * self.delta/(parts-1)
        increments = np.mgrid[0:dim, 0:parts][1]
        if self.maxdiv is not None:
            # dimensions that will not be divided
            increments[self.ndiv >= self.maxdiv] *= 0
            # dimensions that will be divided
            self.ndiv[self.ndiv < self.maxdiv] += parts
        increments = increments * self.delta/(parts-1)
        # take the minimum coordinate and transform it to a column vector
        mincoord = self.extent[:, 0].reshape(-1, 1)
        # add minimum coordinates to the increments
        coords = mincoord + increments
        cells = []
        # loop over all the possible n-uplets of coordinates
        # we use set() to avoid repetitions for dimensions that
        # will not be dvided
        for c in set(itertools.product(*coords)):
            # c is a coordinate n-uplet. Let's transform it to column vector
            c = np.array(c).reshape(-1, 1)
            delta = self.delta/parts
            # FIXME: I forgot my own code!
            #  should this `2` be `parts`? or is it related to delta/2?
            inc = np.mgrid[0:dim, 0:2][1] * delta - delta/2
            extent = c + inc
            cells.append(
                KDTCell(
                    extent, self.calc_pdf, self.min_cell_prob,
                    self.ndiv, self.maxdiv))
        return cells

class moment_tools:

    @classmethod
    def moment_to_mag(moment):
        """Convert moment to magnitude."""
        return (np.log10(moment) - 9.1) / 1.5

    @classmethod
    def mag_to_moment(cls, magnitude):
        """Convert magnitude to moment."""
        return np.power(10, (1.5 * magnitude + 9.1))

    @classmethod
    def source_radius(cls, fc_in_hz, vs_in_m_per_s):
        """
        Compute source radius in meters.

        Madariaga (2009), doi:10.1007/978-1-4419-7695-6_22, eq. 31
        """
        return 0.3724 * vs_in_m_per_s / fc_in_hz

    @classmethod
    def bsd(cls, Mo_in_N_m, ra_in_m):
        """
        Compute Brune stress drop in MPa.

        Madariaga (2009), doi:10.1007/978-1-4419-7695-6_22, eq. 27
        """
        return 7. / 16 * Mo_in_N_m / ra_in_m ** 3 * 1e-6

    @classmethod
    def quality_factor(cls, travel_time_in_s, t_star_in_s):
        """Compute quality factor from travel time and t_star."""
        if t_star_in_s == 0:
            return np.inf
        qf = float(travel_time_in_s) / t_star_in_s
        return qf