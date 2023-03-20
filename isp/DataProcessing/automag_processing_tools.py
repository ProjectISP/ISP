import math

import numpy as np
from obspy.taup import TauPyModel
from scipy.interpolate import interp1d


class signal_preprocess_tools:

        def __init__(self, tr_noise, tr_signal, hypo_dist_in_km, geom_spread_model, geom_spread_n_exponent,
                                geom_spread_cutoff_distance, event_info, rho, spectral_smooth_width_decades):

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
            self.rho = 2500
            self.spectral_smooth_width_decades = spectral_smooth_width_decades

        def do_spectrum(self):
            """Compute the spectrum of an ObsPy Trace object."""
            signal = self.tr_signal.data
            noise = self.tr_noise.data
            delta = self.tr_signal.stats.delta
            amp_signal, freq_signal = self.__dofft(signal, delta)
            amp_noise, freq_noise = self.__dofft(noise, delta)
            # remove DC component (freq=0)
            amp_signal = abs(amp_signal)[1:]
            freq_signal = freq_signal[1:]
            amp_noise = abs(amp_noise)[1:]
            freq_noise = freq_noise[1:]
            geom_spread_signal = self.geometrical_spreading_coefficient(amp_signal, freq_signal)
            geom_spread_noise = self.geometrical_spreading_coefficient(amp_signal, freq_noise)
            amp_signal *= geom_spread_signal
            amp_noise *= geom_spread_noise
            coeff_signal = self.__displacement_to_moment()
            coeff_noise = self.__displacement_to_moment()
            amp_signal *= coeff_signal
            amp_noise *= coeff_noise
            amp_signal_log, freq_signal_log = self._smooth_spectrum(amp_signal, freq_signal, self.spectral_smooth_width_decades)
            amp_noise_log, freq_noise_log = self._smooth_spectrum(amp_noise, freq_noise, self.spectral_smooth_width_decades)
            return amp_signal, freq_signal, amp_noise, freq_noise, amp_signal_log, freq_signal_log, amp_noise_log, freq_noise_log

        def geometrical_spreading_coefficient(self, amp_signal, freq_signal):

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

        def __dofft(self, signal, delta):
            """Compute the complex Fourier transform of a signal."""
            npts = len(signal)
            # if npts is even, we make it odd
            # so that we do not have a negative frequency in the last point
            # (see numpy.fft.rfft doc)
            if not npts % 2:
                npts -= 1

            fft = np.fft.rfft(signal, n=npts) * delta
            fftfreq = np.fft.fftfreq(len(signal), d=delta)
            fftfreq = fftfreq[0:fft.size]
            return fft, fftfreq

        def __displacement_to_moment(self):
            """
            Return the coefficient for converting displacement to seismic moment.

            From Aki&Richards,1980
            """

            v_hypo = self.__get_vel_from_taup(depth_event_km=self.event_info[3])
            v_station = self.__get_vel_from_taup(depth_event_km=0.0)

            v_hypo *= 1000.
            v_station *= 1000.
            v3 = v_hypo ** (5. / 2) * v_station ** (1. / 2)
            rp, rho = self.__get_radiation_pattern_coefficient()
            coeff = 4 * math.pi * v3 * self.rho / (2 * rp)

            return coeff

        def _smooth_spectrum(self, amplitude, freq, smooth_width_decades=0.2):
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


        def __get_vel_from_taup(self, **kwargs):
            depth_event_km = kwargs.pop('depth_event_km', 0.0)
            if depth_event_km < 0:
                depth_event_km = 1e-3
            return self.v_model.evaluate_above(depth_event_km, "P")[0]

        def __get_radiation_pattern_coefficient(self):

            # TODO FROM FOCAL MECHANISM

            #rp = abs(radiation_pattern(strike, dip, rake, takeoff_angle, azimuth))
            # we are interested only in amplitude
            # (P, SV and SH radiation patterns have a sign)


            # P-wave average radiation pattern coefficient:
            rpp = 0.52
            # S-wave average radiation pattern coefficient:
            rps = 0.62

            return rpp

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