# -*- coding: utf8 -*-
# SPDX-License-Identifier: CECILL-2.1
"""
Compute radiated energy from spectral integration.

:copyright:
    2012 Claudio Satriano <satriano@ipgp.fr>

    2013-2014 Claudio Satriano <satriano@ipgp.fr>,
              Emanuela Matrullo <matrullo@geologie.ens.fr>,
              Agnes Chounet <chounet@ipgp.fr>

    2015-2023 Claudio Satriano <satriano@ipgp.fr>
:license:
    CeCILL Free Software License Agreement v2.1
    (http://www.cecill.info/licences.en.html)
"""
import logging
import numpy as np

from isp.DataProcessing.automag_energy import SpectralParameter

logger = logging.getLogger(__name__.split('.')[-1])

class Energy:

    @classmethod
    def radiated_energy(cls, id, spec, specnoise, freq_signal, freq_noise, delta_signal, delta_noise,
                        fc, vel, f_max, rho, t_star, station_pars):

        """Compute radiated energy, using eq. (3) in Lancieri et al. (2012)."""

        # spec_signal & spec_noise in moment units
        # vel m/s
        logger.info('Computing radiated energy...')
        logger.warning('Warning: computing radiated energy from P waves might lead to '
                'an underestimation')

        # Compute signal and noise integrals and subtract noise from signal,
        # under the hypothesis that energy is additive and noise is stationary
        signal_integral = cls.spectral_integral(spec, t_star, freq_signal, f_max, delta_signal)
        noise_integral = cls.spectral_integral(specnoise, t_star, freq_noise, f_max, delta_noise)
        coeff = cls.radiated_energy_coefficient(rho, vel)
        k = delta_signal/delta_noise
        Er = coeff * (signal_integral - k*noise_integral)
        if Er <= 0:
            Er = coeff * (signal_integral)
            # msg = '{} {}: noise energy is larger than signal energy: '.format(
            #     id, spec.stats.instrtype)
            # msg += 'skipping spectrum.'
            # logger.warning(msg)


        R = cls.finite_bandwidth_correction(freq_signal, fc, f_max)
        Er /= R
        logger.info('Computing radiated energy: done')

        try:
            param_Er = station_pars.Er
        except KeyError:

            station_pars.Er = SpectralParameter(
                 id='Er', value=Er, format='{:.3f}')
            station_pars._params['Er'] = Er
            station_pars._params_err['Er'] = (None,None)
            station_pars._is_outlier['Er'] = False

        return station_pars

    @classmethod
    def spectral_integral(cls, spec, t_star, freq, f_max, period_full):
        """Compute spectral integral in eq. (3) from Lancieri et al. (2012)."""
        # Note: eq. (3) from Lancieri et al. (2012) is the same as
        # eq. (1) in Boatwright et al. (2002), but expressed in frequency,
        # instead of angular frequency (2pi factor).

        # Data must be in displacement units,
        # and derive it to velocity through multiplication by 2*pi*freq:
        # (2. is the free-surface amplification factor)
        data = spec * (2 * np.pi * freq)
        # Correct data for attenuation:
        data *= np.exp(np.pi * t_star * freq)
        # Compute the energy integral, up to f_max:
        if f_max is not None:
            data[freq > f_max] = 0.
        integral = np.sum((data ** 2) * period_full)
        return integral

    @classmethod
    def radiated_energy_coefficient(cls, rho, vel):
        """Compute coefficient in eq. (3) from Lancieri et al. (2012)."""
        # Note: eq. (3) from Lancieri et al. (2012) is the same as
        # eq. (1) in Boatwright et al. (2002), but expressed in frequency,
        # instead of angular frequency (2pi factor).
        # In the original eq. (3) from Lancieri et al. (2012), eq. (3),
        # the correction term is:
        #       8 * pi * r**2 * C**2 * rho * vs
        # We do not multiply by r**2, since data is already distance-corrected.
        # From Boatwright et al. (2002), eq. (2), C = <Fs>/(Fs * S),
        # where <Fs> is the average radiation pattern in the focal sphere
        # and Fs is the radiation pattern for the given angle between source
        # and receiver. Here we put <Fs>/Fs = 1, meaning that we rely on the
        # averaging between measurements at different stations, instead of
        # precise measurements at a single station.
        # S is the free-surface amplification factor, which we put = 2
        coeff = 8 * np.pi * (1. / 2.) ** 2 * rho * vel
        return coeff

    @classmethod
    def finite_bandwidth_correction(cls, freq, fc, f_max):
        """
        Compute finite bandwidth correction.

        Expressed as the ratio R between the estimated energy
        and the true energy (Di Bona & Rovelli 1988)
        """
        if f_max is None:
            f_max = freq[-1]
        R = 2. / np.pi * (np.arctan2(f_max, fc) - (f_max / fc) / (1 + (f_max / fc) ** 2.))
        return R





