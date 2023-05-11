from collections import OrderedDict
import logging
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from collections import OrderedDict
logger = logging.getLogger(__name__.split('.')[-1])


sspec_out_comments = {
    'begin': 'SourceSpec output in YAML format',
    'run_info': 'Information on the SourceSpec run',
    'event_info': 'Information on the event',
    'summary_spectral_parameters':
        'Summary spectral parameters, computed using different statistics',
    'station_parameters':
        'Parameters describing each station and spectral measurements\n'
        'performed at that station'
}

class OrderedAttribDict(OrderedDict):
    """
    An ordered dictionary whose values can be accessed as classattributes.
    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

class SummarySpectralParameter(OrderedAttribDict):
    """
    A summary spectral parameter comprising one ore more summary statistics.
    """

    def __init__(self, id, name=None, units=None, format=None):
        self._id = id
        self.name = name
        self.units = units
        # number formatting string
        self._format = format

    def __setattr__(self, attr, value):
        if isinstance(value, SummaryStatistics):
            value._format = self._format
        self[attr] = value

class OrderedAttribDict(OrderedDict):
    """
    An ordered dictionary whose values can be accessed as classattributes.
    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

class SpectralParameter(OrderedAttribDict):
    """A spectral parameter measured at one station."""

    def __init__(self, id, name=None, units=None, value=None, uncertainty=None,
                 lower_uncertainty=None, upper_uncertainty=None,
                 confidence_level=None, format=None):
        self._id = id
        self._format = format
        self.name = name
        self.units = units
        self.value = value
        self.uncertainty = uncertainty
        if (lower_uncertainty is not None and
                lower_uncertainty == upper_uncertainty):
            self.uncertainty = lower_uncertainty
            self.lower_uncertainty = self.upper_uncertainty = None
        else:
            self.lower_uncertainty = lower_uncertainty
            self.upper_uncertainty = upper_uncertainty
        self.confidence_level = confidence_level
        self.outlier = False

    def value_uncertainty(self):
        """Return value and uncertainty as 3-element tuple."""
        if self.lower_uncertainty is not None:
            uncertainty = (self.lower_uncertainty, self.upper_uncertainty)
        else:
            uncertainty = (self.uncertainty, self.uncertainty)
        return (self.value, *uncertainty)

class StationParameters(OrderedAttribDict):
    """
    The parameters describing a given station (e.g., its id and location) and
    the spectral parameters measured at that station.

    Spectral parameters are provided as attributes, using SpectralParameter()
    objects.
    """

    def __init__(self, id, instrument_type=None, latitude=None, longitude=None,
                 hypo_dist_in_km=None, epi_dist_in_km=None, azimuth=None):
        self._id = id
        self.instrument_type = instrument_type
        self.latitude = latitude
        self.longitude = longitude
        self.hypo_dist_in_km = hypo_dist_in_km
        self.epi_dist_in_km = epi_dist_in_km
        self.azimuth = azimuth
        self._params = dict()
        self._params_err = dict()
        self._is_outlier = dict()

    def __setattr__(self, attr, value):
        if isinstance(value, SpectralParameter):
            parname = attr
            par = value
            self._params[parname] = par.value
            if par.uncertainty is not None:
                self._params_err[parname] = (par.uncertainty, par.uncertainty)
            else:
                self._params_err[parname] = (
                    par.lower_uncertainty, par.upper_uncertainty
                )
            self._is_outlier[parname] = par.outlier
        self[attr] = value

    def rebuild_dictionaries(self):
        for key, value in self.items():
            if not isinstance(value, SpectralParameter):
                continue
            parname = key
            par = value
            self._params[parname] = par.value
            if par.uncertainty is not None:
                self._params_err[parname] = (par.uncertainty, par.uncertainty)
            elif par.lower_uncertainty is not None:
                self._params_err[parname] = (
                    par.lower_uncertainty, par.upper_uncertainty
                )
            else:
                self._params_err[parname] = (np.nan, np.nan)
            self._is_outlier[parname] = par.outlier

class SummaryStatistics(OrderedAttribDict):
    """
    A summary statistics (e.g., mean, weighted_mean, percentile), along with
    its uncertainty.
    """

    def __init__(self, type, value=None, uncertainty=None,
                 lower_uncertainty=None, upper_uncertainty=None,
                 confidence_level=None, lower_percentage=None,
                 mid_percentage=None, upper_percentage=None,
                 nobs=None, message=None,
                 format=None):
        # type of statistics: e.g., mean, median
        self._type = type
        self.value = value
        self.uncertainty = uncertainty
        if (lower_uncertainty is not None and
                lower_uncertainty == upper_uncertainty):
            self.uncertainty = lower_uncertainty
            self.lower_uncertainty = self.upper_uncertainty = None
        else:
            self.lower_uncertainty = lower_uncertainty
            self.upper_uncertainty = upper_uncertainty
        self.confidence_level = confidence_level
        self.lower_percentage = lower_percentage
        self.mid_percentage = mid_percentage
        self.upper_percentage = upper_percentage
        self.nobs = nobs
        self.message = message
        self._format = format

    def compact_uncertainty(self):
        """Return uncertainty in a compact form."""
        if self.lower_uncertainty is not None:
            return (self.lower_uncertainty, self.upper_uncertainty)
        else:
            return (self.uncertainty, self.uncertainty)


class SummarySpectralParameter(OrderedAttribDict):
    """
    A summary spectral parameter comprising one ore more summary statistics.
    """

    def __init__(self, id, name=None, units=None, format=None):
        self._id = id
        self.name = name
        self.units = units
        # number formatting string
        self._format = format

    def __setattr__(self, attr, value):
        if isinstance(value, SummaryStatistics):
            value._format = self._format
        self[attr] = value

class SourceSpecOutput(OrderedAttribDict):
    """The output of SourceSpec."""

    def __init__(self):
        self.run_info = OrderedAttribDict()
        self.event_info = OrderedAttribDict()
        self.summary_spectral_parameters = OrderedAttribDict()
        self.station_parameters = OrderedAttribDict()

    def value_energy_array(self, key, filter_outliers=False):
        vals = np.array([x.Er.value for x in self.station_parameters.values()])
        if filter_outliers:
            outliers = self.outlier_array(key)
            vals = vals[~outliers]
        return vals
    def value_array(self, key, filter_outliers=False):
        vals = np.array([x._params.get(key, np.nan) for x in self.station_parameters.values()])
        if filter_outliers:
            outliers = self.outlier_array(key)
            vals = vals[~outliers]
        return vals

    def error_array(self, key, filter_outliers=False):
        errs = np.array([
            x._params_err.get(key, np.nan)
            for x in self.station_parameters.values()
        ])
        if filter_outliers:
            outliers = self.outlier_array(key)
            errs = errs[~outliers]
        else:
            errs[:] = np.nan
            errs = errs.astype(float)
        return errs

    def outlier_array(self, key):
        outliers = np.array([
            # if we cannot find the given key, we assume outlier=True
            x._is_outlier.get(key, True)
            for x in self.station_parameters.values()
        ])
        return outliers

    def find_outliers(self, key, n):
        """
        Find outliers using the IQR method.

        .. code-block::

                Q1-n*IQR   Q1   median  Q3    Q3+n*IQR
                            |-----:-----|
            o      |--------|     :     |--------|    o  o
                            |-----:-----|
            outlier         <----------->            outliers
                                 IQR

        If ``n`` is ``None``, then the above check is skipped.
        ``Nan`` and ``inf`` values are also marked as outliers.
        """
        values = self.value_array(key)
        station_ids = np.array([id for id in self.station_parameters.keys()])
        naninf = np.logical_or(np.isnan(values), np.isinf(values))
        _values = values[~naninf]
        if n is not None and len(_values) > 0:
            Q1, _, Q3 = np.percentile(_values, [25, 50, 75])
            IQR = Q3-Q1
            outliers = np.logical_or(values < Q1 - n*IQR, values > Q3 + n*IQR)
            outliers = np.logical_or(outliers, naninf)
        else:
            outliers = naninf
        for stat_id, outl in zip(station_ids, outliers):
            stat_par = self.station_parameters[stat_id]
            stat_par[key].outlier = outl
            stat_par.rebuild_dictionaries()

    def mean_values(self):
        """Return a dictionary of mean values."""
        return {
            parname: par.mean.value
            for parname, par in self.summary_spectral_parameters.items()
            if isinstance(par, SummarySpectralParameter)
            and 'mean' in par
        }

    def mean_uncertainties(self):
        """Return a dictionary of mean uncertainties."""
        return {
            parname: par.mean.compact_uncertainty()
            for parname, par in self.summary_spectral_parameters.items()
            if isinstance(par, SummarySpectralParameter)
            and 'mean' in par
        }

    def weighted_mean_values(self):
        """Return a dictionary of weighted mean values."""
        return {
            parname: par.weighted_mean.value
            for parname, par in self.summary_spectral_parameters.items()
            if isinstance(par, SummarySpectralParameter)
            and 'weighted_mean' in par
        }

    def weighted_mean_uncertainties(self):
        """Return a dictionary of weighted mean uncertainties."""
        return {
            parname: par.weighted_mean.compact_uncertainty()
            for parname, par in self.summary_spectral_parameters.items()
            if isinstance(par, SummarySpectralParameter)
            and 'weighted_mean' in par
        }

    def percentiles_values(self):
        """Return a dictionary of percentile values."""
        return {
            parname: par.percentiles.value
            for parname, par in self.summary_spectral_parameters.items()
            if isinstance(par, SummarySpectralParameter)
            and 'percentiles' in par
        }

    def percentiles_uncertainties(self):
        """Return a dictionary of percentile uncertainties."""
        return {
            parname: par.percentiles.compact_uncertainty()
            for parname, par in self.summary_spectral_parameters.items()
            if isinstance(par, SummarySpectralParameter)
            and 'percentiles' in par
        }

    def reference_values(self):
        """Return a dictionary of reference values."""
        try:
            ref_stat = self.summary_spectral_parameters.reference_statistics
        except KeyError:
            raise ValueError('No reference statistics defined')
        if ref_stat == 'mean':
            return self.mean_values()
        elif ref_stat == 'weighted_mean':
            return self.weighted_mean_values()
        elif ref_stat == 'percentiles':
            return self.percentiles_values()
        else:
            msg = 'Invalid reference statistics: {}'.format(ref_stat)
            raise ValueError(msg)

    def reference_uncertainties(self):
        """Return a dictionary of reference uncertainties."""
        try:
            ref_stat = self.summary_spectral_parameters.reference_statistics
        except KeyError:
            raise ValueError('No reference statistics defined')
        if ref_stat == 'mean':
            return self.mean_uncertainties()
        elif ref_stat == 'weighted_mean':
            return self.weighted_mean_uncertainties()
        elif ref_stat == 'percentiles':
            return self.percentiles_uncertainties()
        else:
            msg = 'Invalid reference statistics: {}'.format(ref_stat)
            raise ValueError(msg)

    def reference_summary_parameters(self):
        """
        Return a dictionary of reference summary parameters,
        each being a SummaryStatistics() object.
        """
        try:
            ref_stat = self.summary_spectral_parameters.reference_statistics
        except KeyError:
            raise ValueError('No reference statistics defined')
        return {
            parname: par[ref_stat]
            for parname, par in self.summary_spectral_parameters.items()
            if isinstance(par, SummarySpectralParameter)
            and ref_stat in par
        }

def compute_summary_statistics(config, sspec_output):
    """Compute summary statistics from station spectral parameters."""
    #if len(sspec_output.station_parameters) == 0:
    #    logger.info('No source parameter calculated')


    sspec_output.summary_spectral_parameters.reference_statistics = \
        config["reference_statistics"]

    # Mw
    sspec_output.summary_spectral_parameters.Mw = \
        _param_summary_statistics(
            config, sspec_output,
            id='Mw', name='moment magnitude', format='{:.2f}',
            logarithmic=False
        )

    # Mo (N.m)
    sspec_output.summary_spectral_parameters.Mo = \
        _param_summary_statistics(
            config, sspec_output,
            id='Mo', name='seismic moment', units='N.m', format='{:.3e}',
            logarithmic=True
        )

    # fc (Hz)
    sspec_output.summary_spectral_parameters.fc = \
        _param_summary_statistics(
            config, sspec_output,
            id='fc', name='corner frequency', units='Hz', format='{:.3f}',
            logarithmic=True
        )

    # t_star (s)
    sspec_output.summary_spectral_parameters.t_star = \
        _param_summary_statistics(
            config, sspec_output,
            id='t_star', name='t-star', units='s', format='{:.3f}',
            logarithmic=False
        )

    # radius (meters)
    sspec_output.summary_spectral_parameters.radius = \
        _param_summary_statistics(
            config, sspec_output,
            id='radius', name='source radius', units='m', format='{:.3f}',
            logarithmic=True
        )

    # bsd, Brune stress drop (MPa)
    sspec_output.summary_spectral_parameters.bsd = \
        _param_summary_statistics(
            config, sspec_output,
            id='bsd', name='Brune stress drop', units='MPa', format='{:.3e}',
            logarithmic=True
        )

    # Quality factor
    sspec_output.summary_spectral_parameters.Qo = \
        _param_summary_statistics(
            config, sspec_output,
            id='Qo', name='quality factor', format='{:.1f}',
            logarithmic=False
        )

    # Er (N.m)
    sspec_output.summary_spectral_parameters.Er = \
         _param_summary_statistics(
             config, sspec_output,
             id='Er', name='radiated energy', units='N.m', format='{:.3e}',
             logarithmic=True, filter_outliers=False)
    #
    # # Ml
    # if config.compute_local_magnitude:
    #     sspec_output.summary_spectral_parameters.Ml = \
    #         _param_summary_statistics(
    #             config, sspec_output,
    #             id='Ml', name='local magnitude', format='{:.2f}',
    #             logarithmic=False
    #         )

    params_name = ('Mw', 'fc', 't_star')
    means = sspec_output.mean_values()
    sourcepar_mean = {par: means[par] for par in params_name}
    logger.info('params_mean: {}'.format(sourcepar_mean))
    means_weight = sspec_output.weighted_mean_values()
    sourcepar_mean_weight = {par: means_weight[par] for par in params_name}
    logger.info('params_mean_weighted: {}'.format(sourcepar_mean_weight))
    percentiles = sspec_output.percentiles_values()
    sourcepar_percentiles = {par: percentiles[par] for par in params_name}
    logger.info('params_percentiles: {}'.format(sourcepar_percentiles))

    return sspec_output

def _param_summary_statistics(
        config, sspec_output, id, name, format, units=None, logarithmic=False, filter_outliers=True):
    """Compute summary statistics for one spectral parameter."""
    nIQR = config["nIQR"]
    summary = SummarySpectralParameter(
        id=id, name=name, format=format, units=units)
    sspec_output.find_outliers(id, n=nIQR)
    values = sspec_output.value_array(id, filter_outliers=True)
    errors = sspec_output.error_array(id, filter_outliers=filter_outliers)
    # put to NaN infinite values and values whose errors are infinite
    values[np.isinf(values)] = np.nan
    try:
        _cond_err = np.logical_or(np.isinf(errors[:, 0]), np.isinf(errors[:, 1]))
        values[_cond_err] = np.nan
        errors[_cond_err] = np.nan
    except:
        pass

    # only count non-NaN values
    nobs = len(values[~np.isnan(values)])
    # mean
    mean_value, mean_error = _avg_and_std(values, logarithmic=logarithmic)
    mean_error *= config["n_sigma"]
    conf_level = _normal_confidence_level(config["n_sigma"])
    summary.mean = SummaryStatistics(
        type='mean', value=mean_value,
        lower_uncertainty=mean_error[0],
        upper_uncertainty=mean_error[1],
        confidence_level=conf_level, nobs=nobs)
    # weighted mean (only if errors are defined)
    if not np.all(np.isnan(errors)):
        wmean_value, wmean_error = _avg_and_std(
            values, errors, logarithmic=logarithmic)
        wmean_error *= config["n_sigma"]
        summary.weighted_mean = SummaryStatistics(
            type='weighted_mean', value=wmean_value,
            lower_uncertainty=wmean_error[0],
            upper_uncertainty=wmean_error[1],
            confidence_level=conf_level, nobs=nobs)
    # percentiles
    low_pctl, mid_pctl, up_pctl = _percentiles(
        values, config["lower_percentage"], config["mid_percentage"],
        config["upper_percentage"])
    conf_level = round(
        (config["upper_percentage"] - config["lower_percentage"]), 2)
    summary.percentiles = SummaryStatistics(
        type='percentiles', value=mid_pctl,
        lower_uncertainty=mid_pctl - low_pctl,
        upper_uncertainty=up_pctl - mid_pctl,
        confidence_level=conf_level,
        lower_percentage=config["lower_percentage"],
        mid_percentage=config["mid_percentage"],
        upper_percentage=config["upper_percentage"],
        nobs=nobs)
    return summary

def _avg_and_std(values, errors=None, logarithmic=False):
    """
    Return the average and standard deviation.

    Optionally:
    - errors can be specified for weighted statistics
    - logarithmic average and standard deviation
    """
    average = std = np.nan
    if len(values) == 0:
        return average, np.array((std, std))
    if errors is not None and len(errors) == 0:
        return average, np.array((std, std))
    if np.all(np.isnan(values)):
        return average, np.array((std, std))
    if errors is None:
        weights = None
    else:
        # negative errors should not happen
        errors[errors < 0] = 0
        values_minus = values - errors[:, 0]
        values_plus = values + errors[:, 1]
        if logarithmic:
            # compute the width of the error bar in log10 units
            # replace negative left values with 1/10 of the central value
            values_minus[values_minus <= 0] = values[values_minus <= 0]/10
            values_log_minus = np.log10(values_minus)
            values_log_plus = np.log10(values_plus)
            errors_width = values_log_plus - values_log_minus
        else:
            # compute the width of the error bar in linear units
            errors_width = values_plus - values_minus
        # fix for infinite weight (zero error width)
        errors_width[errors_width == 0] =\
            np.nanmin(errors_width[errors_width > 0])
        weights = 1./(errors_width**2.)
    if logarithmic:
        values = np.log10(values)
    notnan = ~np.isnan(values)
    if weights is not None:
        notnan = np.logical_and(notnan, ~np.isnan(weights))
        weights = weights[notnan]
    values = values[notnan]
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    std = np.sqrt(variance)
    if logarithmic:
        log_average = 10.**average
        minus = log_average - 10.**(average-std)
        plus = 10.**(average+std) - log_average
        return log_average, np.array((minus, plus))
    else:
        return average, np.array((std, std))

def _percentiles(
        values, low_percentage=25, mid_percentage=50, up_percentage=75):
    """Compute lower, mid and upper percentiles."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    low_percentile, mid_percentile, up_percentile =\
        np.nanpercentile(
            values, (low_percentage, mid_percentage, up_percentage))
    return low_percentile, mid_percentile, up_percentile

def _normal_confidence_level(n_sigma):
    """
    Compute the confidence level of a normal (Gaussian) distribution
    between -n_sigma and +n_sigma.
    """
    def gauss(x):
        return norm.pdf(x, 0, 1)
    confidence, _ = quad(gauss, -n_sigma, n_sigma)
    return np.round(confidence*100, 2)