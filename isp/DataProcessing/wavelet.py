import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import scipy
from deprecated import deprecated
from obspy import read, Trace, UTCDateTime
from obspy.signal.filter import lowpass
from scipy.signal import argrelextrema

from isp.Exceptions import InvalidFile
from isp.Structures.structures import TracerStats
from isp.Utils import ObspyUtil, MseedUtil


class ConvolveWaveletBase:
    """
    Class to apply wavelet convolution to a mseed file. The bank of atoms is computed at the class initialisation.
        Examples
        --------
        cw = ConvolveWavelet(file_path)
        convolve = cw.ccwt_ba_fast()
    """

    def __init__(self, data, **kwargs):
        """
        Class to apply wavelet convolution to a mseed file.
        The bank of atoms is computed at the class initialisation.
        :param data: Either the mseed file path or an obspy Tracer.
        :keyword kwargs:
        :keyword wmin: Minimum number of cycles. Default = 6.
        :keyword wmax: Maximum number of cycles. Default = 6.
        :keyword tt: Period of the Morlet Wavelet. Default = 2.
        :keyword fmin: Minimum Central frequency (in Hz). Default = 2.
        :keyword fmax: Maximum Central frequency (in Hz). Default = 12.
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax. Default = 20.
        :keyword use_rfft: True if it should use rfft instead of fft. Default = True.
        :keyword decimate: True if it should try to decimate the trace. Default = False. The decimation
            factor is equal to q = 0.4*SR/fmax. For SR=200Hz and fmax=40Hz, q=2. This will downsize the
            sample rate to 100 Hz.
        :raise InvalidFile: If file is not a valid mseed.
        :example:
        >>> cw = ConvolveWavelet(data)
        >>> cw.setup_wavelet()
        >>> sc = cw.scalogram_in_dbs
        >>> cf = cw.cf_lowpass()
        """

        if isinstance(data, Trace):
            self.stats = TracerStats.from_dict(data.stats)
            self.trace: Trace = data
        else:
            if not MseedUtil.is_valid_mseed(data):
                raise InvalidFile("The file: {} is not a valid mseed.".format(data))
            self.trace: Trace = read(data)[0]
            self.stats = ObspyUtil.get_stats(data)


        self._wmin = float(kwargs.get("wmin", 6.))
        self._wmax = float(kwargs.get("wmax", 6.))
        self._tt = float(kwargs.get("tt", 2.))
        self._fmin = float(kwargs.get("fmin", 2.))
        self._fmax = float(kwargs.get("fmax", 12.))
        self._nf = int(kwargs.get("nf", 20))
        self._use_rfft = kwargs.get("use_rfft", True)
        self._decimate = kwargs.get("decimate", False)

        self._validate_kwargs()
        # print(self.stats)

        self._data = None
        self._npts = 0
        self._tf = None
        self._start_time = self.stats.StartTime
        self._end_time = self.stats.EndTime
        self._sample_rate = self.stats.Sampling_rate

        self._frex = None
        self._n_cycles = None
        self._wtime = None
        self._half_wave = None

    def __repr__(self):
        return "ConvolveWavelet(data={}, wmin={}, wmax={}, tt={}, fmin={}, fmax={}, nf={})".format(
            self.trace, self._wmin, self._wmax, self._tt, self._fmin, self._fmax, self._nf)

    def __eq__(self, other):
        # noinspection PyProtectedMember
        return self.trace == other.trace and self._wmin == other._wmin and self._wmax == other._wmax \
               and self._tt == other._tt and self._fmin == other._fmin and self._fmax == other._fmax \
               and self._nf == other._nf and self._use_rfft == other._use_rfft \
               and self._start_time == other._start_time and self._end_time == other._end_time \
               and self._decimate == other._decimate

    def _validate_data(self):
        if self._data is None:
            raise AttributeError("Data not found. Run setup_wavelet().")

    def _validate_kwargs(self):
        if self._wmax < self._wmin:
            AttributeError("The kwarg wmin can't be bigger than wmax. wmin = {}, wmax = {}".
                           format(self._wmin, self._wmax))

        if self._fmax < self._fmin:
            AttributeError("The kwarg fmin can't be bigger than fmax. fmin = {}, fmax = {}".
                           format(self._fmin, self._fmax))

    @property
    def npts(self):
        return self._npts

    def filter_win(self, freq, index):
        # Create the Morlet wavelet and get its fft
        s = self._n_cycles[index] / (2 * np.pi * freq)
        # Normalize Factor
        normalization = 1 / (np.pi * s ** 2) ** 0.25
        # Complex sine = np.multiply(1j*2*(np.pi)*frex[fi],wtime))
        # Gaussian = np.exp(-1*np.divide(np.power(wtime,2),2*s**2))
        cmw = np.multiply(np.exp(np.multiply(1j * 2 * np.pi * freq, self._wtime)),
                          np.exp(-1 * np.divide(np.power(self._wtime, 2), 2 * s ** 2)))
        cmw = cmw.conjugate()
        # Normalizing. The square root term causes the wavelet to be normalized to have an energy of 1.
        cmw = normalization * cmw

        if self._use_rfft:
            cmw = np.real(cmw)

        return cmw

    def setup_wavelet(self, start_time: UTCDateTime = None, end_time: UTCDateTime = None, **kwargs):
        """
        Recompute the bank of atoms based on the new kwargs and the waveform data range. If start_time or end_time
        is not given then it will read the whole data from the mseed file.
        :param start_time: The start time of the waveform data. If not given default is the start time from the
            mseed header.
        :param end_time: The end time of the waveform data. If not given default is the end time from the
            mseed header.
        :keyword  kwargs:
        :keyword wmin: Minimum number of cycles.
        :keyword wmax: Maximum number of cycles.
        :keyword tt: Central frequency of the Morlet Wavelet.
        :keyword fmin: Minimum frequency (in Hz).
        :keyword fmax: Maximum frequency (in Hz).
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax.
        :return:
        """

        self._start_time = start_time if start_time else self.stats.StartTime
        self._end_time = end_time if end_time else self.stats.EndTime

        self.__setup_wavelet(start_time, end_time, **kwargs)

    def setup_atoms(self, **kwargs):
        """
        Recompute the bank of atoms based on the new kwargs. This method will only recompute the atoms. Use
        :class:`setup_wavelet()` if you want to change the data.
        :keyword  kwargs:
        :keyword wmin: Minimum number of cycles.
        :keyword wmax: Maximum number of cycles.
        :keyword tt: Central frequency of the Morlet Wavelet.
        :keyword fmin: Minimum frequency (in Hz).
        :keyword fmax: Maximum frequency (in Hz).
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax.
        :return:
        """
        self._wmin = float(kwargs.get("wmin", self._wmin))
        self._wmax = float(kwargs.get("wmax", self._wmax))
        self._tt = float(kwargs.get("tt", self._tt))
        self._fmin = float(kwargs.get("fmin", self._fmin))
        self._fmax = float(kwargs.get("fmax", self._fmax))
        self._nf = int(kwargs.get("nf", self._nf))

        self._validate_kwargs()
        self._tf = None  # Makes tf none to force to recompute tf when calling other methods.
        self._setup_atoms()

    def _setup_atoms(self):
        self._validate_data()

        self._frex = np.logspace(np.log10(self._fmin), np.log10(self._fmax), self._nf, base=10)
        self._n_cycles = np.linspace(self._wmin, self._wmax, self._nf)
        dt = 1 / self._sample_rate
        self._wtime = np.arange(-self._tt, self._tt + dt, dt)  # Kernel of the Mother Morlet Wavelet
        self._half_wave = (len(self._wtime) - 1) / 2

    def __get_data_in_time(self, start_time, end_time):
        tr = self.trace.copy()
        tr.trim(starttime=start_time, endtime=end_time)
        if self._decimate:
            tr = self.decimate_data(tr)
        tr.detrend(type='demean')
        tr.taper(max_percentage=0.05)
        self._npts = tr.stats.npts
        self._sample_rate = tr.stats.sampling_rate
        return tr.data

    def __get_resample_factor(self):
        rf = int(0.4 * self._sample_rate / self._fmax)
        return rf

    def decimate_data(self, tr: Trace):
        rf = self.__get_resample_factor()
        if rf > 1:
            data = scipy.signal.decimate(tr.data, rf, ftype='fir', zero_phase=True)
            new_stats = tr.stats
            new_stats["npts"] = len(data)
            new_stats["sampling_rate"] /= rf
            new_stats["delta"] = 1. / new_stats["sampling_rate"]
            return Trace(data, new_stats)

        return tr

    def get_nproc(self):
        total_cpu = multiprocessing.cpu_count()
        nproc = total_cpu - 2 if total_cpu > 3 else total_cpu - 1  # avoid to get all available cores.
        nproc = min(nproc, self._nf)
        nproc = max(nproc, 1)
        return nproc

    @staticmethod
    def __tapper(data, max_percentage=0.05):
        tr = Trace(data)
        tr.taper(max_percentage=max_percentage, type='blackman')
        return tr.data

    # def __chop_data(self, delta_time, start_time: UTCDateTime, end_time: UTCDateTime):
    #     total_time = (end_time - start_time) / 3600.
    #     n = np.math.ceil(total_time / delta_time)
    #
    #     data_set = []
    #     for h in range(n):
    #         dt = h * 3600 * delta_time
    #         dt2 = (h + 1) * 3600 * delta_time
    #         data = self.__get_data_in_time(start_time + dt, start_time + dt2)
    #         if data is not None:
    #             self._npts = int(self.stats.Sampling_rate * delta_time * 3600) + 1
    #             data = self.__pad_data(data, self._npts)
    #             data_set.append(data)
    #
    #     return data_set

    def __setup_wavelet(self, start_time: UTCDateTime, end_time: UTCDateTime, **kwargs):
        self._data = self.__get_data_in_time(start_time, end_time)
        self.setup_atoms(**kwargs)

    def _convolve_atoms(self, parallel: bool):
        # implement at the child.
        pass

    def scalogram_in_dbs(self):
        if self._tf is None:
            self.compute_tf()

        sc = np.abs(self._tf) ** 2
        return 10. * (np.log10(sc / np.max(sc)))

    def get_data_window(self):
        start = int(self._half_wave + 1)
        end = self._npts + int(self._half_wave + 1)
        return start, end

    # def __ccwt_ba_multitread(self):
    #     nproc = self.get_nproc()
    #     nproc = min(nproc, len(self._data))
    #
    #     with ThreadPool(nproc) as pool:
    #         ro = pool.map(self.__cwt_ba, self._data)
    #
    #     cwt = np.array([]).reshape(self._nf, 0)
    #     for index, r in enumerate(ro):
    #         cwt = np.concatenate((cwt, r), axis=1)
    #
    #     return cwt

    def compute_tf(self, parallel=True):
        pass

    def cf(self, tapper=True, parallel=True):
        """
        Characteristic function.
        Compute the mean values of the log10 differences of the convolved waveform with the wavelet from fmin to fmax.
        :param tapper: True for tapper the result. Default=True.
        :param parallel: Either or not it should run cwt in parallel. Default=True.
        :return: Mean values of the log10 difference of the convolved waveform with the wavelet from fmin to fmax.
        """

        if self._tf is None:
            self.compute_tf(parallel=parallel)

        cf = np.mean(np.diff(np.log10(np.abs(self._tf) ** 2)), axis=0, dtype=np.float32)

        if tapper:
            cf = self.__tapper(cf)

        return cf

    def cf_lowpass(self, tapper=True, parallel=True, freq=0.15):
        """
        Characteristic function with lowpass.
        Compute the mean values of the log10 differences of the convolved waveform with the wavelet from fmin to fmax
        with a low pass filter.
        :param tapper: True for tapper the result. Default=True.
        :param parallel: Either or not it should run cwt in parallel. Default=True.
        :param freq: Filter corner frequency. Default=0.15.
        :return: The filtered (lowpass, fmin=0.15) mean values of the log10 difference of the convolved waveform with
            the wavelet from fmin to fmax.
        """

        cf = lowpass(self.cf(tapper, parallel=parallel), freq, df=self._sample_rate, corners=3, zerophase=True)

        return cf

    def get_time_delay(self):
        """
        Compute the time delay in seconds of the wavelet.
        :return: The time delay of the wavelet in seconds.
        """
        return 0.5 * self._wmin / (2. * np.pi * self._fmin)

    def detect_max_pick_in_time(self, data: np.ndarray):
        """
        Get the time of the maximum pick.
        :param data: The data from ccwt_ba_fast method.
        :return: Return the obspy.UTCDateTime at the maximum pick if detected. If there is no pick
            above the detection limit it returns None.
        """
        filtered_data = np.abs(np.where(np.abs(data) >= self.get_detection_limit(data), data, 0.))
        if filtered_data.sum() != 0.:
            max_index = np.argmax(filtered_data)
            time_s = max_index / self._sample_rate
            return self._start_time + time_s + self.get_time_delay()
        return None

    def detect_picks_in_time(self, data: np.ndarray, sigmas=5.):
        """
        Get the times of the local maximums that are above the detection limit.
        :param data: The data from cf_lowpass method.
        :param sigmas: The detection limit in sigmas.
        :return: Return a list of obspy.UTCDateTime at the local maximums that are
            over the detection limit.
        """
        max_indexes = self.detect_picks(data, sigmas=sigmas)
        delay = self.get_time_delay()
        times_s = max_indexes / self._sample_rate
        events_time = []
        for t in times_s:
            events_time.append(self._start_time + t + delay)
        return events_time

    def detect_picks(self, data, sigmas=5.):
        """
        Get the local maximums that are above the detection limit.
        :param data: The data from cf_lowpass method.
        :param sigmas: The detection limit in sigmas.
        :return: Return the indexes of the local maximums over the detection limit.
        """
        limit = self.get_detection_limit(data, sigmas=sigmas)
        filtered_data = np.where(data >= limit, data, 0.)
        ind = scipy.signal.argrelextrema(filtered_data, np.greater)
        return ind[0]

    @staticmethod
    def get_detection_limit(data: np.ndarray, sigmas=5.):
        """
        Compute the detection limit of ccwt_ba_fast data.
        :param data: The data from cf_lowpass method.
        :param sigmas: The limit for detection. (Default=5 sigmas)
        :return: The detection limit sigmas * sqrt(variance)
        """
        var = np.sqrt(np.var(data))
        return sigmas * var


@deprecated(reason="You should use ConvolveWaveletScipy")
class ConvolveWavelet(ConvolveWaveletBase):
    """
    Class to apply wavelet convolution to a mseed file. The bank of atoms is computed at the class initialisation.
        Examples
        --------
        cw = ConvolveWavelet(file_path)
        convolve = cw.ccwt_ba_fast()
    """

    def __init__(self, data, **kwargs):
        """
        Class to apply wavelet convolution to a mseed file.
        The bank of atoms is computed at the class initialisation.
        :param data: Either the mseed file path or an obspy Tracer.
        :keyword kwargs:
        :keyword wmin: Minimum number of cycles. Default = 6.
        :keyword wmax: Maximum number of cycles. Default = 6.
        :keyword tt: Central frequency of the Morlet Wavelet. Default = 2.
        :keyword fmin: Minimum frequency (in Hz). Default = 2.
        :keyword fmax: Maximum frequency (in Hz). Default = 12.
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax. Default = 20.
        :keyword use_rfft: True if it should use rfft instead of fft. Default = True.
        :keyword decimate: True if it should try to decimate the trace. Default = False. The decimation
            factor is equal to q = 0.4*SR/fmax. For SR=200Hz and fmax=40Hz, q=2. This will downsize the
            sample rate to 100 Hz.
        :raise TypeError: If file is not a valid mseed.
        :example:
        >>> cw = ConvolveWavelet(data)
        >>> cw.setup_wavelet()
        >>> cf = cw.cf_lowpass()
        """

        super(ConvolveWavelet, self).__init__(data, **kwargs)

        self.__conv = None  # convolution of ba and data_fft
        self.__n_conv = 0

    def _setup_atoms(self):
        super()._setup_atoms()
        self._convolve_atoms()

    # def __chop_data(self, delta_time, start_time: UTCDateTime, end_time: UTCDateTime):
    #     total_time = (end_time - start_time) / 3600.
    #     n = np.math.ceil(total_time / delta_time)
    #
    #     data_set = []
    #     for h in range(n):
    #         dt = h * 3600 * delta_time
    #         dt2 = (h + 1) * 3600 * delta_time
    #         data = self.__get_data_in_time(start_time + dt, start_time + dt2)
    #         if data is not None:
    #             self._npts = int(self.stats.Sampling_rate * delta_time * 3600) + 1
    #             data = self.__pad_data(data, self._npts)
    #             data_set.append(data)
    #
    #     return data_set

    def _convolve_atoms(self, parallel=False):

        # FFT parameters
        n_kern = len(self._wtime)
        self.__n_conv = 2 ** np.math.ceil(np.math.log2(self._npts + n_kern))

        # loop over frequencies
        array_size = self.__n_conv / 2 + 1 if self._use_rfft else self.__n_conv
        self.__conv = np.zeros((int(self._nf), int(array_size)), dtype=np.complex64)
        # FFT data
        if self._use_rfft:
            data_fft = np.fft.rfft(self._data, n=self.__n_conv)
        else:
            data_fft = np.fft.fft(self._data, n=self.__n_conv)

        for ii, fi in enumerate(self._frex):
            cmw = self.filter_win(fi, ii)
            if self._use_rfft:
                # Calculate the fft of the "atom"
                cmw_fft = np.fft.rfft(cmw, self.__n_conv)
            else:
                cmw_fft = np.fft.fft(cmw, self.__n_conv)

            # convolution of ba and data_fft.
            self.__conv[ii, :] = np.multiply(cmw_fft, data_fft, dtype=np.complex64)

    def __compute_cwt(self, data):
        start = int(self._half_wave + 1)
        end = self._npts + int(self._half_wave + 1)
        if self._use_rfft:
            cwt = np.fft.irfft(data)[start:end]
        else:
            cwt = np.fft.ifft(data, n=self.__n_conv)[start:end]
        # subtract the mean value
        cwt = cwt - np.mean(cwt, dtype=np.float32)
        return cwt

    def __cwt_ba(self, parallel=False):
        """
        Compute the time frequency or scalogram in time and frequency domain.
        :param parallel: True if it should run in parallel. If the computer has only 1 core this will have no effect.
        :return: The time frequency representation of the convolved waveform with the wavelet.
            The type is a np.array.
        """

        n_proc = self.get_nproc()
        if parallel and n_proc > 1:
            # pool = ThreadPool(processes=n_proc)
            # results = [pool.apply_async(self.__compute_cwt, args=(row,)) for row in m]
            # tf = np.array([p.get() for p in results], copy=False, dtype=np.float32)
            # pool.close()
            with ThreadPool(n_proc) as pool:
                tf = np.array(pool.map(self.__compute_cwt, self.__conv), copy=False, dtype=np.float32)

        else:
            tf = np.array([self.__compute_cwt(row) for row in self.__conv], copy=False, dtype=np.float32)

        # release conv from memory.
        self.__conv = None
        del self.__conv

        return tf

    # def __ccwt_ba_multitread(self):
    #     nproc = self.get_nproc()
    #     nproc = min(nproc, len(self._data))
    #
    #     with ThreadPool(nproc) as pool:
    #         ro = pool.map(self.__cwt_ba, self._data)
    #
    #     cwt = np.array([]).reshape(self._nf, 0)
    #     for index, r in enumerate(ro):
    #         cwt = np.concatenate((cwt, r), axis=1)
    #
    #     return cwt

    def compute_tf(self, parallel=True):
        """
        Compute the convolved waveform with the wavelet from fmin to fmax.
        :param parallel: Either or not it should run cwt in parallel. Default=True.
        :return:
        """
        self._validate_data()
        self._tf = self.__cwt_ba(parallel=parallel)


class ConvolveWaveletScipy(ConvolveWaveletBase):
    """
    Class to apply wavelet convolution to a mseed file. The bank of atoms is computed at the class initialisation.
        Examples
        --------
        cw = ConvolveWavelet(file_path)
        convolve = cw.ccwt_ba_fast()
    """

    def __init__(self, data, **kwargs):
        """
        Class to apply wavelet convolution to a mseed file.
        The bank of atoms is computed at the class initialisation.
        :param data: Either the mseed file path or an obspy Tracer.
        :keyword kwargs:
        :keyword wmin: Minimum number of cycles. Default = 6.
        :keyword wmax: Maximum number of cycles. Default = 6.
        :keyword tt: Central frequency of the Morlet Wavelet. Default = 2.
        :keyword fmin: Minimum frequency (in Hz). Default = 2.
        :keyword fmax: Maximum frequency (in Hz). Default = 12.
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax. Default = 20.
        :keyword use_rfft: True if it should use rfft instead of fft. Default = True.
        :keyword decimate: True if it should try to decimate the trace. Default = False. The decimation
            factor is equal to q = 0.4*SR/fmax. For SR=200Hz and fmax=40Hz, q=2. This will downsize the
            sample rate to 100 Hz.
        :raise TypeError: If file is not a valid mseed.
        :example:
        >>> cw = ConvolveWavelet(data)
        >>> cw.setup_wavelet()
        >>> cf = cw.cf_lowpass()
        """

        super().__init__(data, **kwargs)

    def __convolve(self, freq: tuple):
        freq, index = freq
        cmw = self.filter_win(freq, index)
        return scipy.signal.oaconvolve(self._data, cmw, mode='same')

    def _convolve_atoms(self, parallel):

        d_type = np.float32 if self._use_rfft else np.complex64

        n_proc = self.get_nproc()
        if parallel and n_proc > 1:
            with ThreadPool(n_proc) as pool:
                tf = np.array(pool.map(self.__convolve, [(fi, i) for i, fi in enumerate(self._frex)]),
                              copy=False, dtype=d_type)
        else:
            tf = np.array([self.__convolve((fi, i)) for i, fi in enumerate(self._frex)], copy=False, dtype=d_type)

        return tf

    def compute_tf(self, parallel=True):
        """
        Compute the convolved waveform with the wavelet from fmin to fmax.
        :param parallel: Either or not it should run cwt in parallel. Default=True.
        :return:
        """
        self._validate_data()
        self._tf = self._convolve_atoms(parallel)
