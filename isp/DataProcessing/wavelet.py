import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import scipy
from obspy import read, Trace
from obspy.signal.filter import lowpass
from scipy.signal import argrelextrema

from isp.Exceptions import InvalidFile
from isp.Utils import ObspyUtil, MseedUtil


class ConvolveWavelet:
    """
    Class to apply wavelet convolution to a mseed file. The bank of atoms is computed at the class initialisation.

        Examples

        --------

        cw = ConvolveWavelet(file_path)

        convolve = cw.ccwt_ba_fast()
    """

    def __init__(self, file_path, chop_data=True, **kwargs):
        """
        Class to apply wavelet convolution to a mseed file.

        The bank of atoms is computed at the class initialisation.

        :param str file_path: The mseed file path.

        :param bool chop_data: (Default = True). If True the data will be chopped in 30 mins chunks. False, will analise
            the data without chunk. Chunking the data is more efficient manly for computers with more than 1 core since
            the analise can run in parallel.

        :keyword kwargs:

        :keyword wmin: Minimum number of cycles.

        :keyword wmax: Maximum number of cycles.

        :keyword tt: Central frequency of the Morlet Wavelet.

        :keyword fmin: Minimum frequency (in Hz).

        :keyword fmax: Maximum frequency (in Hz).

        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax.

        :raise InvalidFile: If file is not a valid mseed.

        :example:
        >>> cw = ConvolveWavelet(file_path)
        >>> convolve = cw.ccwt_ba_fast()
        """

        if not MseedUtil.is_valid_mseed(file_path):
            raise InvalidFile("The file: {} is not a valid mseed.".format(file_path))

        self.file_path = file_path
        self.__is_chunked = chop_data

        self.__wmin = float(kwargs.get("wmin", 6.))
        self.__wmax = float(kwargs.get("wmax", 6.))
        self.__tt = float(kwargs.get("tt", 2.))
        self.__fmin = float(kwargs.get("fmin", 2.))
        self.__fmax = float(kwargs.get("fmax", 12.))
        self.__nf = int(kwargs.get("nf", 20))

        self.stats = ObspyUtil.get_stats(self.file_path)
        # print(self.stats)

        self.__data = None
        self.__n_conv = 0

        # set atoms params and compute ba
        self.__setup_atoms()

    def __repr__(self):
        return "ConvolveWavelet(file_path={}, chop_data={}, wmin={}, wmax={}, tt={}, fmin={}, fmax={}, nf={})".format(
            self.file_path, self.__is_chunked, self.__wmin,
            self.__wmax, self.__tt, self.__fmin, self.__fmax, self.__nf
        )

    def setup_wavelet(self, **kwargs):
        """
        Recompute the bank of atoms based on the new kwargs.

        :keyword  kwargs:
        :keyword wmin: Minimum number of cycles.
        :keyword wmax: Maximum number of cycles.
        :keyword tt: Central frequency of the Morlet Wavelet.
        :keyword fmin: Minimum frequency (in Hz).
        :keyword fmax: Maximum frequency (in Hz).
        :keyword nf: Number of logarithmically spaced frequencies between fmin and fmax.

        :return:
        """
        self.__wmin = float(kwargs.get("wmin", self.__wmin))
        self.__wmax = float(kwargs.get("wmax", self.__wmax))
        self.__tt = float(kwargs.get("tt", self.__tt))
        self.__fmin = float(kwargs.get("fmin", self.__fmin))
        self.__fmax = float(kwargs.get("fmax", self.__fmax))
        self.__nf = int(kwargs.get("nf", self.__nf))

        self.__setup_atoms()

    def __setup_atoms(self):
        self.__ba = np.array([0])
        self.__frex = np.logspace(np.log10(self.__fmin), np.log10(self.__fmax), self.__nf, base=10)
        self.__n_cycles = np.linspace(self.__wmin, self.__wmax, self.__nf)
        dt = 1 / self.stats.Sampling_rate
        self.__wtime = np.arange(-self.__tt, self.__tt + dt, dt)  # Kernel of the Mother Morlet Wavelet
        self.__half_wave = (len(self.__wtime) - 1) / 2

        if self.__data is None:
            print("Read data")
            self.__setup_wavelet(chop_data=self.__is_chunked)
        else:
            print("Only atmos")
            ntpt = len(self.__data[0]) if self.__is_chunked else len(self.__data)
            self.__compute_atoms(ntpt)

    def __get_data_in_time(self, start_time, end_time):
        st = read(self.file_path, starttime=start_time, endtime=end_time)
        if st:
            tr = st[0]
            tr.detrend(type='demean')
            tr.taper(max_percentage=0.05)
            return tr.data, tr.stats.npts
        return None, 0

    @staticmethod
    def get_nproc():
        total_cpu = multiprocessing.cpu_count()
        nproc = total_cpu - 2 if total_cpu > 3 else total_cpu - 1
        nproc = max(nproc, 1)
        return nproc

    @staticmethod
    def __pad_data(data: np.ndarray, expect_size: int):
        data_size = len(data)
        return np.pad(data, (0, expect_size - data_size), mode='constant', constant_values=0.)

    @staticmethod
    def __tapper(data, max_percentage=0.01):
        tr = Trace(data)
        tr.taper(max_percentage=max_percentage)
        return tr.data

    def __chop_data(self, delta_time):
        total_time = (self.stats.EndTime - self.stats.StartTime) / 3600.
        n = np.math.ceil(total_time / delta_time)
        npts = int(self.stats.Sampling_rate * delta_time * 3600) + 1

        data_set = []
        for h in range(n):
            dt = h * 3600 * delta_time
            dt2 = (h + 1) * 3600 * delta_time
            data, nt = self.__get_data_in_time(self.stats.StartTime + dt, self.stats.StartTime + dt2)
            if data is not None:
                data = self.__pad_data(data, npts)
                data_set.append(data)

        return data_set, npts

    def __setup_wavelet(self, chop_data):

        start_time = self.stats.StartTime
        end_time = self.stats.EndTime

        if not chop_data:
            self.__data, npts = self.__get_data_in_time(start_time, end_time)
        else:
            self.__data, npts = self.__chop_data(0.5)

        self.__compute_atoms(npts)

    def __compute_atoms(self, npts: int):

        # FFT parameters
        n_kern = len(self.__wtime)
        self.__n_conv = 2 ** np.math.ceil(np.math.log2(npts + n_kern))

        # loop over frequencies
        ba = []
        for ii, fi in enumerate(self.__frex):
            # Create the Morlet wavelet and get its fft
            s = self.__n_cycles[ii] / (2 * np.pi * fi)
            # Normalize Factor
            normalization = 1 / (np.pi * s ** 2) ** 0.25
            # Complex sine = np.multiply(1j*2*(np.pi)*frex[fi],wtime))
            # Gaussian = np.exp(-1*np.divide(np.power(wtime,2),2*s**2))
            cmw = np.multiply(np.exp(np.multiply(1j * 2 * np.pi * fi, self.__wtime)),
                              np.exp(-1 * np.divide(np.power(self.__wtime, 2), 2 * s ** 2)))
            cmw = cmw.conjugate()
            # Normalizing. The square root term causes the wavelet to be normalized to have an energy of 1.
            cmw = normalization * cmw
            cmw = np.real(cmw)
            # Calculate the fft of the "atom"
            cmw_fft = np.fft.rfft(cmw, self.__n_conv)

            # Convolution
            ba.append(cmw_fft)

        self.__ba = np.asarray(ba)

    def __ccwt_ifft(self, data, npts):
        cwt = np.fft.irfft(data, n=self.__n_conv)
        cwt = cwt - np.mean(cwt)
        d = np.diff(np.log10(np.abs(cwt[int(self.__half_wave + 1):npts + int(self.__half_wave + 1)])))
        return d

    def __ccwt_ba(self, data, parallel=False):
        npts = len(data)
        # FFT data
        data_fft = np.fft.rfft(data, n=self.__n_conv)
        data_fft = data_fft - np.mean(data_fft)
        m = np.multiply(self.__ba, data_fft)

        parallel = parallel if len(self.__frex) > 1 else False
        if parallel:
            nproc = self.get_nproc()
            nproc = min(nproc, len(self.__frex))
            pool = ThreadPool(processes=nproc)
            results = [pool.apply_async(self.__ccwt_ifft, args=(row, npts)) for row in m]
            tf = [p.get() for p in results]
            pool.close()
        else:
            tf = []
            for row in m:
                tf.append(self.__ccwt_ifft(row, npts))

        tf = np.asarray(tf)  # convert to array
        sc = np.mean(tf, axis=0, dtype=np.float64)

        return sc

    def __ccwt_ba_multitread(self):
        nproc = self.get_nproc()
        nproc = min(nproc, len(self.__data))

        with ThreadPool(nproc) as pool:
            ro = pool.map(self.__ccwt_ba, self.__data)

        sc = np.array([])
        ro_size = len(ro)
        for index, r in enumerate(ro):
            max_percentage = 0.01 if index < ro_size - 1 else 0.05  # increase tapper at the end of data.
            sc = np.concatenate((sc, self.__tapper(r, max_percentage=max_percentage)))

        return sc

    def ccwt_ba_fast(self, tapper=True):
        """
        Compute the mean values of the log10 differences of the convolved waveform with the wavelet from fmin to fmax.

        :param tapper: (Default=True)True for tapper the result.

        :return: The filtered (lowpass, fmin=0.15) mean values of the log10 difference of the convolved waveform with
            the wavelet from fmin to fmax.
        """
        sc = np.array([])
        if self.__data is not None:

            if isinstance(self.__data, list):
                sc = self.__ccwt_ba_multitread()
            else:
                sc = self.__ccwt_ba(self.__data, parallel=True)

            if tapper:
                sc = self.__tapper(sc)

            sc = lowpass(sc, 0.15, df=self.stats.Sampling_rate, corners=3, zerophase=True)
        return sc

    def get_time_delay(self):
        """
        Compute the time delay in seconds of the wavelet.
        :return: The time delay of the wavelet in seconds.
        """
        return 0.5 * self.__wmin / (2. * np.pi * self.__fmin)

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
            time_s = max_index / self.stats.Sampling_rate
            return self.stats.StartTime + time_s + self.get_time_delay()
        return None

    def detect_picks_in_time(self, data: np.ndarray, sigmas=5.):
        """
        Get the times of the local maximums that are above the detection limit.

        :param data: The data from ccwt_ba_fast method.

        :return: Return a list of obspy.UTCDateTime at the local maximums over the detection limit.
        """
        max_indexes = self.detect_picks(data, sigmas=sigmas)
        delay = self.get_time_delay()
        times_s = max_indexes / self.stats.Sampling_rate
        events_time = []
        for t in times_s:
            events_time.append(self.stats.StartTime + t + delay)
        return events_time

    def detect_picks(self, data, sigmas=5.):
        """
        Get the local maximums that are above the detection limit.

        :param data: The data from ccwt_ba_fast method.

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

        :param data: The data from ccwt_ba_fast method.

        :param sigmas: The limit for detection. (Default=5 sigmas)

        :return: The detection limit sigmas * sqrt(variance)
        """
        var = np.sqrt(np.var(data))
        return sigmas * var
