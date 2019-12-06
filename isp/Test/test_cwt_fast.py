import cProfile
import math
import multiprocessing
import os
import time
import unittest
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from pstats import Stats

import numpy as np
import obspy
from matplotlib import pyplot as plt, dates
from obspy import read, UTCDateTime, Trace
from obspy.signal.filter import highpass, lowpass, bandpass
from obspy.signal.trigger import classic_sta_lta

from isp import ROOT_DIR
from isp.DataProcessing import ConvolveWavelet
from isp.Utils import time_method
from isp.c_lib import ccwt_cy


def compute_atoms(npts, srate,fmin, fmax, wmin, wmax,tt, nf):
    """
    Continuous Wavelet Transformation, using Morlet Wavelet

    :param data: time dependent signal.
    :param srate: sampling frequency
     Tradeoff between time and frequency resolution (number of cycles, wmin, wmax)
    :param wmin: number of cycles minumum
    :param wmin: number of cycles maximum,
    :Central frequency of the Morlet Wavelet
    :param fmin: minimum frequency (in Hz)
    :param fmax: maximum frequency (in Hz)
    :param nf: number of logarithmically spaced frequencies between fmin and

    :return: time frequency representation of st, type numpy.ndarray of complex values, shape = (nf, len(st)).
    """

    # Wavelet parameters
    dt = 1/srate
    frex = np.logspace(np.log10(fmin), np.log10(fmax), nf, base=10)  # Logarithmically space central frequencies
    wtime = np.arange(-tt, tt+dt, dt)  # Kernel of the Mother Morlet Wavelet
    half_wave = (len(wtime) - 1)/2
    n_cycles = np.logspace(np.log10(wmin), np.log10(wmax), nf)

    # FFT parameters
    n_kern = len(wtime)
    n_conv = npts + n_kern
    n_conv = 2 ** math.ceil(math.log2(n_conv))

    # loop over frequencies
    ba = []
    for ii, fi in enumerate(frex):
        # Create the Morlet wavelet and get its fft
        s = n_cycles[ii]/(2*np.pi*fi)
        # Normalize Factor
        normalization = 1/(np.pi*s**2)**0.25
        # Complex sine = np.multiply(1j*2*(np.pi)*frex[fi],wtime))
        # Gaussian = np.exp(-1*np.divide(np.power(wtime,2),2*s**2))
        cmw = np.multiply(np.exp(np.multiply(1j*2*np.pi*fi, wtime)), np.exp(-1*np.divide(np.power(wtime, 2), 2*s**2)))
        cmw = cmw.conjugate()
        # Normalizing. The square root term causes the wavelet to be normalized to have an energy of 1.
        cmw = normalization * cmw
        cmw = np.real(cmw)
        # Calculate the fft of the "atom"
        cmw_fft = np.fft.rfft(cmw, n_conv)

        # Convolution
        ba.append(cmw_fft)

    ba = np.asarray(ba)

    return ba, n_conv, frex, half_wave


def ccwt_ifft(data, n, half_wave, npts):
    cwt = np.fft.irfft(data, n=n)
    cwt = cwt - np.mean(cwt)
    d = np.diff(np.log10(np.abs(cwt[int(half_wave + 1):npts + int(half_wave + 1)])))
    return d


def get_nproc():
    total_cpu = multiprocessing.cpu_count()
    nproc = total_cpu - 2 if total_cpu > 3 else total_cpu - 1
    nproc = max(nproc, 1)
    return nproc


def ccwt_ba_fast(data, param: tuple, parallel=False):
    ba, nConv, frex, half_wave = param
    npts = len(data)

    # FFT data
    data_fft = np.fft.rfft(data, n=nConv)
    data_fft = data_fft - np.mean(data_fft)
    m = np.multiply(ba, data_fft)

    parallel = parallel if len(frex) > 1 else False
    if parallel:
        nproc = get_nproc()
        nproc = min(nproc, len(frex))
        pool = ThreadPool(processes=nproc)
        results = [pool.apply_async(ccwt_ifft, args=(row, nConv, half_wave, npts)) for row in m]
        tf = [p.get() for p in results]
        pool.close()
    else:
        tf = []
        for row in m:
            tf.append(ccwt_ifft(row, nConv, half_wave, npts))

    tf = np.asarray(tf)  # convert to array
    sc = np.mean(tf, axis=0, dtype=np.float64)

    return sc


def wrap_envelope(data, srate):
    N = len(data)
    data = highpass(data, 0.5, srate, corners=3, zerophase=True)
    D = 2 ** math.ceil(math.log2(N))

    z = np.zeros(D - N)

    data = np.concatenate((data, z), axis=0)
    ###Necesary padding with zeros
    data_envelope = obspy.signal.filter.envelope(data)
    data_envelope = data_envelope[0:N]
    return data_envelope


def sta_lta(data,sampling_rate, sta, lta):

    data = highpass(data, 0.5, 50, corners=3,zerophase=True)
    cft = classic_sta_lta(data, int(1 * sampling_rate), int(40 * sampling_rate))
    return cft


def f(x):
    return x*x


def get_pick_time(data, sampling_rate, start_time):
    max_index = np.argmax(data)
    print(max_index)
    time_s = max_index / sampling_rate
    if type(start_time) == str:
        start_time = UTCDateTime(start_time)
    event_time = start_time + time_s + 6./(2.*2.*np.pi*2.)
    return event_time


def print_result(x):
    print(x)


def get_data(hours, chop_data=True, nf=20):
    wmin = 6.
    wmax = 6.
    tt = 2.
    fmin = 2.
    fmax = 12.

    file_path = os.path.join(ROOT_DIR, "260", "RAW", "WM.OBS01..SHZ.D.2015.260")
    st = read(file_path)
    sampling_rate = st[0].stats.sampling_rate
    start_time = st[0].stats.starttime

    if not chop_data:
        span = hours * 3600
        st = read(file_path, starttime=start_time, endtime=start_time + span)
        tr = st[0]
        tr.detrend(type='demean')
        tr.taper(max_percentage=0.05)
        tr.filter('bandpass', freqmin=0.5, freqmax=14, corners=3, zerophase=True)
        data = tr.data
        npts = int(sampling_rate * hours * 3600)
        atoms = compute_atoms(npts, sampling_rate, fmin, fmax, wmin, wmax, tt, nf)
        return data, atoms

    # delta_t = 5. if hours >= 5 else hours  # in hours
    delta_t = .5
    n = int(hours / delta_t)
    npts = int(sampling_rate * delta_t * 3600)
    atoms = ccwt_cy.compute_atoms(npts, sampling_rate, fmin, fmax, wmin, wmax, tt, nf)

    data_set = []

    for h in range(n):
        dt = h * 3600 * delta_t
        dt2 = (h + 1) * 3600 * delta_t
        st = read(file_path, starttime=start_time + dt, endtime=start_time + dt2)
        # print(npts, st[0].stats.npts)
        if st:
            tr = st[0]
            tr.detrend(type='demean')
            tr.taper(max_percentage=0.05)
            # tr.filter('bandpass', freqmin=0.5, freqmax=14, corners=3, zerophase=True)
            data_set.append(tr.data)

    return data_set, atoms


class TestCWUT(unittest.TestCase):

    def setUp(self):
        self.hours = 26
        self.data_set, self.atoms = get_data(self.hours, True)
        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self):
        """finish any test"""
        p = Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('cumtime')
        p.print_stats()

    def test_plot_picks(self):
        file_path = os.path.join("/media/junqueira/DATA/Eva_geysir", "VI.G1..HHZ.2017.349.mseed")
        cw = ConvolveWavelet(file_path, fmin=4., fmax=40., nf=40, chop_data=False)
        sc = cw.ccwt_ba_fast()


        file_path = os.path.join("/media/junqueira/DATA/Eva_geysir", "2017_12_15_eruption_picks.txt")
        pick_times = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                split_line = line.split()
                time_str = "{} {}".format(split_line[0], split_line[1])
                pick_times.append(UTCDateTime(time_str).matplotlib_date)
        print(pick_times)
        print("Picks: ", len(pick_times))
        fig, ax = plt.subplots()
        ax.plot_date(pick_times, np.ones(len(pick_times)))
        for sig in [2.5, 2.6, 2.7, 2.8, 2.9, 3., 4., 5.]:
            auto_pick = cw.detect_picks_in_time(sc, sig)
            auto_pick = [t.matplotlib_date for t in auto_pick]
            if math.isclose(sig, 3.):
                print(auto_pick)

            print("Picks {} sigmas: {}".format(sig, len(auto_pick)))
            ax.plot_date(auto_pick, sig * np.ones(len(auto_pick)))

        fig.autofmt_xdate()
        plt.show()




    def test_cwt_class(self):
        # file_path = os.path.join(ROOT_DIR, "260", "RAW", "WM.OBS01..SHZ.D.2015.260")
        file_path = os.path.join("/media/junqueira/DATA/Eva_geysir", "VI.G1..HHZ.2017.349.mseed")
        st = read(file_path)
        tr = st[0]
        tr.taper(max_percentage=0.01)
        wf = tr.data
        wf = bandpass(wf, freqmin=4, freqmax=12, df=200, corners=3, zerophase=True)
        wf /= np.amax(wf)
        cw = ConvolveWavelet(file_path, fmin=4., fmax=40., nf=50, chop_data=False)
        print(cw)
        t0 = time.time()
        data = cw.ccwt_ba_fast()
        print(time.time() - t0)
        data /= np.amax(data)
        n_0 = (15 * 3600 + 10 * 60) * 50
        n_f = (15 * 3600 + 10 * 60 + 240) * 50
        print("Max value at index: ", np.argmax(data[n_0:n_f]))
        # print("Max value at time: ", get_pick_time(data, 50., "2015-09-17T00:00:27.840000Z"))
        print("Max value at time: ", cw.detect_max_pick_in_time(data))
        print("Max values at time: ", cw.detect_picks_in_time(data, sigmas=5))
        sigma = np.sqrt(np.var(data))
        s_p = np.zeros(shape=len(data))
        s_p[:] = 5 * sigma
        plt.plot(data)
        plt.plot(wf)
        plt.plot(s_p)
        plt.plot(-1*s_p)
        for indx in cw.detect_picks(data, 5):
            plt.axvline(indx, color="green")
        plt.show()

    # @time_method(loop=1)
    def test_treading_fft(self):
        from scipy.signal import sosfilt, iirfilter

        t0 = time.time()
        with ThreadPool(10) as pool:
            ro = pool.map(partial(ccwt_cy.ccwt_ba_fast, param=self.atoms), self.data_set)

        data = np.array([])
        for r in ro:
            tr = Trace(r)
            tr.taper(max_percentage=0.05)
            data = np.concatenate((data, tr.data))
        print(time.time() - t0)
        sos = iirfilter(4, 0.1, btype='low', ftype='butter', fs=50, output='sos')
        data3 = sosfilt(sos, data)
        data2 = lowpass(data, 0.1, df=50, corners=3, zerophase=False)
        data = lowpass(data, 0.1, df=50, corners=3, zerophase=True)
        n_0 = (15 * 3600 + 10 * 60) * 50
        n_f = (15 * 3600 + 10 * 60 + 240) * 50
        # print("Max value at index: ", np.argmax(data2[n_0:n_f]))
        print("Max value at index: ", np.argmax(data[n_0:n_f]))
        # print("Max value at time: ", get_pick_time(data2, 50., "2015-09-17T00:00:27.840000Z"))
        # print("Max value at time: ", get_pick_time(data3, 50., "2015-09-17T00:00:27.840000Z"))
        print("Max value at time: ", get_pick_time(data, 50., "2015-09-17T00:00:27.840000Z"))
        plt.plot(data)
        # plt.plot(data2)
        # plt.plot(data3[n_0:n_f])
        plt.show()

    @time_method(loop=1)
    def test_multiprocess_fft(self):
        with multiprocessing.Pool() as pool:
            ro = pool.map(partial(ccwt_ba_fast, param=self.atoms), self.data_set)
            # ro = pool.map_async(f, [i for i in range(12)], callback=print_result)
            # ro.wait()
            # results = ro.get()
        # print(result_objects)
        # pool.join()
        # data = np.concatenate(ro)
        # data = lowpass(data, 0.1, 50, corners=3, zerophase=True)
        # n_0 = (15*3600+10*60)*50
        # n_f = (15*3600+10*60 + 240)*50
        # plt.plot(data)
        # plt.show()

    @time_method(loop=1)
    def test_multiprocess_apply(self):
        with multiprocessing.Pool() as pool:
            ro = pool.map_async(f, [i for i in range(12)])
            r = ro.get()
        pool.join()
        print(r)

    # @time_method(loop=1)
    def test_cwt(self):
        t0 = time.time()
        data = ccwt_ba_fast(self.data_set, self.atoms, parallel=True)
        print(time.time() - t0)
        # data = arr_mul(self.data_set, self.atoms)
        data = lowpass(data, 0.1, 50, corners=3, zerophase=True)
        n_0 = (15 * 3600 + 10 * 60) * 50
        n_f = (15 * 3600 + 10 * 60 + 240) * 50
        print("Max value at index: ",  np.argmax(data[n_0:n_f]))
        print("Max value at time: ", get_pick_time(data, 50., "2015-09-17T00:00:27.840000Z"))
        plt.plot(data)
        plt.show()

    # @time_method(loop=1)
    def test_cwt2(self):
        output = []
        t0 = time.time()
        for data in self.data_set:
            sc = ccwt_ba_fast(data, self.atoms, parallel=False)
            output.append(sc)

        data = np.concatenate(output)
        print(time.time() - t0)
        data = lowpass(data, 0.1, 50, corners=3, zerophase=True)
        n_0 = (15 * 3600 + 10 * 60) * 50
        n_f = (15 * 3600 + 10 * 60 + 240) * 50
        plt.plot(data)
        plt.show()

    @time_method(loop=1)
    def test_snr(self):
        # data_2, atoms_2 = get_data(23, False)
        data = sta_lta(self.data_set, 50, 1, 40)
        # data = lowpass(data, 0.1, 50, corners=3, zerophase=True)
        # n_0 = (15 * 3600 + 10 * 60) * 50
        # n_f = (15 * 3600 + 10 * 60 + 240) * 50
        # plt.plot(data[n_0:n_f])
        # plt.show()

    def test_plot_performance(self):
        performance = {"ccwt": [], "env": [], "snr": []}
        hours = []
        nproc = get_nproc()
        pool = ThreadPool(processes=5)
        for h in range(24):
            print("Doing for, ", h + 1, "h data.")
            hours.append(h + 1)
            data, atoms = get_data(h + 1)
            data_2, atoms_2 = get_data(h + 1, False)

            # time snr
            t0 = time.time()
            sta_lta(data_2, 50, 1, 40)
            dt = time.time() - t0
            performance["snr"].append(dt)


            # ccwt method with multiprocess.
            t0 = time.time()
            ro = pool.map(partial(ccwt_ba_fast, param=atoms, parallel=False), data)
            data = np.concatenate(ro)

            # ccwt_ba_fast(data_2, atoms_2, parallel=True)
            dt = time.time() - t0
            performance["ccwt"].append(dt)

            # time envelope
            t0 = time.time()
            wrap_envelope(data_2, 50)
            dt = time.time() - t0
            performance["env"].append(dt)

        pool.close()
        for key, item in performance.items():
            print(key)
            plt.plot(hours, item, label=key)
        print(performance.keys())
        plt.legend(performance.keys())
        plt.show()

    def test_converge(self):
        max_index = []
        # plt.ion()
        n_0 = (15 * 3600 + 10 * 60) * 50
        n_f = (15 * 3600 + 10 * 60 + 240) * 50
        for nf in range(40):
            data, atoms = get_data(22, chop_data=False, nf=nf+1)
            data_out = ccwt_ba_fast(data, atoms, parallel=True)
            data_out = lowpass(data_out, 0.1, 50, corners=3, zerophase=True)
            max_index.append(np.argmax(data_out[n_0:n_f]))
            plt.xlim(-1, 41)
            plt.plot(range(1, len(max_index) + 1), max_index, 'o')
            plt.plot(range(1, len(max_index) + 1), max_index, '--')
            plt.xlabel("Nf")
            plt.ylabel("Index of the peak")
            plt.draw()
            plt.pause(1)
            plt.clf()

        plt.xlim(-1, 41)
        plt.xlabel("Nf")
        plt.ylabel("Index of the peak")
        plt.plot(range(1, len(max_index) + 1), max_index, 'o')
        plt.plot(range(1, len(max_index) + 1), max_index, '--')
        plt.show()

    def test_scipy_cwt(self):
        from scipy import signal

        dt = 1. / 50.
        frex = np.logspace(np.log10(2), np.log10(12), 20, base=10)  # Logarithmically space central frequencies
        data, atoms = get_data(22, chop_data=False)
        # cwtmatr = ccwt_ba_fast(data, atoms, parallel=True)
        widths = np.arange(-2, 2 + dt, dt)  # Kernel of the Mother Morlet Wavelet
        output = np.zeros([len(frex), len(data) - 1])
        half_wave = (len(widths) - 1) / 2
        length = len(widths)
        for ii, f in enumerate(frex):
            s = 5 / (2 * np.pi * f)
            conv = signal.convolve(data, signal.morlet(length, 5, s, complete=False), mode='same')
            d = np.diff(np.log10(np.abs(conv)))
            output[ii, :] = d

        cwtmatr = output
        sc = np.mean(cwtmatr, axis=0, dtype=np.float64)
        data = lowpass(sc, 0.1, df=50, corners=3, zerophase=True)
        plt.plot(data)
        # plt.imshow(cwtmatr, cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        plt.show()


if __name__ == '__main__':
    cProfile.run('slow()')
    unittest.main()
