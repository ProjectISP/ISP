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
from matplotlib import pyplot as plt
# from dask import delayed
from obspy import read, UTCDateTime
from obspy.signal.filter import highpass, lowpass
from obspy.signal.trigger import classic_sta_lta

from isp import ROOT_DIR
from isp.Utils import time_method


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

    ##Wavelet parameters
    dt = 1/srate

    #time = (np.arange(0, npts-1, 1))/srate
    frex = np.logspace(np.log10(fmin), np.log10(fmax), nf, base=10)  # Logarithmically space central frequencies
    # print("Frequncies = ", np.shape(frex))
    # print("Frequncies = ", frex)
    wtime = np.arange(-tt, tt+dt, dt)  # Kernel of the Mother Morlet Wavelet
    half_wave = (len(wtime)-1)/2
    nCycles=np.logspace(np.log10(wmin), np.log10(wmax), nf)

    ###FFT parameters
    nKern = len(wtime)





    nConv = npts+nKern

    nConv=2**math.ceil(math.log2(nConv))
    diff1=nConv-nKern
    diff2=nConv-npts
    #####Proyect#####
    if (nConv % 2) == 0:
       nConv2 = nConv/2 + 1
    else:
       nConv2 = (nConv+1)/2

    nConv2=int(nConv2)
    # tf = np.zeros((len(frex), nConv2))
    # tf = np.transpose(tf)

    ##loop over frequencies
    tf = []
    for fi in range(len(frex)):
        ##Create the morlet wavelet and get its fft
        s = nCycles[fi]/(2*np.pi*frex[fi])
        #Normalize Factor
        A = 1/(np.pi*s**2)**0.25
        #complexsine = np.multiply(1j*2*(np.pi)*frex[fi],wtime))
        #Gaussian = np.exp(-1*np.divide(np.power(wtime,2),2*s**2))
        cmw = np.multiply(np.exp(np.multiply(1j*2*(np.pi)*frex[fi],wtime)),np.exp(-1*np.divide(np.power(wtime,2),2*s**2)))
        cmw = cmw.conjugate()
        #Normalizing.The square root term causes the wavelet to be normalized to have an energy (squared integral) of 1.
        cmw = A*cmw
        cmw = np.real(cmw)
        #Calculate the fft of the "atom"
        cmwX = np.fft.rfft(cmw, nConv)

        #Convolution
        # tf[fi, :] = cmwX
        tf.append(cmwX)

    tf = np.asarray(tf)

    return tf, nConv, frex, half_wave


def ccwt_ifft(data, n, half_wave, npts):
    # t0 = time.time()
    cwt = np.fft.irfft(data, n=n)
    d = np.diff(np.log10(np.abs(cwt[int(half_wave + 1):npts + int(half_wave + 1)])))
    # print("I took: ", time.time() - t0, " s to finish")
    return d

def get_nproc():
    total_cpu = multiprocessing.cpu_count()
    nproc = total_cpu - 2 if total_cpu > 3 else total_cpu - 1
    nproc = max(nproc, 1)
    return nproc


def ccwt_ba_fast(data, param: tuple, parallel=False):
    ba, nConv, frex, half_wave = param
    t0 = time.time()
    npts = len(data)
    # print("Data size: ", npts)
    # print("nConv: ", nConv)

    ##FFT data
    data_fft = np.fft.rfft(data, n=nConv)

    ba = np.real(ba)
    data_fft = np.real(data_fft)
    # Should u remove the mean value of FFT to remove 0 frequencies ?
    # print(np.mean(data_fft))
    # data_fft = data_fft - np.mean(data_fft)
    m = np.multiply(ba, data_fft)
    print("Before: ", time.time() - t0)

    t0 = time.time()
    parallel = parallel if len(frex) > 1 else False
    if parallel:
        nproc = get_nproc()
        nproc = min(nproc, len(frex))
        print("Process: ", nproc)
        pool = ThreadPool(processes=nproc)
        # results = pool.map_async(partial(ccwt_ifft, n=nConv, half_wave=half_wave, npts=npts), [row for row in m])
        results = [pool.apply_async(ccwt_ifft, args=(row, nConv, half_wave, npts)) for row in m]
        tf = [p.get() for p in results]
        pool.close()
        # pool.join()
    else:
        tf = []
        for row in m:
            tf.append(ccwt_ifft(row, nConv, half_wave, npts))
    print("Ifft: ", time.time() - t0)

    t0 = time.time()
    tf = np.asarray(tf)  # convert to array
    sc = np.sum(tf, axis=0, dtype=np.float64)
    print("After: ", time.time() - t0)

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

    data=highpass(data, 0.5, 50, corners=3,zerophase=True)
    cft = classic_sta_lta(data, int(1 * sampling_rate), int(40 * sampling_rate))
    return cft


def f(x):
    return x*x


def get_pick_time(data, sampling_rate, start_time):
    max_index = np.argmax(data)
    time_s = max_index / sampling_rate
    if type(start_time) == str:
        start_time = UTCDateTime(start_time)
    time = start_time + time_s
    return time


def print_result(x):
    print(x)


def get_data(hours, chop_data=True, nf=40):
    wmin = 5
    wmax = 5
    tt = 2
    fmin = 2
    fmax = 12

    file_path = os.path.join(ROOT_DIR, "260", "RAW", "WM.OBS01..SHZ.D.2015.260")
    st = read(file_path)
    sampling_rate = st[0].stats.sampling_rate
    start_time = st[0].stats.starttime

    if not chop_data:
        span = hours * 3600
        st = read(file_path, starttime=start_time, endtime=start_time + span)
        tr = st[0]
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
    atoms = compute_atoms(npts, sampling_rate, fmin, fmax, wmin, wmax, tt, nf)

    data_set = []

    for h in range(n):
        dt = h * 3600 * delta_t
        dt2 = (h + 1) * 3600 * delta_t
        st = read(file_path, starttime=start_time + dt, endtime=start_time + dt2)
        tr = st[0]
        tr.taper(max_percentage=0.05)
        tr.filter('bandpass', freqmin=0.5, freqmax=14, corners=3, zerophase=True)
        data_set.append(tr.data)

    return data_set, atoms


class TestCWUT(unittest.TestCase):

    def setUp(self):
        self.hours = 22
        self.data_set, self.atoms = get_data(self.hours, False)
        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self):
        """finish any test"""
        p = Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('cumtime')
        p.print_stats()

    # @time_method(loop=1)
    def test_treading_fft(self):
        t0 = time.time()
        with ThreadPool() as pool:
            ro = pool.map(partial(ccwt_ba_fast, param=self.atoms), self.data_set)
        data = np.concatenate(ro)
        print(time.time() - t0)
        data = lowpass(data, 0.1, 50, corners=3, zerophase=True)
        n_0 = (15*3600+10*60)*50
        n_f = (15*3600+10*60 + 240)*50
        plt.plot(data)
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
        plt.plot(data[n_0:n_f])
        plt.show()


    # @time_method(loop=1)
    def test_cwt2(self):
        output = []
        t0 = time.time()
        for data in self.data_set:
            sc = ccwt_ba_fast(data, self.atoms, parallel=True)
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

if __name__ == '__main__':
    cProfile.run('slow()')
    unittest.main()
