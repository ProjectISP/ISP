import cProfile
import math
import os
import time
import unittest
from pstats import Stats

import numpy as np
from matplotlib import pyplot as plt
from obspy import read, UTCDateTime
from obspy.signal.filter import bandpass

from isp.DataProcessing import ConvolveWavelet, ConvolveWaveletScipy


class TestCWUT(unittest.TestCase):

    def setUp(self):
        self.file_path = os.path.join("/media/junqueira/DATA/Eva_geysir", "VI.G1..HHZ.2017.349.mseed")

    def tearDown(self):
        """finish any test"""
        pass

    def test_plot_picks(self):
        file_path = os.path.join("/media/junqueira/DATA/Eva_geysir", "VI.G1..HHZ.2017.349.mseed")
        tr = read(file_path)[0]
        cw = ConvolveWaveletScipy(tr, fmin=4., fmax=40., nf=40)
        cw.setup_wavelet()
        sc = cw.cf_lowpass()

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
        cw = ConvolveWavelet(file_path, fmin=4., fmax=40., nf=50)
        cw.setup_wavelet()
        print(cw)
        t0 = time.time()
        data = cw.cf_lowpass()
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


if __name__ == '__main__':
    cProfile.run('slow()')
    unittest.main()
