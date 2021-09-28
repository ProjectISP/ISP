import numpy as np
import math

from numba import jit


class noise_processing:

    def __init__(self, tr):
        self.tr = tr

    def normalize(self, clip_factor=6, clip_weight=10, norm_win=None, norm_method="1bit"):

        if norm_method == 'clipping':
            lim = clip_factor * np.std(self.tr.data)
            self.tr.data[self.tr.data > lim] = lim
            self.tr.data[self.tr.data < -lim] = -lim

        elif norm_method == "clipping_iter":
            lim = clip_factor * np.std(np.abs(self.tr.data))

            # as long as still values left above the waterlevel, clip_weight
            while self.tr.data[np.abs(self.tr.data) > lim] != []:
                self.tr.data[self.tr.data > lim] /= clip_weight
                self.tr.data[self.tr.data < -lim] /= clip_weight

        elif norm_method == 'ramn':
            lwin = int(self.tr.stats.sampling_rate * norm_win)
            st = 0  # starting point
            N = lwin  # ending point

            while N < self.tr.stats.npts:
                win = self.tr.data[st:N]
                w = np.mean(np.abs(win)) / (2. * lwin + 1)

                # weight center of window
                self.tr.data[int(st + lwin / 2)] /= w

                # shift window
                st += 1
                N += 1

            # taper edges
            taper = self.get_window(self.tr.stats.npts)
            self.tr.data *= taper

        elif norm_method == "1bit":
            self.tr.data = np.sign(self.tr.data)
            self.tr.data = np.float32(self.tr.data)


    def get_window(self, N, alpha=0.2):

        window = np.ones(N)
        x = np.linspace(-1., 1., N)
        ind1 = (abs(x) > 1 - alpha) * (x < 0)
        ind2 = (abs(x) > 1 - alpha) * (x > 0)
        window[ind1] = 0.5 * (1 - np.cos(np.pi * (x[ind1] + 1) / alpha))
        window[ind2] = 0.5 * (1 - np.cos(np.pi * (x[ind2] - 1) / alpha))
        return window


    def whiten_new(self, freq_width=0.02, taper_edge=False):

        """"
        freq_width: Frequency smoothing windows [Hz] / both sides
        taper_edge: taper with cosine window  the low frequencies

        return: whithened trace (Phase is not modified)
        """""
        tr = self.tr.copy()
        fs = self.tr.stats.sampling_rate
        N = self.tr.count()
        D = 2 ** math.ceil(math.log2(N))
        freq_res = 1 / (D / fs)
        # N_smooth = int(freq_width / (2 * freq_res))
        N_smooth = int(freq_width / (freq_res))

        if N_smooth % 2 == 0:  # To have a central point
            N_smooth = N_smooth + 1
        else:
            pass

        # avarage_window_width = (2 * N_smooth + 1) #Denominador
        avarage_window_width = (N_smooth + 1)  # Denominador
        half_width = int((N_smooth + 1) / 2)  # midpoint
        half_width_pos = half_width - 1

        # Prefilt
        #self.tr.detrend(type='simple')
        #self.tr.taper(max_percentage=0.05)

        # ready to whiten
        data = self.tr.data
        data_f = np.fft.rfft(data, D)
        #freq = np.fft.rfftfreq(D, 1. / fs)
        N_rfft = len(data_f)
        data_f_whiten = data_f.copy()
        index = np.arange(0, N_rfft - half_width, 1)

        for j in index:
            den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
            if (den == 0):
                #den = 0.000001
                print("warning value of den is",den)
            # den = np.mean(np.abs(data_f[j:j + 2 * half_width]))
            data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den

        # Taper (optional) and remove mean diffs in edges of the frequency domain

        wf = (np.cos(np.linspace(np.pi / 2, np.pi, half_width)) ** 2)

        if taper_edge:

            diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])) * wf)

        else:

            diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])))

        diff_mean2 = np.abs(
            np.mean(np.abs(data_f[(N_rfft - half_width):])) - np.mean(np.abs(data_f_whiten[(N_rfft - half_width):])))

        data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean)  # First part of spectrum
        data_f_whiten[(N_rfft - half_width):] = (data_f[(N_rfft - half_width):]) / diff_mean2  # end of spectrum
        data = np.fft.irfft(data_f_whiten)
        data = data[0:N]

        self.tr.data = data

    @jit(nopython=True, parallel=True)
    def whiten_aux(self, data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
         for j in index:
             den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
             data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den
         return data_f_whiten
    #
    #
    # def whiten_new(self, freq_width=0.02, taper_edge=True):
    #
    #     """"
    #     freq_width: Frequency smoothing windows [Hz] / both sides
    #     taper_edge: taper with cosine window  the low frequencies
    #
    #     return: whithened trace (Phase is not modified)
    #     """""
    #
    #     fs = self.tr.stats.sampling_rate
    #     N = self.tr.count()
    #     D = 2 ** math.ceil(math.log2(N))
    #     freq_res = 1 / (D / fs)
    #     # N_smooth = int(freq_width / (2 * freq_res))
    #     N_smooth = int(freq_width / (freq_res))
    #
    #     if N_smooth % 2 == 0:  # To have a central point
    #         N_smooth = N_smooth + 1
    #     else:
    #         pass
    #
    #     # avarage_window_width = (2 * N_smooth + 1) #Denominador
    #     avarage_window_width = (N_smooth + 1)  # Denominador
    #     half_width = int((N_smooth + 1) / 2)  # midpoint
    #     half_width_pos = half_width - 1
    #
    #     # Prefilt
    #     self.tr.detrend(type='simple')
    #     self.tr.taper(max_percentage=0.05)
    #
    #     # ready to whiten
    #     data = self.tr.data
    #     data_f = np.fft.rfft(data, D)
    #     freq = np.fft.rfftfreq(D, 1. / fs)
    #     N_rfft = len(data_f)
    #     data_f_whiten = data_f.copy()
    #     index = np.arange(0, N_rfft - half_width, 1)
    #
    #     data_f_whiten = self.whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)
    #
    #     # Taper (optional) and remove mean diffs in edges of the frequency domain
    #
    #     wf = (np.cos(np.linspace(np.pi / 2, np.pi, half_width)) ** 2)
    #
    #     if taper_edge:
    #
    #         diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])) * wf)
    #
    #     else:
    #
    #         diff_mean = np.abs(np.mean(np.abs(data_f[0:half_width])) - np.mean(np.abs(data_f_whiten[half_width:])))
    #
    #     diff_mean2 = np.abs(
    #         np.mean(np.abs(data_f[(N_rfft - half_width):])) - np.mean(np.abs(data_f_whiten[(N_rfft - half_width):])))
    #
    #     if taper_edge:
    #
    #         data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean) * wf  # First part of spectrum tapered
    #     else:
    #
    #         data_f_whiten[0:half_width] = ((data_f[0:half_width]) / diff_mean)
    #
    #     data_f_whiten[(N_rfft - half_width):] = (data_f[(N_rfft - half_width):]) / diff_mean2  # end of spectrum
    #     data = np.fft.irfft(data_f_whiten)
    #     data = data[0:N]
    #     self.tr.data = data
    #
