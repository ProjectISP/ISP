import numpy as np
import math

from numba import jit


class noise_processing:

    def __init__(self, tr):
        self.tr = tr

    def phase_match(self, tr, interp_c, distance, filter_parameter = 0.5):

        signal = tr.data
        dt = tr.stats.delta
        n = tr.stats.npts
        tshift = (n * dt) / 2
        shifted_fft = np.fft.fftshift(np.fft.fft(signal))
        shifted_fft_bins = np.fft.fftshift(np.fft.fftfreq(len(signal), d=dt))

        phase_correction = []
        for bin_ in shifted_fft_bins:
            try:
                wavenumber = 2 * np.pi * bin_ / interp_c(np.abs(bin_))  # bin_ contiene el signo de la frecuencia ya, no hace falta meterlo en el paso siguiente
                phase_correction.append(np.exp(1j * distance * wavenumber))
            except FloatingPointError:
                phase_correction.append(np.exp(0))

        frequency_domain_compressed_signal = shifted_fft * np.array(phase_correction) * np.exp(
            -1j * tshift * 2 * np.pi * shifted_fft_bins)
        time_domain_compressed_signal = np.fft.ifft(np.fft.ifftshift(frequency_domain_compressed_signal))
        gauss_window = np.exp(-0.25 * ((np.arange(0, len(signal), 1) - len(signal) / 2)) * 2 / (filter_parameter / dt * 2))
        tshift = len(signal) * dt / 2
        #time_domain_decompressed_signal_freq = np.fft(time_domain_compressed_signal*gauss_window)*np.exp(-1j * tshift * 2 * np.pi * fft_bins)


        tr.data = time_domain_compressed_signal
        return tr

    # def phase_matched_filter2(self, signal, dt, reference_c):
    #     # reference_c es un scipy.interpolate.interp1d
    #     tshift = (len(signal) * dt) / 2  # We will shift the collapsed Rayleigh waves to the center of the time series
    #
    #     fft = np.fft.rfft(signal)
    #     fft_bins = np.fft.rfftfreq(len(signal), d=dt)
    #
    #     # Compute and apply the phase correction and time shifts, return to the time domain
    #     phase_correction = np.exp(
    #         1j * 2 * np.pi * fft_bins * np.divide(distance, reference_c(fft_bins), out=np.zeros_like(fft_bins),
    #                                               where=reference_c(fft_bins) != 0))
    #     time_shift = np.exp(-1j
    #     tshift
    #     2
    #     np.pi
    #     fft_bins)
    #     frequency_domain_compressed_signal = fft * phase_correction * time_shift
    #     time_domain_compressed_signal = np.fft.irfft(frequency_domain_compressed_signal)
    #
    #     # Apply a Gaussian window in the time domain
    #     sigma = 2 / dt  # standard deviation in seconds divided by dt gives us samples
    #     gauss_window = np.exp(-0.25 * (
    #     (np.arange(0, len(time_domain_compressed_signal), 1) - len(time_domain_compressed_signal) / 2)) ** 2 / (
    #                                       sigma ** 2))
    #     windowed_compressed_signal = time_domain_compressed_signal * gauss_window
    #     windowed_compressed_signal[np.abs(windowed_compressed_signal) < 1e-16] = 0
    #
    #     # FFT to the frequency domain, disperse the signal and back to the time domain
    #     fft = np.fft.rfft(windowed_compressed_signal)
    #     fft_bins = np.fft.rfftfreq(len(signal), d=dt)
    #     phase_correction = np.exp(
    #         -1j * distance * np.divide(2 * np.pi * fft_bins, reference_c(fft_bins), out=np.zeros_like(fft_bins),
    #                                    where=reference_c(fft_bins) != 0))
    #     time_shift = np.exp(1j
    #     tshift
    #     2
    #     np.pi
    #     fft_bins)
    #     frequency_domain_uncompressed_signal = fft
    #     phase_correction
    #     time_shift
    #     time_domain_uncompressed_signal = np.fft.irfft(frequency_domain_uncompressed_signal)

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

        #data_f_whiten = self.whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)
        for j in index:
            den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
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

class noise_processing_horizontals:

    def __init__(self, tr_N, tr_E):
        self.tr_N = tr_N
        self.tr_E = tr_E

    def normalize(self, clip_factor=6, clip_weight=10, norm_win=None, norm_method='ramn'):


        # modified to compare maximum of both means
        if norm_method == 'ramn':
            lwin = int(self.tr_N.stats.sampling_rate * norm_win)
            st = 0  # starting point
            N = lwin  # ending point

            while N < self.tr_N.stats.npts and N < self.tr_E.stats.npts:
                win_N = self.tr_N.data[st:N]
                win_E = self.tr_E.data[st:N]

                w_N = np.mean(np.abs(win_N)) / (2. * lwin + 1)
                w_E = np.mean(np.abs(win_E)) / (2. * lwin + 1)
                max_value = max(w_N, w_E)
                # weight center of window
                self.tr_N.data[int(st + lwin / 2)] /= max_value
                self.tr_E.data[int(st + lwin / 2)] /= max_value

                # shift window
                st += 1
                N += 1

            # taper edges
            taper = self.get_window(self.tr_N.stats.npts)
            self.tr_N.data *= taper
            taper = self.get_window(self.tr_E.stats.npts)
            self.tr_E.data *= taper



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
        tr_E = self.tr_E.copy()
        tr_N = self.tr_N.copy()
        fs = self.tr_N.stats.sampling_rate
        N = self.tr_N.count()
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
        data_N = self.tr_N.data
        data_E = self.tr_E.data
        data_f_N = np.fft.rfft(data_N, D)
        data_f_E = np.fft.rfft(data_E, D)
        #freq = np.fft.rfftfreq(D, 1. / fs)
        N_rfft = len(data_f_N)
        data_f_whiten_N = data_f_N.copy()
        data_f_whiten_E = data_f_E.copy()
        index = np.arange(0, N_rfft - half_width, 1)

        #data_f_whiten_N,data_f_whiten_E = self.whiten_aux_horizontals(data_f_N, data_f_whiten_N, data_f_E,
        #                        data_f_whiten_E, index, half_width, avarage_window_width, half_width_pos)
        for j in index:
            den = 0.5*(np.sum(np.abs(data_f_N[j:j + 2 * half_width]) + np.abs(data_f_E[j:j + 2 * half_width]))/avarage_window_width)
            data_f_whiten_N[j + half_width_pos] = data_f_whiten_N[j + half_width_pos] / den
            data_f_whiten_E[j + half_width_pos] = data_f_whiten_E[j + half_width_pos] / den

        # Taper (optional) and remove mean diffs in edges of the frequency domain

        wf = (np.cos(np.linspace(np.pi / 2, np.pi, half_width)) ** 2)

        if taper_edge:

            diff_mean_N = np.abs(np.mean(np.abs(data_f_N[0:half_width])) - np.mean(np.abs(data_f_whiten_N[half_width:])) * wf)
            diff_mean_E = np.abs(np.mean(np.abs(data_f_E[0:half_width])) - np.mean(np.abs(data_f_whiten_E[half_width:])) * wf)


        else:

            diff_mean_N = np.abs(np.mean(np.abs(data_f_N[0:half_width])) - np.mean(np.abs(data_f_whiten_N[half_width:])))
            diff_mean_E = np.abs(np.mean(np.abs(data_f_E[0:half_width])) - np.mean(np.abs(data_f_whiten_E[half_width:])))

        diff_mean2_N = np.abs(
            np.mean(np.abs(data_f_N[(N_rfft - half_width):])) - np.mean(np.abs(data_f_whiten_N[(N_rfft - half_width):])))
        diff_mean2_E = np.abs(
            np.mean(np.abs(data_f_E[(N_rfft - half_width):])) - np.mean(
                np.abs(data_f_whiten_E[(N_rfft - half_width):])))

        data_f_whiten_E[0:half_width] = ((data_f_E[0:half_width]) / diff_mean_E)  # First part of spectrum
        data_f_whiten_N[0:half_width] = ((data_f_N[0:half_width]) / diff_mean_N)  # First part of spectrum

        data_f_whiten_E[(N_rfft - half_width):] = (data_f_E[(N_rfft - half_width):]) / diff_mean2_E  # end of spectrum
        data_f_whiten_N[(N_rfft - half_width):] = (data_f_N[(N_rfft - half_width):]) / diff_mean2_N  # end of spectrum
        data_N = np.fft.irfft(data_f_whiten_N)
        data_E = np.fft.irfft(data_f_whiten_E)
        data_N = data_N[0:N]
        data_E = data_E[0:N]

        self.tr_N.data = data_N
        self.tr_N.data = data_E

    @jit(nopython=True, parallel=True)
    def whiten_aux_horizontals(self, data_f_N, data_f_whiten_N, data_f_E, data_f_whiten_E, index, half_width,
                               avarage_window_width, half_width_pos):
        for j in index:

            den_N = np.sum(np.abs(data_f_N[j:j + 2 * half_width])) / avarage_window_width
            den_E = np.sum(np.abs(data_f_E[j:j + 2 * half_width])) / avarage_window_width
            mean = np.mean(den_N, den_E)
            # den = np.mean(np.abs(data_f[j:j + 2 * half_width]))
            data_f_whiten_N[j + half_width_pos] = data_f_whiten_N[j + half_width_pos] / mean
            data_f_whiten_E[j + half_width_pos] = data_f_whiten_E[j + half_width_pos] / mean

        return data_f_whiten_N, data_f_whiten_E


