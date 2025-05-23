import os.path
import re
import numpy as np
import math
from scipy import stats
# Cython code
#from isp.cython_code.whiten import whiten_aux, whiten_aux_horizontals
import pandas as pd
import os
from isp import DISP_REF_CURVES
from scipy import interpolate
class noise_processing:

    def __init__(self, tr):
        self.tr = tr

    @classmethod
    def sort_verticals(cls, list_item, info):

        data = list_item[1:]
        starts = info[2]
        # # Sort lists
        data = [x for _, x in sorted(zip(starts, data))]
        starts.sort()
        list_item[1:] = data
        info[2] = starts
        return list_item, info

    @classmethod
    def clean_horizontals_unique(cls, list_item_horizontals, info_N, info_E):

        data_N = list_item_horizontals["North"][1:]
        data_E = list_item_horizontals["East"][1:]
        starts_N = info_N[2]
        starts_E = info_E[2]

        idx_N_delete = []
        idx_E_delete = []
        # Sort lists
        data_N = [x for _, x in sorted(zip(starts_N, data_N))]
        data_E = [x for _, x in sorted(zip(starts_E, data_E))]
        starts_N.sort()
        starts_E.sort()

        for idx_N, value1 in enumerate(starts_N):
            value1_valid = False
            for value2 in starts_E:
                if abs(value2-value1) < 5:
                    value1_valid = True

            if value1_valid:
                value1_valid = False
            else:
                idx_N_delete.append(idx_N)

        # CLEAN N
        idx_N_list=sorted(idx_N_delete,reverse=True)
        for idx in idx_N_list:
            if idx < len(data_N):
                data_N.pop(idx)
                starts_N.pop(idx)


        for idx_E, value2 in enumerate(starts_E):
            value2_valid = False
            for value1 in starts_N:
                if abs(value2 - value1) < 5:
                    value2_valid = True

            if value2_valid:
                value2_valid = False
            else:
                idx_E_delete.append(idx_E)

        # CLEAN E
        idx_E_list=sorted(idx_E_delete,reverse=True)
        for idx in idx_E_list:
            if idx < len(data_E):
                data_E.pop(idx)
                starts_E.pop(idx)

        # Finally delete duplicated days, otherwise is almost impossible to delete it later

        starts_N_diff = np.diff(np.array(starts_N))
        starts_E_diff = np.diff(np.array(starts_E))
        idx_N_diff = np.argwhere(starts_N_diff <= 5)
        idx_E_diff = np.argwhere(starts_E_diff <= 5)

        try:

            # CLEAN N
            if len(idx_N_diff) > 0:
                idx_N_diff = idx_N_diff[0].tolist()
                idx_N_list = sorted(idx_N_diff, reverse=True)
                for idx in idx_N_list:
                    if idx < len(data_N):
                        data_N.pop(idx)
                        starts_N.pop(idx)

            # CLEAN E
            if len(idx_E_diff) > 0:
                idx_E_diff = idx_E_diff[0].tolist()
                idx_E_list = sorted(idx_E_diff, reverse=True)
                for idx in idx_E_list:
                    if idx < len(data_E):
                        data_E.pop(idx)
                        starts_E.pop(idx)
        except:
                pass


        list_item_horizontals["North"][1:] = data_N
        list_item_horizontals["East"][1:] = data_E
        info_N[2] = starts_N
        info_E[2] = starts_E

        return list_item_horizontals, info_N, info_E

    def __plot_phase_match(self, t, gauss_window, time_domain_compressed_signal, windowed_compressed_signal):

        import matplotlib.pyplot as plt
        from isp.Gui.Frames import MatplotlibFrame
        fig, ax1 = plt.subplots(figsize=(6, 6))
        self.mpf = MatplotlibFrame(fig)
        t = t[1:]
        ax1.plot(t, time_domain_compressed_signal/np.max(time_domain_compressed_signal), linewidth=0.5,
                 color='steelblue', label="time_domain_compressed_signal")
        ax1.plot(t, gauss_window, linewidth=0.5, color='green', label="Gauss pulse")
        ax1.plot(t, windowed_compressed_signal/np.max(windowed_compressed_signal), linewidth=0.5, color='red', label="windowed_compressed_signal")

        plt.ylabel('Amplitude')
        plt.xlabel('Time')
        plt.legend()
        self.mpf.show()


    def get_disp(self, type, phaseMacthmodel):

        curves = []
        modes = ["fundamental", "first"]
        if phaseMacthmodel == "ak-135f":
            curves = ["ak135_earth_velocity_fundamental_mode.txt", "ak135_earth_velocity_first_mode.txt"]

        elif phaseMacthmodel == "ak-135f (Ocean-shallow waters)":
            curves = ["ak135_earth_ocean_shallow_velocity_fundamental_mode.txt",
                      "ak135_earth_ocean_shallow_velocity_first_mode.txt"]

        elif phaseMacthmodel == "ak-135f (Ocean-intermediate waters)":

            curves = ["ak135_earth_ocean_intermediate_velocity_fundamental_mode.txt",
                      "ak135_earth_ocean_intermediate_velocity_first_mode.txt"]

        elif phaseMacthmodel == "ak-135f (Ocean-deep waters)":
            curves = ["ak135_earth_ocean_deep_velocity_fundamental_mode.txt",
                      "ak135_earth_ocean_deep_velocity_first_mode.txt"]

        all_curves = {}
        if len(curves) > 0:
            for i, curve in enumerate(curves):
                path = os.path.join(DISP_REF_CURVES, curve)
                df = pd.read_csv(path)
                all_curves[modes[i]] = {}

                if type == "Rayleigh":
                    all_curves[modes[i]]["U"] = df['group_velocity_rayleigh'].to_numpy()
                    all_curves[modes[i]]["PHV"] = df['phase_velocity_rayleigh'].to_numpy()
                    all_curves[modes[i]]["period"] = df['period'].to_numpy()
                if type == "Love":
                    all_curves[modes[i]]["U"] = df['group_velocity_love'].to_numpy()
                    all_curves[modes[i]]["PHV"] = df['phase_velocity_love'].to_numpy()
                    all_curves[modes[i]]["period"] = df['period'].to_numpy()


        return all_curves

    def __get_reference_disp(self, type, phaseMacthmodel, fft_bins):

        if phaseMacthmodel == "ak-135f":

            path = os.path.join(DISP_REF_CURVES, "ak135_earth_velocity_first_mode.txt")
            df = pd.read_csv(path)
            freq_ref = 1/(df['period'].to_numpy())
            if type == "Rayleigh":
                vel = df['phase_velocity_rayleigh'].to_numpy()
            if type == "Love":
                vel = df['phase_velocity_love'].to_numpy()

        elif phaseMacthmodel == "ak-135f (Ocean-shallow waters)":

            path = os.path.join(DISP_REF_CURVES, "ak135_earth_ocean_shallow_velocity_first_mode.txt")
            df = pd.read_csv(path)
            freq_ref = 1/(df['period'].to_numpy())
            if type == "Rayleigh":
                vel = df['phase_velocity_rayleigh'].to_numpy()
            if type == "Love":
                vel = df['phase_velocity_love'].to_numpy()

        elif phaseMacthmodel == "ak-135f (Ocean-intermediate waters)":

            path = os.path.join(DISP_REF_CURVES, "ak135_earth_ocean_intermediate_velocity_first_mode.txt")
            df = pd.read_csv(path)
            freq_ref = 1/(df['period'].to_numpy())
            if type == "Rayleigh":
                vel = df['phase_velocity_rayleigh'].to_numpy()
            if type == "Love":
                vel = df['phase_velocity_love'].to_numpy()

        elif phaseMacthmodel == "ak-135f (Ocean-deep waters)":

            path = os.path.join(DISP_REF_CURVES, "ak135_earth_ocean_deep_velocity_first_mode.txt")
            df = pd.read_csv(path)
            freq_ref = 1/(df['period'].to_numpy())
            if type == "Rayleigh":
                vel = df['phase_velocity_rayleigh'].to_numpy()
            if type == "Love":
                vel = df['phase_velocity_love'].to_numpy()
        freq_ref = np.flip(freq_ref)
        vel = np.flip(vel)
        f = interpolate.interp1d(freq_ref, vel, fill_value="extrapolate")
        vel2 = f(fft_bins)
        return freq_ref, vel2

    @staticmethod
    def statisics_fit(x, y, type, deg):
        x_new = np.array(x)
        y_new = np.array(y)

        if type == "Straight Line":
            deg = 1
            p = np.polyfit(x_new, y_new, deg)
            m = p[0]
            c = p[1]
            print(f'The fitted straight line has equation y = {m:.3f}x {c:=+6.3f}')
        elif type == "Polynom":
            p = np.polyfit(x_new, y_new, deg)
            m = p[0]
            c = p[1]
        elif type == "Logarithmic":
            log_x = np.log(x)
            p = np.polyfit(log_x, y, 1)
            m = p[0]
            c = p[1]
            print(f'The fitted Logarithmic has equation y = {c:.3f}log(x) {m:=+6.3f}')
        elif type == "Exponential":
            p = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
            m = p[0]
            c = p[1]


        # Number of observations
        n = y_new.size
        # Number of parameters: equal to the degree of the fitted polynomial (ie the
        # number of coefficients) plus 1 (ie the number of constants)
        m = p.size
        # Degrees of freedom (number of observations - number of parameters)
        dof = n - m
        # Significance level
        alpha = 0.05
        # We're using a two-sided test
        tails = 2
        # The percent-point function (aka the quantile function) of the t-distribution
        # gives you the critical t-value that must be met in order to get significance
        t_critical = stats.t.ppf(1 - (alpha / tails), dof)


        # Model the data using the parameters of the fitted straight line
        y_model = np.polyval(p, x_new)

        # Create the linear (1 degree polynomial) model
        model = np.poly1d(p)
        # Fit the model
        y_model = model(x_new)
        # Mean
        y_bar = np.mean(y_new)
        # Coefficient of determination, R²
        R2 = np.sum((y_model - y_bar) ** 2) / np.sum((y_new - y_bar) ** 2)

        #print("Straight line coefficients and R^2", m, c, R2)
        # Calculate the residuals (the error in the data, according to the model)
        resid = y_new - y_model
        # Chi-squared (estimates the error in data)
        chi2 = sum((resid / y_model) ** 2)
        # Reduced chi-squared (measures the goodness-of-fit)
        chi2_red = chi2 / dof
        # Standard deviation of the error
        std_err = np.sqrt(sum(resid ** 2) / dof)
        print(f'R² = {R2:.2f}', f'std_err = {std_err:.2f}')
        # Confidence interval
        ci = t_critical * std_err * np.sqrt(1 / n + (x_new - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        # Prediction Interval
        pi = t_critical * std_err * np.sqrt(1 + 1 / n + (x_new - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))

        return m, n, R2, p, y_model, model, c, t_critical, resid, chi2_red, std_err, ci, pi, x_new, y_new

    def phase_matched_filter(self, type, phaseMacthmodel, distance, filter_parameter=2):

        distance = distance/1000
        #reference_c es un scipy.interpolate.interp1d
        tr_process = self.tr.copy()
        signal = tr_process.data
        dt = tr_process.stats.delta
        tshift = (len(signal) * dt) / 2  # We will shift the collapsed Rayleigh waves to the center of the time series

        fft = np.fft.rfft(signal)
        fft_bins = np.fft.rfftfreq(len(signal), d=dt)
        freq_ref, reference_disp_phase_vel = self.__get_reference_disp(type, phaseMacthmodel, fft_bins)

        # Compute and apply the phase correction and time shifts, return to the time domain 2*pi*[distance/phase_Vel]
        #phase_correction = np.exp(1j * 2 * np.pi * fft_bins * np.divide(distance, reference_disp_phase_vel(fft_bins), out=np.zeros_like(fft_bins),
        #                                           where=reference_disp_phase_vel(fft_bins) != 0))

        phase_correction = np.exp(1j * 2 * np.pi * fft_bins * np.divide(distance, reference_disp_phase_vel))
        time_shift = np.exp(-1j*tshift*2*np.pi*fft_bins)
        frequency_domain_compressed_signal = fft * phase_correction * time_shift
        time_domain_compressed_signal = np.fft.irfft(frequency_domain_compressed_signal)
        # Apply a Gaussian window in the time domain
        sigma = filter_parameter / dt  # standard deviation in seconds divided by dt gives us samples
        gauss_window = np.exp(-0.25 * ((np.arange(0, len(time_domain_compressed_signal), 1) - len(time_domain_compressed_signal) / 2)) ** 2
                               / (sigma ** 2))
        windowed_compressed_signal = time_domain_compressed_signal * gauss_window
        windowed_compressed_signal[np.abs(windowed_compressed_signal) < 1e-16] = 0
        #self.__plot_phase_match(tr_process.times(), gauss_window, time_domain_compressed_signal, windowed_compressed_signal)
        # FFT to the frequency domain, disperse the signal and back to the time domain
        signal_freq = np.fft.rfft(windowed_compressed_signal)
        fft_bins = np.fft.rfftfreq(len(signal), d=dt)
        # Estimate phase corrections to unwrap the filtered signal
        # phase_correction_unwrap = np.exp(-1j * distance * np.divide(2 * np.pi * fft_bins, reference_disp_phase_vel(fft_bins), out=np.zeros_like(fft_bins),
        #                                                       where=reference_disp_phase_vel(fft_bins) != 0))

        phase_correction_unwrap = np.exp(-1j * 2 * np.pi * fft_bins * np.divide(distance, reference_disp_phase_vel))
        #phase_correction_unwrap = np.exp(-1j * distance * np.divide(2 * np.pi * fft_bins, reference_disp_phase_vel))
        time_shift_unwrap = np.exp(1j*tshift*2*np.pi*fft_bins)
        frequency_domain_uncompressed_signal = signal_freq*phase_correction_unwrap*time_shift_unwrap
        time_domain_uncompressed_signal = np.real(np.fft.irfft(frequency_domain_uncompressed_signal))
        tr_process.data = time_domain_uncompressed_signal
        return tr_process


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

        elif norm_method == 'running avarage':

            # lwin = int(self.tr.stats.sampling_rate * norm_win)
            # st = 0  # starting point
            # N = lwin  # ending point
            #
            # while N < self.tr.stats.npts:
            #     win = self.tr.data[st:N]
            #     w = np.sum(np.abs(win)) / (2. * lwin + 1)
            #
            #     # weight center of window
            #     if w > 0.0:
            #         self.tr.data[int(st + lwin / 2)] /= w
            #
            #
            #     # shift window
            #     st += 1
            #     N += 1

            try:
                fs = self.tr.stats.sampling_rate
                norm_win = int(norm_win*fs)
                window = np.ones(norm_win)/norm_win
                self.tr.data = self.tr.data/np.convolve(np.abs(self.tr.data), window, mode='same')
                self.tr.taper(type="blackman", max_percentage=0.05)
            except:
                print("Cannot compute time normalization at", self.tr.id)

        # if norm_method == 'running avarage' or "clipping_iter" or 'clipping':
        #     self.tr.taper(type="blackman", max_percentage=0.05)

        elif norm_method == "1 bit":
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


    def whiten_new(self, freq_width=0.02):

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

        # Calling Cython / numba version
        # data_f_whiten = whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)

        # with no pre-compilation
        for j in index:

            den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
            if den != 0: # it can be 0 because rfft is padding to 0 data_f
                # den = np.mean(np.abs(data_f[j:j + 2 * half_width]))
                data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den

        # Taper (optional) and remove mean diffs in edges of the frequency domain

        wf = (np.cos(np.linspace(np.pi / 2, np.pi, half_width)) ** 2)
        mean_1 = np.mean(np.abs(data_f[0:half_width]))
        mean_2 = np.mean(np.abs(data_f[(N_rfft - half_width):]))
        wf_flip = np.flip(wf)

        if mean_1 != 0:
            data_f_whiten[0:half_width] = (data_f[0:half_width]/mean_1) * wf   # First part of spectrum
        if mean_2 != 0:
            data_f_whiten[(N_rfft - half_width):] = (data_f[(N_rfft - half_width):]/mean_2) * wf_flip # end of spectrum

        try:
            data = np.fft.irfft(data_f_whiten)
            data = data[0:N]
        except:
            print("whitenning cannot be done")


        self.tr.data = data

# def whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
#     return __whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)
#
# def __whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
#     for j in index:
#         den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
#         if den != 0:
#             data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den
#     return data_f_whiten
#
# __whiten_aux = jit(nopython=True, parallel=True)(__whiten_aux)

    # @jit(nopython=True, parallel=True)
    # def whiten_aux(self, data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos):
    #      for j in index:
    #          den = np.sum(np.abs(data_f[j:j + 2 * half_width])) / avarage_window_width
    #          data_f_whiten[j + half_width_pos] = data_f[j + half_width_pos] / den
    #      return data_f_whiten

class noise_processing_horizontals:

    def __init__(self, tr_N, tr_E):
        self.tr_N = tr_N
        self.tr_E = tr_E

    def normalize(self, clip_factor=6, clip_weight=10, norm_win=None, norm_method='ramn'):

        if norm_method == 'clipping':
            lim_N = clip_factor * np.std(self.tr_N.data)
            lim_E = clip_factor * np.std(self.tr_N.data)
            self.tr_N.data[self.tr_N.data > lim_N] = lim_N
            self.tr_E.data[self.tr_E.data > lim_E] = lim_E
            self.tr_N.data[self.tr_N.data < -lim_N] = -lim_N
            self.tr_E.data[self.tr_N.data < -lim_E] = -lim_N

        # elif norm_method == "clipping_iter":
        #     lim_N = clip_factor * np.std(np.abs(self.tr_N.data))
        #
        #     # as long as still values left above the waterlevel, clip_weight
        #     while self.tr.data[np.abs(self.tr.data) > lim_N] != []:
        #         self.tr_N.data[self.tr.data > lim] /= clip_weight
        #         self.tr_E.data[self.tr.data > lim] /= clip_weight
        #
        #         self.tr_N.data[self.tr.data < -lim] /= clip_weight
        #         self.tr_E.data[self.tr.data < -lim] /= clip_weight

        # modified to compare maximum of both means
        if norm_method == 'ramn':
            # lwin = int(self.tr_N.stats.sampling_rate * norm_win)
            # st = 0  # starting point
            # N = lwin  # ending point
            #
            # while N < self.tr_N.stats.npts and N < self.tr_E.stats.npts:
            #     win_N = self.tr_N.data[st:N]
            #     win_E = self.tr_E.data[st:N]
            #
            #     w_N = np.sum(np.abs(win_N)) / (2. * lwin + 1)
            #     w_E = np.sum(np.abs(win_E)) / (2. * lwin + 1)
            #     max_value = max(w_N, w_E)
            #     # weight center of window
            #     if max_value > 0.0:
            #         self.tr_N.data[int(st + lwin / 2)] /= max_value
            #         self.tr_E.data[int(st + lwin / 2)] /= max_value
            #
            #     # shift window
            #     st += 1
            #     N += 1
            try:
                fs = self.tr_N.stats.sampling_rate
                norm_win = int(norm_win*fs)
                window = np.ones(norm_win)/norm_win
                norm_data_N = np.convolve(np.abs(self.tr_N.data), window, mode='same')
                norm_data_E = np.convolve(np.abs(self.tr_E.data), window, mode='same')
                norm_data = np.maximum(norm_data_N, norm_data_E)
                self.tr_N.data = self.tr_N.data/norm_data
                self.tr_E.data = self.tr_E.data/norm_data
                self.tr_N.taper(type="blackman", max_percentage=0.05)
                self.tr_E.taper(type="blackman", max_percentage=0.05)
            except:
                print("Cannot compute time normalization at", self.tr_N.id, self.tr_E.id)



            # taper edges
            #taper = self.get_window(self.tr_N.stats.npts)
            #self.tr_N.data *= taper
            #taper = self.get_window(self.tr_E.stats.npts)
            #self.tr_E.data *= taper

            if norm_method == "1 bit":
                self.tr_N.data = np.sign(self.tr_N.data)
                self.tr_E.data = np.float32(self.tr_E.data)

            # if norm_method == 'running avarage' or "clipping_iter" or 'clipping':
            #     self.tr_N.taper(type="blackman", max_percentage=0.05)
            #     self.tr_E.taper(type="blackman", max_percentage=0.05)





    def get_window(self, N, alpha=0.2):

        window = np.ones(N)
        x = np.linspace(-1., 1., N)
        ind1 = (abs(x) > 1 - alpha) * (x < 0)
        ind2 = (abs(x) > 1 - alpha) * (x > 0)
        window[ind1] = 0.5 * (1 - np.cos(np.pi * (x[ind1] + 1) / alpha))
        window[ind2] = 0.5 * (1 - np.cos(np.pi * (x[ind2] - 1) / alpha))
        return window


    def whiten_new(self, freq_width=0.02):

        """"
        freq_width: Frequency smoothing windows [Hz] / both sides
        taper_edge: taper with cosine window  the low frequencies

        return: whithened trace (Phase is not modified)
        """""

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

        # calling cython version faster
        #data_f_whiten_N,data_f_whiten_E = whiten_aux_horizontals(data_f_N, data_f_whiten_N, data_f_E,
        #                        data_f_whiten_E, index, half_width, avarage_window_width, half_width_pos)

        # whitout cython
        for j in index:
            den = 0.5*(np.sum(np.abs(data_f_N[j:j + 2 * half_width]) + np.abs(data_f_E[j:j + 2 * half_width]))/avarage_window_width)
            if den != 0:
                data_f_whiten_N[j + half_width_pos] = data_f_whiten_N[j + half_width_pos] / den
                data_f_whiten_E[j + half_width_pos] = data_f_whiten_E[j + half_width_pos] / den

        # Taper (optional) and remove mean diffs in edges of the frequency domain

        wf = (np.cos(np.linspace(np.pi / 2, np.pi, half_width)) ** 2)
        wf_flip = np.flip(wf)
        mean_1_N = np.mean(np.abs(data_f_N[0:half_width]))
        mean_1_E = np.mean(np.abs(data_f_N[0:half_width]))
        mean_2_N = np.mean(np.abs(data_f_N[(N_rfft - half_width):]))
        mean_2_E = np.mean(np.abs(data_f_E[(N_rfft - half_width):]))

        if mean_1_E !=0:
            data_f_whiten_E[0:half_width] = (data_f_E[0:half_width] / mean_1_E) * wf
        if mean_1_N != 0:
            data_f_whiten_N[0:half_width] = (data_f_N[0:half_width] / mean_1_N) * wf

        if mean_2_E != 0:
            data_f_whiten_E[(N_rfft - half_width):] = (data_f_E[(N_rfft - half_width):] / mean_2_E) * wf_flip

        if mean_2_N != 0:
            data_f_whiten_N[(N_rfft - half_width):] = (data_f_N[(N_rfft - half_width):] / mean_2_N) * wf_flip

        data_N = np.fft.irfft(data_f_whiten_N)
        data_E = np.fft.irfft(data_f_whiten_E)
        data_N = data_N[0:N]
        data_E = data_E[0:N]

        self.tr_N.data = data_N
        self.tr_E.data = data_E

    # @jit(nopython=True, parallel=True)
    # def whiten_aux_horizontals(self, data_f_N, data_f_whiten_N, data_f_E, data_f_whiten_E, index, half_width,
    #                            avarage_window_width, half_width_pos):
    #     for j in index:
    #
    #         den_N = np.sum(np.abs(data_f_N[j:j + 2 * half_width])) / avarage_window_width
    #         den_E = np.sum(np.abs(data_f_E[j:j + 2 * half_width])) / avarage_window_width
    #         mean = np.mean(den_N, den_E)
    #         # den = np.mean(np.abs(data_f[j:j + 2 * half_width]))
    #         data_f_whiten_N[j + half_width_pos] = data_f_whiten_N[j + half_width_pos] / mean
    #         data_f_whiten_E[j + half_width_pos] = data_f_whiten_E[j + half_width_pos] / mean

        return data_f_whiten_N, data_f_whiten_E

    def rotate2NE(self, baz):

        x0 = self.tr_E.data
        y0 = self.tr_N.data
        rad = math.pi/180
        try:
            x1 = x0*math.cos(baz*rad) - y0*math.sin(baz*rad)
            y1 = x0*math.sin(baz*rad) + y0*math.cos(baz*rad)
            self.tr_N.data = y1
            self.tr_E.data = x1
        except:
            pass


class ManageEGF:

    def filter_project_keys(self, files_list, **kwargs):

        # filter dict by python wilcards remind

        # * --> .+
        # ? --> .

        net = kwargs.pop('net', '.+')
        station = kwargs.pop('station', '.+')
        channel = kwargs.pop('channel', '.+')
        if net == '':
            net = '.+'
        if station == '':
            station = '.+'
        if channel == '':
            channel = '.+'


        # filter for regular expresions
        event = [net, station, channel]
        filtered_list = []
        for file in files_list:
            key = os.path.basename(file)
            name_list = key.split('.')
            net = name_list[0]
            sta = name_list[1]
            channel = name_list[2]
            if re.search(event[0], net) and re.search(event[1], sta) and re.search(event[2], channel):
                filtered_list.append(file)

        return filtered_list
