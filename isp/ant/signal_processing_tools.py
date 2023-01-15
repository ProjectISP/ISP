import numpy as np
import math
#from numba import jit
from scipy import stats
# Cython code
#from isp.cython_code.whiten import whiten_aux, whiten_aux_horizontals

class noise_processing:

    def __init__(self, tr):
        self.tr = tr

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

    def __get_reference_disp(self, type, phaseMacthmodel, fft_bins):
        import pandas as pd
        import os
        from isp import DISP_REF_CURVES
        from scipy import interpolate

        if phaseMacthmodel == "ak-135f":

            path = os.path.join(DISP_REF_CURVES, "ak135_earth_velocity.txt")
            df = pd.read_csv(path)
            freq_ref = 1/(df['period'].to_numpy())
            if type == "Rayleigh":
                vel = df['phase_velocity_rayleigh'].to_numpy()
            if type == "Love":
                vel = df['phase_velocity_love'].to_numpy()

        elif phaseMacthmodel == "ak-135f (Ocean-shallow waters)":

            path = os.path.join(DISP_REF_CURVES, "ak135_earth_ocean_shallow_velocity.txt")
            df = pd.read_csv(path)
            freq_ref = 1/(df['period'].to_numpy())
            if type == "Rayleigh":
                vel = df['phase_velocity_rayleigh'].to_numpy()
            if type == "Love":
                vel = df['phase_velocity_love'].to_numpy()

        elif phaseMacthmodel == "ak-135f (Ocean-intermediate waters)":

            path = os.path.join(DISP_REF_CURVES, "ak135_earth_ocean_intermediate_velocity.txt")
            df = pd.read_csv(path)
            freq_ref = 1/(df['period'].to_numpy())
            if type == "Rayleigh":
                vel = df['phase_velocity_rayleigh'].to_numpy()
            if type == "Love":
                vel = df['phase_velocity_love'].to_numpy()

        elif phaseMacthmodel == "ak-135f (Ocean-deep waters)":

            path = os.path.join(DISP_REF_CURVES, "ak135_earth_ocean_deep_velocity.txt")
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
        return vel2

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

        print(f'R² = {R2:.2f}')
        #print("Straight line coefficients and R^2", m, c, R2)
        # Calculate the residuals (the error in the data, according to the model)
        resid = y_new - y_model
        # Chi-squared (estimates the error in data)
        chi2 = sum((resid / y_model) ** 2)
        # Reduced chi-squared (measures the goodness-of-fit)
        chi2_red = chi2 / dof
        # Standard deviation of the error
        std_err = np.sqrt(sum(resid ** 2) / dof)

        return m, n, R2, p, y_model, model, c, t_critical, resid, chi2_red, std_err, x_new, y_new

    def phase_matched_filter(self, type, phaseMacthmodel, distance, filter_parameter = 2):

        distance = distance/1000
        #reference_c es un scipy.interpolate.interp1d
        tr_process = self.tr.copy()
        signal = tr_process.data
        dt = tr_process.stats.delta
        tshift = (len(signal) * dt) / 2  # We will shift the collapsed Rayleigh waves to the center of the time series

        fft = np.fft.rfft(signal)
        fft_bins = np.fft.rfftfreq(len(signal), d=dt)
        reference_disp_phase_vel = self.__get_reference_disp(type, phaseMacthmodel, fft_bins)

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
            #taper = self.get_window(self.tr.stats.npts)
            #self.tr.data *= taper
            # ensure no glitches in the extremes due to previous pre-filt
            self.tr.taper(type="blackman", max_percentage=0.025)

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

        # Calling Cython version
        #data_f_whiten = whiten_aux(data_f, data_f_whiten, index, half_width, avarage_window_width, half_width_pos)

        # with no pre-compilation
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
        try:
            data = np.fft.irfft(data_f_whiten)
            data = data[0:N]
        except:
            print("whitenning cannot be done")


        self.tr.data = data

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
            #taper = self.get_window(self.tr_N.stats.npts)
            #self.tr_N.data *= taper
            #taper = self.get_window(self.tr_E.stats.npts)
            #self.tr_E.data *= taper

            self.tr_N.taper(type="blackman", max_percentage=0.025)
            self.tr_E.taper(type="blackman", max_percentage=0.025)




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
        x1 = x0*math.cos(baz*rad) - y0*math.sin(baz*rad)
        y1 = x0*math.sin(baz*rad) + y0*math.con(baz*rad)

        self.tr_N.data = y1
        self.tr_E.data = x1

