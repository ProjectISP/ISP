import math
import os
from obspy import read, Stream
from scipy import signal
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="RuntimeWarning: divide by zero encountered in log")

class RemoveComplianceTilt:

    def __init__(self, N, E, Z, H):

        """
        N, E, Z, H -> North, East, Vertical and Hydrophon traces

        :param obs_file_path: The file path of pick observations.
        """

        self.N = N
        self.E = E
        self.Z = Z
        self.H = H
        self.Enew = self.E.copy()
        self.Nnew = self.N.copy()
        self.Znew = self.Z.copy()
        self.fs = self.Z.stats.sampling_rate
        self.fn = self.fs/2
        self.transfer_info = {}

    @staticmethod
    def power_log(x):
        n = math.ceil(math.log(x, 2))
        return n

    @staticmethod
    def find_nearest(a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return a.flat[idx], idx

    def transfer_function(self, channels, nfft = 15, noverlap = 50):

        # source --> X, Y and H
        # response --> Z

        nfft = nfft * self.fs * 60  # 15 minutes in samples
        noverlap = int(nfft * (noverlap/100))

        s = channels["source"]
        r = channels["response"]
        channels = [s, r]
        maxstart = np.max([tr.stats.starttime for tr in channels])
        minend = np.min([tr.stats.endtime for tr in channels])

        s.trim(maxstart, minend)
        r.trim(maxstart, minend)
        s.detrend(type='simple')
        r.detrend(type='simple')

        f, Pss = signal.welch(s.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                           detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Prr = signal.welch(r.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                           detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Prs = signal.csd(r.data, s.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                         detrend='linear', return_onesided=True, scaling='density', axis=-1)

        coherence = Prs/(np.sqrt(Prr*Pss))
        transfer = np.conj(coherence*np.sqrt(Prr/Pss))

        #self.transfer_info["cpsd"] = Prs
        #self.transfer_info["source_power"] = Pss
        #self.transfer_info["response_power"] = Prr
        self.transfer_info["transfer"] = transfer
        self.transfer_info["coherence"] = coherence
        self.transfer_info["frequency"] = f

    def remove_noise(self, channels):

        #print("Calculating new Trace in Frequency Domain")
        s = channels["source"]
        r = channels["response"]
        s.detrend(type="simple")
        s.taper(type="Hamming", max_percentage=0.05)
        r.detrend(type="simple")
        r.taper(type="Hamming",max_percentage=0.05)

        f = self.transfer_info["frequency"]
        Thr = self.transfer_info["transfer"]

        Sf = np.fft.rfft(s.data, 2 ** self.power_log(len(s.data)))
        Rf = np.fft.rfft(r.data, 2 ** self.power_log(len(r.data)))

        ##Interpolate Thz to Hf

        freq1 = np.fft.rfftfreq(2 ** self.power_log(len(r.data)), 1/self.fs)
        set_interp = interp1d(f, Thr, kind='cubic')
        Thrf = set_interp(freq1)
        # set_interp = interp1d(f, np.abs(Thr), kind='cubic')
        #phase_angle = np.angle(Rf)
        #Rff = (np.abs(Rf) - np.abs(Thrf)*np.abs(Hf))* np.exp(1j * phase_angle)
        Rff = (Rf) - (Thrf * Sf)
        value, idx = self.find_nearest(freq1, 0.1)
        Rff[idx:] = Rf[idx:]
        Rnew_data = np.fft.irfft(Rff)

        if r.stats.channel[2] == "Z":
            self.Znew.data = Rnew_data[0:len(r.data)]
        elif r.stats.channel[2] == "N" or r.stats.channel[2] == "1" or r.stats.channel[2] == "Y":
            self.Nnew.data = Rnew_data[0:len(r.data)]
        elif r.stats.channel[2] == "E" or r.stats.channel[2] == "2" or r.stats.channel[2] == "X":
            self.Enew.data = Rnew_data[0:len(r.data)]


    def plot_coherence_transfer(self, channels,  **kwargs):

        save_fig = kwargs.pop('save_fig', False)
        path_save = kwargs.pop('path_save', os.getcwd())

        s = channels["source"]
        r = channels["response"]
        label = "Coherence "+s.id+"_"+r.id
        phase = np.angle(self.transfer_info["coherence"])
        coherence = np.abs(self.transfer_info["coherence"])
        f = self.transfer_info["frequency"]

        fig, axs = plt.subplots(2, figsize=(10, 10))
        title = 'Coherence between channels'+"."+str(s.stats.starttime.julday)+"."+str(s.stats.starttime.year)
        fig.suptitle(title, fontsize=16)
        axs[0].semilogx(self.transfer_info["frequency"], coherence, linewidth=0.75, color='steelblue', label = label)
        axs[0].grid(True, which="both", ls="-", color='grey')
        axs[0].set_xlim(f[1], f[len(f) - 1])
        axs[0].set_ylim(-0.1, 1.1)
        axs[0].set_xlabel('Frequency [Hz]')
        axs[0].set_ylabel('Coherence')
        axs[0].legend()

        axs[1].semilogx(self.transfer_info["frequency"], phase, linewidth=0.75, color='red', label = label)
        axs[1].set_xlim(f[1], f[len(f) - 1])
        axs[1].grid(True, which="both", ls="-", color='grey')
        axs[1].set_xlabel('Frequency [Hz]')
        axs[1].set_ylabel('Phase')
        if save_fig:
            name = label+"."+str(s.stats.starttime.julday)+"."+str(s.stats.starttime.year)+".png"
            path = os.path.join(path_save,name)
            plt.savefig(path, dpi=150, format = 'png')
            plt.close()
        else:
            plt.show()
            plt.close()


    def plot_transfer_function(self, channels, **kwargs):

        save_fig = kwargs.pop('save_fig', False)
        path_save = kwargs.pop('path_save', os.getcwd())

        s = channels["source"]
        r = channels["response"]
        label = "Transfer "+s.id+"-"+r.id
        transfer = np.abs(self.transfer_info["transfer"])
        f = self.transfer_info["frequency"]
        value_min, idx_min = self.find_nearest(f, 0.001)
        value_max, idx_max = self.find_nearest(f, 0.1)
        transfer = transfer[idx_min: idx_max]
        f = f[idx_min: idx_max]
        fig, axs = plt.subplots(figsize=(10, 10))
        fig.suptitle('Transfer Function', fontsize=16)
        axs.semilogx(f, transfer, linewidth=0.75, color='steelblue', label = label)
        axs.grid(True, which="both", ls="-", color='grey')
        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel('Amplitude')
        axs.legend()

        if save_fig:
            name = label+"."+str(s.stats.starttime.julday)+"."+str(s.stats.starttime.year)+".Transfer"+".png"
            path = os.path.join(path_save,name)
            plt.savefig(path, dpi=150, format = 'png')
            plt.close()
        else:

            plt.show()
            plt.close()

    def plot_compare_spectrums(self, channels, nfft = 15, noverlap = 50, **kwargs):

        save_fig = kwargs.pop('save_fig', False)
        path_save = kwargs.pop('path_save', os.getcwd())

        s = channels["source"]
        r = channels["response"]
        label = r.id + "-" + s.id

        nfft = nfft * self.fs * 60  # 15 minutes in samples
        noverlap = int(nfft * (noverlap / 100))
        f, Zpow = signal.welch(self.Z.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap,
                              nfft=nfft, detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Zpownew = signal.welch(self.Znew.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                               detrend='linear', return_onesided=True, scaling='density', axis=-1)
        ##
        fig, axs = plt.subplots(figsize=(8, 8))
        fig.suptitle('Power Spectrum Comparison', fontsize=16)
        axs.semilogx(f[1:], 10*np.log(Zpow[1:]/2*np.pi*f[1:]), linewidth=0.75, color='steelblue', label=r.id)
        axs.semilogx(f[1:], 10*np.log(Zpownew[1:]/2*np.pi*f[1:]), linewidth=0.75, color='green', label=label)

        #axs.loglog(f, Zpow, linewidth=0.5, color='steelblue', label=r.id)
        #axs.loglog(f, Zpownew, linewidth=0.5, color='red', label=label)
        #axs.set_ylim(-175, -100)
        axs.set_xlim(0.001, 0.1)
        axs.grid(True, which="both", ls="-", color='grey')
        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel('Amplitude Acceleration [counts/s]^2')
        axs.legend()
        if save_fig:
            name = label+"."+str(s.stats.starttime.julday)+"."+str(s.stats.starttime.year)+".Tilt"+".png"
            path = os.path.join(path_save,name)
            plt.savefig(path, dpi=150, format = 'png')
            plt.close()
        else:

            plt.show()
            plt.close()


    def plot_compare_spectrums_full(self, channels, Ztilt, nfft = 15, noverlap = 50, **kwargs):

        save_fig = kwargs.pop('save_fig', False)
        path_save = kwargs.pop('path_save', os.getcwd())

        s = channels["source"]
        r = channels["response"]
        label = r.id + "-" + s.id

        nfft = nfft * self.fs * 60  # 15 minutes in samples
        noverlap = int(nfft * (noverlap / 100))

        f, Zpow = signal.welch(self.Z.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap,
                              nfft=nfft, detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Zpowtilt = signal.welch(Ztilt, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap,
                               nfft=nfft, detrend='linear', return_onesided=True, scaling='density', axis=-1)
        f, Zpownew = signal.welch(self.Znew.data, fs=self.fs, window='hamming', nperseg=nfft, noverlap=noverlap, nfft=nfft,
                               detrend='linear', return_onesided=True, scaling='density', axis=-1)
        ##
        fig, axs = plt.subplots(figsize=(8, 8))
        fig.suptitle('Power Spectrum Comparison', fontsize=16)
        axs.semilogx(f[1:], 10*np.log(Zpow[1:]/2*np.pi*f[1:]), linewidth=0.75, color='steelblue', label=r.id)
        axs.semilogx(f[1:], 10*np.log(Zpowtilt[1:]/2*np.pi*f[1:]), linewidth=0.75, color='green', label=r.id+"- Tilt removed")
        axs.semilogx(f[1:], 10*np.log(Zpownew[1:]/2*np.pi*f[1:]), linewidth=0.75, color='red', label=r.id+"- Compliance removed")
        axs.set_xlim(0.001, 0.1)
        axs.grid(True, which="both", ls="-", color='grey')
        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel('Amplitude Acceleration dB [(counts/s^2)^2 / Hz] ')
        axs.legend()
        if save_fig:
            name = label+"."+str(s.stats.starttime.julday)+"."+str(s.stats.starttime.year)+".Tilt+Compliance"+".png"
            path = os.path.join(path_save,name)
            plt.savefig(path, dpi=150, format = 'png')
            plt.close()
        else:

            plt.show()
            plt.close()

    # test



if __name__ == "__main__":
    
    #CWD = os.path.getcwd()
    #data_example = "./"
    #path_files_example = os.path.join(CWD, data_example)
    path_files_example = '/Users/robertocabieces/Desktop/desarrollo/denoise_OBS_data/data_compliance'
    #
    file_z = read(os.path.join(path_files_example, 'WM.OBS06..SHZ.D.2015.258'))
    file_e = read(os.path.join(path_files_example, 'WM.OBS06..SHX.D.2015.258'))
    file_n = read(os.path.join(path_files_example, 'WM.OBS06..SHY.D.2015.258'))
    file_h = read(os.path.join(path_files_example, 'WM.OBS06..SDH.D.2015.258'))
    #
    tr_z = file_z[0]
    tr_e = file_e[0]
    tr_n = file_n[0]
    tr_h = file_h[0]
    # 
    noise = RemoveComplianceTilt(tr_n, tr_e, tr_z, tr_h)
    channels = {}

    # First Tilt Noise (between horizontal components)
    # Y' = Y - Tyx*X
    channels["source"] = tr_e
    channels["response"] = tr_n
    noise.transfer_function(channels)
    noise.plot_coherence_transfer(channels)
    noise.remove_noise(channels)

    # Z' = Z- Tzx*X

    channels["source"] = tr_e
    channels["response"] = tr_z
    noise.transfer_function(channels)
    noise.plot_coherence_transfer(channels)
    noise.remove_noise(channels)

    # Second Tilt Noise (horizontal - Vertical)

    # Z'' = Z' - Tz'y'*Y'
    channels["source"] = noise.Nnew
    channels["response"] = noise.Znew

    noise.transfer_function(channels)
    noise.plot_coherence_transfer(channels)
    noise.plot_transfer_function(channels)
    noise.remove_noise(channels)
    noise.plot_compare_spectrums(channels)

    # Third Compliance (Hydrophone - Vertical)
    # Z''' = Z'' - Tz''h*H

    channels["source"] = tr_h
    channels["response"] = noise.Znew
    Ztilt = noise.Znew.copy()
    noise.transfer_function(channels)
    noise.plot_transfer_function(channels)
    noise.plot_coherence_transfer(channels)
    noise.remove_noise(channels)
    noise.plot_compare_spectrums_full(channels, Ztilt)
    tr_znew = noise.Znew
    tr_znew.stats.channel = 'SCZ'
    st = Stream([tr_z, tr_znew])
    st.detrend(type='simple')
    st.taper(max_percentage=0.05, type="blackman")
    st.filter(type='bandpass', freqmin=0.001, freqmax=0.05, corners=4, zerophase=False)
    st.normalize()
    st.plot()

