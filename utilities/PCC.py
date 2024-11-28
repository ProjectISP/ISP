import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


#===========================================================
# Routine by Luis-Fabian Bonilla (IPGP & IFSTTAR), Jan 2020.
#===========================================================

# Tapering with a Hanning window

def taper(x,p):
    if p <= 0.0:
        return x
    else:
        f0 = 0.5
        f1 = 0.5
        n  = len(x)
        nw = int(p*n)

        if nw > 0:
            ow = np.pi/nw

            w = np.ones( n )
            for i in range( nw ):
                w[i] = f0 - f1 * np.cos(ow*i)

            for i in range( n-nw,n ):
                w[i] = 1.0 - w[i-n+nw]

            return x * w
        elif nw == 0:
            return x

# Bitwise version

def next_power_of_2(n):
    """
    Return next power of 2 greater than or equal to n
    """
    return 2**(n-1).bit_length()

# PCC2 from Ventosa el al. (2019)

def pcc2(x1, x2, dt, lag0, lagu):
    # Preprocessing

    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)
    x1 = taper(x1, 0.05)
    x2 = taper(x2, 0.05)
    N  = len(x1)
    Nz = next_power_of_2( 2*N )

    # Analytic signal and normalization

    xa1 = scipy.signal.hilbert(x1)
    xa2 = scipy.signal.hilbert(x2)
    xa1 = xa1 / np.abs(xa1)
    xa2 = xa2 / np.abs(xa2)

    # Padding zeros

    xa1 = np.append(xa1, np.zeros((Nz-N), dtype=np.complex64))
    xa2 = np.append(xa2, np.zeros((Nz-N), dtype=np.complex64))

    # FFT, correlation and IFFT

    xa1 = np.fft.fft(xa1)
    xa2 = np.fft.fft(xa2)
    amp = xa1 * np.conj(xa2)
    pcc = np.real( np.fft.ifft(amp) ) / N
    pcc = np.fft.ifftshift(pcc)
    tt  = Nz//2 * dt
    t   = np.arange(-tt, tt, dt)

    return t[(t >= lag0) & (t <= lagu)], pcc[(t >= lag0) & (t <= lagu)]

# PCC2 for autocorrelation

def apcc2(x1, dt, lag0, lagu):
    # Preprocessing

    x1 = x1 - np.mean(x1)
    x1 = taper(x1, 0.05)
    N  = len(x1)
    Nz = next_power_of_2( 2*N )

    # Analytic signal and normalization

    xa1 = scipy.signal.hilbert(x1)
    xa1 = xa1 / np.abs(xa1)

    # Padding zeros

    xa1 = np.append(xa1, np.zeros((Nz-N), dtype=np.complex_))

    # FFT, correlation and IFFT

    xa1 = np.fft.fft(xa1)
    amp = xa1 * np.conj(xa1)
    pcc = np.real( np.fft.ifft(amp) ) / N
    pcc = np.fft.ifftshift(pcc)
    tt  = Nz//2 * dt
    t  = np.arange(-tt, tt, dt)

    return t[(t >= lag0) & (t <= lagu)], pcc[(t >= lag0) & (t <= lagu)]

# Provided pcc2 function with the suggested optimization



def next_power_of_2mod(x):
    return 1 << (x - 1).bit_length()

def normalize_analytic_signal(X):
    magnitude = np.abs(X)
    magnitude[magnitude == 0] = 1  # Prevent division by zero
    return X / magnitude

def pcc3(x1, x2, dt, lag0, lagu):
    # Preprocessing
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)
    x1 = taper(x1, 0.05)
    x2 = taper(x2, 0.05)
    N = len(x1)
    Nz = next_power_of_2(2 * N)

    # Compute the Fourier transform of the signals
    xa1 = np.fft.fft(x1, n=Nz)
    xa2 = np.fft.fft(x2, n=Nz)

    # Set the negative frequencies to zero to get the analytic signal
    xa1[int(Nz/2)+1:] = 0
    xa2[int(Nz/2)+1:] = 0
    # if Nz % 2 == 0:  # Even number of samples
    #     xa1[int(Nz / 2) + 1:] = 0
    #     xa2[int(Nz / 2) + 1:] = 0
    # else:  # Odd number of samples
    #     xa1[int((Nz + 1) / 2):] = 0
    #     xa2[int((Nz + 1) / 2):] = 0

    # Normalize the amplitude of the analytic signal
    xa1 = normalize_analytic_signal(xa1)
    xa2 = normalize_analytic_signal(xa2)

    # Compute the cross-correlation in the frequency domain
    amp = xa1 * np.conj(xa2)
    pcc = np.real(np.fft.ifft(amp)) / N
    pcc = np.fft.ifftshift(pcc)
    tt = Nz // 2 * dt
    t = np.arange(-tt, tt, dt)

    return t[(t >= lag0) & (t <= lagu)], pcc[(t >= lag0) & (t <= lagu)]

def create_gaussian_pulse(length, std, delay, dt):
    t = np.arange(0, length, dt)
    pulse = signal.gaussian(len(t), std=std)
    delayed_pulse = np.roll(pulse, int(delay / dt))
    return t, pulse, delayed_pulse

if __name__ == "__main__":

    # Example parameters
    length = 10  # seconds
    std = 1  # standard deviation of the Gaussian
    delay = 2  # seconds
    dt = 0.01  # time step

    t, pulse, delayed_pulse = create_gaussian_pulse(length, std, delay, dt)

    # Testing the optimized pcc2 function
    lag0 = -5  # start of the lag range
    lagu = 5  # end of the lag range

    lags, pcc = pcc3(pulse, delayed_pulse, dt, lag0, lagu)

    # Plotting the results
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Upper plot: Gaussian pulses
    axs[0].plot(t, pulse, label='Original Pulse')
    axs[0].plot(t, delayed_pulse, label='Delayed Pulse')
    axs[0].legend()
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Gaussian Pulses')

    # Lower plot: Phase Cross-Correlation
    axs[1].plot(lags, pcc)
    axs[1].set_xlabel('Lag [s]')
    axs[1].set_ylabel('PCC')
    axs[1].set_title('Phase Cross-Correlation')

    plt.tight_layout()
    plt.show()


    # frequency domain shifting
    # Example signal
    # N = 10
    # x = np.random.rand(N)
    #
    # # FFT and IFFT
    # X = np.fft.fft(x)
    # x_reconstructed = np.fft.ifft(X)
    #
    # # Cross-correlation in frequency domain (example with same signal for simplicity)
    # cross_corr_freq = X * np.conj(X)
    # cross_corr_time = np.fft.ifft(cross_corr_freq)
    #
    # # If we directly plot cross_corr_time
    # plt.figure()
    # plt.plot(cross_corr_time.real, label='Without IFFT Shift')
    #
    # # After applying ifftshift
    # cross_corr_time_shifted = np.fft.ifftshift(cross_corr_time)
    # plt.plot(cross_corr_time_shifted.real, label='With IFFT Shift')
    # plt.legend()
    # plt.show()