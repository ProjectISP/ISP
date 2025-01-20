import numpy as np
import matplotlib.pyplot as plt
import math
# Generate Gaussian pulses
def gaussian_pulse(t, center, width):
    return np.exp(-((t - center) ** 2) / (2 * width ** 2))

def next_power_of_2(n):
    """
    Return next power of 2 greater than or equal to n
    """
    return 2**(n-1).bit_length()

# Time axis
dt = 10
n_points = 4500
t = np.linspace(0, dt, n_points)

# Create two Gaussian pulses
center1, width1 = 1, 0.25
center2, width2 = 8, 0.25
signal1 = gaussian_pulse(t, center1, width1)
signal2 = gaussian_pulse(t, center2, width2)

# Pad signals with zeros to the next power of two for FFT
next_pow2 = 2 ** int(np.ceil(np.log2(2 * n_points)))
next_pow2_eq = 2 ** math.ceil(math.log2(n_points))
next_pow2_eq_new = next_power_of_2( 2*n_points)
padded_signal1 = np.pad(signal1, (0, next_pow2 - len(signal1)))
padded_signal2 = np.pad(signal2, (0, next_pow2 - len(signal2)))

# Compute FFTs
fft_signal1 = np.fft.rfft(padded_signal1)
fft_signal2 = np.fft.rfft(padded_signal2)

# Compute cross-correlation via multiplication in the frequency domain
cross_corr_freq = fft_signal1 * np.conj(fft_signal2)
cross_corr_full_time = np.fft.irfft(cross_corr_freq).real
cross_corr_full_time = np.fft.ifftshift(cross_corr_full_time)
# Crop the cross-correlation result in the time domain
output_length = len(signal1) + len(signal2) - 1
start_idx = (len(cross_corr_full_time) - output_length) // 2
end_idx = start_idx + output_length
cross_corr_time = cross_corr_full_time[start_idx:end_idx]

# Adjust time axis for the cropped result
corr_time = np.linspace(-1*dt, dt, output_length)

# Find zero lag position
zero_lag_idx = np.argmax(cross_corr_time)
zero_lag_time = corr_time[zero_lag_idx]

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, signal1, label="Signal 1")
plt.plot(t, signal2, label="Signal 2")
plt.title("Original Signals")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(corr_time, cross_corr_time, label="Cross-Correlation")
plt.title("Cross-Correlation")
plt.axvline(x=zero_lag_time, color='r', linestyle='--', label=f"Zero Lag ({zero_lag_time:.2f})")
plt.legend()
plt.grid()



plt.tight_layout()
plt.show()