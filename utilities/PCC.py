import scipy.signal
import numpy as np

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

    xa1 = np.append(xa1, np.zeros((Nz-N), dtype=np.complex_))
    xa2 = np.append(xa2, np.zeros((Nz-N), dtype=np.complex_))

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
    t   = np.arange(-tt, tt, dt)

    return t[(t >= lag0) & (t <= lagu)], pcc[(t >= lag0) & (t <= lagu)]