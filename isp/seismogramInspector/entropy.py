import numpy as np
from scipy.signal import periodogram, welch


def spectral_entropy(x, sf, method='fft', nperseg=None, normalize=False):
    """

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    sf : float
        Sampling frequency
    method : str
        Spectral estimation method ::

        'fft' : Fourier Transform (via scipy.signal.periodogram)
        'welch' : Welch periodogram (via scipy.signal.welch)

    nperseg : str or int
        Length of each FFT segment for Welch method.
        If None, uses scipy default of 256 samples.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.

    Returns
    -------
    se : float
        Spectral Entropy

    Notes
    -----
    Spectral Entropy is defined to be the Shannon Entropy of the Power
    Spectral Density (PSD) of the data:

    .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} PSD(f) log_2[PSD(f)]

    Where :math:`PSD` is the normalised PSD, and :math:`f_s` is the sampling
    frequency.

    References
    ----------
    .. [1] Inouye, T. et al. (1991). Quantification of EEG irregularity by
       use of the entropy of the power spectrum. Electroencephalography
       and clinical neurophysiology, 79(3), 204-210.

    Examples
    --------
    1. Spectral entropy of a pure sine using FFT

    #    >>> from entropy import spectral_entropy
    #    >>> import numpy as np
    #    >>> sf, f, dur = 100, 1, 4
    #    >>> N = sf * duration # Total number of discrete samples
    #    >>> t = np.arange(N) / sf # Time vector
    #    >>> x = np.sin(2 * np.pi * f * t)
    #    >>> print(np.round(spectral_entropy(x, sf, method='fft'), 2)
    #        0.0

    2. Spectral entropy of a random signal using Welch's method

    #    >>> from entropy import spectral_entropy
    #    >>> import numpy as np
    #    >>> np.random.seed(42)
    #    >>> x = np.random.rand(3000)
    #    >>> print(spectral_entropy(x, sf=100, method='welch'))
            9.939

    3. Normalized spectral entropy

    #    >>> print(spectral_entropy(x, sf=100, method='welch', normalize=True))
            0.995
    """
    x = np.array(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = periodogram(x, sf)
    elif method == 'welch':
        _, psd = welch(x, sf, nperseg=nperseg)
    psd_norm = np.divide(psd, psd.sum())
    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    if normalize:
        se /= np.log2(psd_norm.size)
    return se


