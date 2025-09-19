from __future__ import annotations
import numpy as np
from obspy.core.trace import Trace
from isp.cython_code import rec_filter, lib_rec_hos, lib_rec_rms
# from isp.cython_code import lib_rosenberger  # not used for 1C

class CFMB:

    """

    Multiband filtering + CF (envelope/kurtosis) for a single ObsPy Trace
    Clean multi-band CF for a single ObsPy Trace.
    Returns:

    """

    # ----------------- public API -----------------
    @staticmethod
    def compute_tf(
        tr: Trace,
        *,
        CN_HP: np.ndarray | None = None,
        CN_LP: np.ndarray | None = None,
        fmin: float = 0.5,
        fmax: float = 10.0,
        filter_norm: np.ndarray | None = None,
        filter_npoles: int = 2,
        var_w: bool = False,
        CF_type: str = "envelope",
        CF_decay_win: float = 0.5,      # seconds
        hos_order: int = 4,
        hos_sigma: float | None = None, # None => -1.0 (always update var)
        rec_memory=None,
        apply_taper: bool = True):

        assert isinstance(tr, Trace), "Input must be an ObsPy Trace"
        delta = float(tr.stats.delta)
        freqs = np.geomspace(fmin, min(fmax, 0.98 * (0.5 / delta)), 20)
        y = np.asarray(tr.data, dtype=np.float64)
        tr.detrend("demean").detrend("linear")
        Tn = 1.0 / freqs
        Nb = len(freqs)
        CF_decay_nsmps = CF_decay_win / delta

        # compute coefficients/norm if not provided (same helpers as your code)
        if CN_HP is None or CN_LP is None:
            CN_HP, CN_LP = CFMB.rec_filter_coeff(freqs, delta)
        if filter_norm is None:
            filter_norm = CFMB.rec_filter_norm(freqs, delta, CN_HP, CN_LP, npoles=filter_npoles)

        if hos_sigma is None:
            hos_sigma = -1.0

        # ---- single component branch (preserved) ----
        YN1 = np.zeros((Nb, len(y)), float)
        CF1 = np.zeros((Nb, len(y)), float)

        tr_id = getattr(tr, "id", "NET.STA.LOC.CHAN")
        wave_type = "P"  # preserved default semantic; not used here but kept
        for n in range(Nb):
            rmem = rec_memory[(tr_id, wave_type)][n] if rec_memory is not None else None

            # recursive_filter may return tuple in your build; take [0] safely.
            band = CFMB._rf(y, CN_HP[n], CN_LP[n], filter_npoles, rmem)
            YN1[n] = band / (filter_norm[n] if filter_norm[n] != 0 else 1.0)

            # decay constant
            if var_w and CF_type == 'envelope':
                CF_decay_nsmps_mb = (Tn[n] / delta) * CF_decay_nsmps
            else:
                CF_decay_nsmps_mb = CF_decay_nsmps
            CF_decay_constant = 1.0 / CF_decay_nsmps_mb

            # CF per band (preserved)
            if CF_type == 'envelope':
                CF1[n] = CFMB._rms(YN1[n], CF_decay_constant)
            elif CF_type == 'kurtosis':
                CF1[n] = CFMB._hos(
                    YN1[n],
                    C_WIN=CF_decay_constant,
                    order=hos_order,
                    sigma_min=hos_sigma
                )
            else:
                raise ValueError("CF_type must be 'envelope' or 'kurtosis'")

        cf_stack = (CF1).max(axis=0)

        if apply_taper:
            cf_stack = cf_stack * CFMB._blackman_taper(len(cf_stack), p=0.05)

        return YN1, CF1, cf_stack, Tn, Nb, freqs

    # ----------------- helpers you already had (kept) -----------------
    @staticmethod
    def _cosine_taper(npts: int, p: float = 0.05) -> np.ndarray:
        if not (0.0 < p < 0.5):
            return np.ones(npts)
        n_taper = int(np.floor(p * npts))
        if n_taper == 0:
            return np.ones(npts)
        win = np.ones(npts)
        taper = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        win[:n_taper] = taper
        win[-n_taper:] = taper[::-1]
        return win

    @staticmethod
    def _blackman_taper(npts: int, p: float = 0.05) -> np.ndarray:
        """
        Return a Blackman taper window applied to the beginning and end of a signal.

        Parameters
        ----------
        npts : int
            Number of samples in the signal.
        p : float, optional
            Fraction of the signal length to taper on each side (0 < p < 0.5).
            Default is 0.05.

        Returns
        -------
        win : ndarray
            Window array of length npts with Blackman tapers at both ends.
        """
        if not (0.0 < p < 0.5):
            return np.ones(npts)

        n_taper = int(np.floor(p * npts))
        if n_taper == 0:
            return np.ones(npts)

        win = np.ones(npts)

        # Full Blackman window of size 2*n_taper
        full_blackman = np.blackman(2 * n_taper)

        # Take the first half for rising taper, second half for falling taper
        taper_start = full_blackman[:n_taper]
        taper_end = full_blackman[-n_taper:]

        win[:n_taper] = taper_start
        win[-n_taper:] = taper_end

        return win

    @staticmethod
    def rec_filter_coeff(freqs, delta):
        freqs = np.atleast_1d(freqs).astype(float)
        nyq = 1.0 / (2.0 * delta)
        T = 1.0 / freqs
        w = T / (2.0 * np.pi)
        rel = freqs / nyq
        # same empirical correction close to Nyquist
        mask = rel >= 0.2
        w[mask] /= (rel[mask] * 7.0)
        C_HP = w / (w + delta)      # high-pass constant
        C_LP = delta / (w + delta)  # low-pass constant
        return C_HP, C_LP

    @staticmethod
    def rec_filter_norm(freqs, delta, C_HP, C_LP, npoles):
        freqs = np.atleast_1d(freqs).astype(float)
        norm = np.zeros_like(freqs, dtype=float)
        for i, f in enumerate(freqs):
            length = 4.0 / f
            t = np.arange(0, length + delta, delta)
            sig = np.sin(2*np.pi*f*t)
            y = rec_filter.recursive_filter(sig, C_HP[i], C_LP[i], npoles)
            y = y[0] if isinstance(y, tuple) else y
            m = np.max(np.abs(y)) if y.size else 1.0
            norm[i] = m if m != 0.0 else 1.0
        return norm

    # adapters (your cython funcs sometimes return tuples)
    @staticmethod
    def _rf(signal, C_HP, C_LP, npoles, rmem=None):
        out = rec_filter.recursive_filter(signal, C_HP, C_LP, npoles, rmem)
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _rms(x, C_WIN):
        out = lib_rec_rms.recursive_rms(x, C_WIN)
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _hos(x, C_WIN, order, sigma_min=-1.0):
        out = lib_rec_hos.recursive_hos(x, sigma_min=sigma_min, C_WIN=C_WIN, order=order)
        return out[0] if isinstance(out, tuple) else out

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from obspy import read
    import matplotlib.pyplot as plt

    st = read("/Users/robertocabiecesdiaz/Documents/ISP/isp/examples/Earthquake_location_test/ES.EADA..HHZ.D.2015.260")
    tr = st[0].detrend("demean").detrend("linear")

    YN1, CF1, cf_stack, Tn, Nb, freqs = CFMB.compute_tf(tr, filter_npoles=4, var_w=True, CF_type='kurtosis',
                                                        CF_decay_win=4.00, hos_order=4, apply_taper=True)

    t = tr.times()
    mid = len(freqs) // 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(t, tr.data / np.abs(tr.data).max(), 'k', lw=0.8, label="Raw")
    ax1.plot(t, YN1[mid] / (np.abs(YN1[mid]).max() + 1e-12), lw=0.8, label=f"Filtered ~{freqs[mid]:.2f} Hz")
    ax1.legend(loc="upper right")

    ax2.plot(t, cf_stack, lw=0.9, label="CF (kurtosis) – max stack")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()