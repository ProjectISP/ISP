# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, floor, ceil
cimport cython
import numpy as np
cimport numpy as np

# =========================
# Internal C-level routines
# =========================

cdef void _gausscoeff(double sigma, double* A, int* nA, double* B, int* nB) nogil:
    cdef int i
    cdef double q
    cdef double* b
    cdef double q2, q3

    if sigma > 0.5:
        q = 0.98711*sigma - 0.96330
    elif sigma == 0.5:
        q = 3.97156 - 4.14554 * sqrt(1.0 - 0.26891*sigma)
    else:
        # Provide a benign filter if sigma < 0.5 (C version exited; here we just pass-through)
        A[0] = 1.0
        for i in range(1, 4):
            A[i] = 0.0
        B[0] = 1.0
        nA[0] = 4
        nB[0] = 1
        return

    b = <double*> malloc(4 * sizeof(double))
    if b == NULL:
        A[0] = 1.0
        for i in range(1, 4):
            A[i] = 0.0
        B[0] = 1.0
        nA[0] = 4
        nB[0] = 1
        return

    q2 = q * q
    q3 = q2 * q

    b[0] = 1.57825 + 2.44413*q + 1.4281*q2 + 0.422205*q3
    b[1] = 2.44413*q + 2.85619*q2 + 1.26661*q3
    b[2] = -(1.4281*q2 + 1.26661*q3)
    b[3] = 0.422205*q3

    B[0] = 1.0 - ((b[1] + b[2] + b[3]) / b[0])

    A[0] = 1.0
    for i in range(1, 4):
        A[i] = -b[i] / b[0]

    nA[0] = 4
    nB[0] = 1
    free(b)


cdef void _lfilter(const double* signal, double* filt_signal, int npts,
                   const double* A, int nA, const double* B, int nB) nogil:
    cdef int n, na, nb
    for n in range(npts):
        filt_signal[n] = 0.0
        for nb in range(nB):
            if nb > n:
                break
            filt_signal[n] += B[nb] * signal[n - nb]
        for na in range(1, nA):
            if na > n:
                break
            filt_signal[n] -= A[na] * filt_signal[n - na]
        filt_signal[n] /= A[0]


cdef void _reverse(const double* signal, double* rev_signal, int npts) nogil:
    cdef int n, endi
    cdef double tmp
    for n in range(npts):
        rev_signal[n] = signal[n]
    endi = npts - 1
    for n in range(npts // 2):
        tmp = rev_signal[n]
        rev_signal[n] = rev_signal[endi]
        rev_signal[endi] = tmp
        endi -= 1


cdef void _Gaussian1D(double* signal, int npts, double sigma) nogil:
    cdef double* A = NULL
    cdef double* B = NULL
    cdef int nA = 0
    cdef int nB = 0
    cdef double* rev_filt_signal = NULL
    cdef double* filt_signal = NULL
    cdef int n

    if npts < 4:
        return

    A = <double*> malloc(4 * sizeof(double))
    B = <double*> malloc(4 * sizeof(double))
    if A == NULL or B == NULL:
        if A != NULL: free(A)
        if B != NULL: free(B)
        return

    _gausscoeff(sigma, A, &nA, B, &nB)

    rev_filt_signal = <double*> malloc(npts * sizeof(double))
    filt_signal     = <double*> malloc(npts * sizeof(double))
    if rev_filt_signal == NULL or filt_signal == NULL:
        if A != NULL: free(A)
        if B != NULL: free(B)
        if rev_filt_signal != NULL: free(rev_filt_signal)
        if filt_signal != NULL: free(filt_signal)
        return

    _lfilter(signal, filt_signal, npts, A, nA, B, nB)
    _reverse(filt_signal, rev_filt_signal, npts)
    _lfilter(rev_filt_signal, filt_signal, npts, A, nA, B, nB)
    _reverse(filt_signal, rev_filt_signal, npts)

    for n in range(npts):
        signal[n] = rev_filt_signal[n]

    free(A)
    free(B)
    free(rev_filt_signal)
    free(filt_signal)


cdef void _local_CCr(const double* signal1, const double* signal2, int npts,
                     double* cc, int lmax, double sigma) nogil:
    cdef int n, l, l_f, l_g
    cdef double* _cc
    cdef double shift1_1, shift2_2, shift1_2, shift2_1

    for l in range(-lmax, lmax):
        l_f = <int> floor(l / 2.0)
        l_g = <int> ceil(l / 2.0)

        _cc = &cc[npts * (l + lmax)]

        for n in range(npts):
            if (n - l_f) >= 0 and (n - l_f) < npts:
                shift1_1 = signal1[n - l_f]
            else:
                shift1_1 = 0.0

            if (n + l_g) >= 0 and (n + l_g) < npts:
                shift2_2 = signal2[n + l_g]
            else:
                shift2_2 = 0.0

            if (n - l_g) >= 0 and (n - l_g) < npts:
                shift1_2 = signal1[n - l_g]
            else:
                shift1_2 = 0.0

            if (n + l_f) >= 0 and (n + l_f) < npts:
                shift2_1 = signal2[n + l_f]
            else:
                shift2_1 = 0.0

            _cc[n] = 0.5 * (shift1_1 * shift2_2 + shift1_2 * shift2_1)

        if sigma > 0.0:
            _Gaussian1D(_cc, npts, sigma)

# =======================================
# Python-callable API (NumPy interface)
# =======================================

@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian1d(np.ndarray[np.double_t, ndim=1, mode='c'] signal, double sigma):
    """
    In-place Gaussian smoothing (Young & Vliet IIR). Returns the same array.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    cdef int npts = signal.shape[0]
    if npts < 4:
        raise ValueError("Signal too short (need >= 4 samples).")
    with nogil:
        _Gaussian1D(<double*> signal.data, npts, sigma)
    return signal


@cython.boundscheck(False)
@cython.wraparound(False)
def local_CCr(np.ndarray[np.double_t, ndim=1, mode='c'] signal1,
              np.ndarray[np.double_t, ndim=1, mode='c'] signal2,
              int lmax, double sigma=0.0):
    """
    Local cross-correlation for l in [-lmax, ..., lmax-1].
    Returns array of shape (2*lmax, npts).
    """
    if signal1.ndim != 1 or signal2.ndim != 1:
        raise ValueError("signal1 and signal2 must be 1D")
    if signal1.shape[0] != signal2.shape[0]:
        raise ValueError("signals must have the same length")
    if lmax <= 0:
        raise ValueError("lmax must be > 0")

    cdef int npts = signal1.shape[0]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] cc = np.zeros((2*lmax, npts), dtype=np.float64)

    with nogil:
        _local_CCr(<const double*> signal1.data,
                   <const double*> signal2.data,
                   npts,
                   <double*> cc.data,
                   lmax,
                   sigma)
    return cc
