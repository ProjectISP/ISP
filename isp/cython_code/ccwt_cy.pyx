import math
import multiprocessing
from multiprocessing.pool import ThreadPool
from cpython cimport bool

import numpy as np

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t DCTYPE_t


def compute_atoms(int npts, int srate, float fmin, float fmax, float wmin, float wmax, float tt, int nf):
    # Wavelet parameters
    cdef float dt = 1./srate
    cdef cnp.ndarray[DTYPE_t, ndim=1]  frex = np.logspace(np.log10(fmin), np.log10(fmax), nf, base=10)  # Logarithmically space central frequencies
    cdef cnp.ndarray[DTYPE_t, ndim=1] wtime = np.arange(-tt, tt+dt, dt)  # Kernel of the Mother Morlet Wavelet
    cdef float half_wave = (len(wtime) - 1)/2
    cdef cnp.ndarray[DTYPE_t, ndim=1] n_cycles = np.logspace(np.log10(wmin), np.log10(wmax), nf)

    # FFT parameters
    cdef int n_kern = len(wtime)
    cdef int n_conv = npts + n_kern

    n_conv = 2 ** math.ceil(math.log2(n_conv))

    # loop over frequencies
    cdef list ba_list = []
    cdef int ii
    cdef float fi
    cdef float s
    cdef float normalization
    cdef cnp.ndarray[DCTYPE_t, ndim=1] cmw
    cdef cnp.ndarray[DTYPE_t, ndim=1] cmw_real
    cdef cnp.ndarray[DCTYPE_t, ndim=1] cmw_fft

    for ii, fi in enumerate(frex):
        # Create the Morlet wavelet and get its fft
        s = n_cycles[ii]/(2*np.pi*fi)
        # Normalize Factor
        normalization = 1/(np.pi*s**2)**0.25
        # Complex sine = np.multiply(1j*2*(np.pi)*frex[fi],wtime))
        # Gaussian = np.exp(-1*np.divide(np.power(wtime,2),2*s**2))
        cmw = np.multiply(np.exp(np.multiply(1j*2*np.pi*fi, wtime)), np.exp(-1*np.divide(np.power(wtime, 2), 2*s**2)))
        cmw = cmw.conjugate()
        # Normalizing. The square root term causes the wavelet to be normalized to have an energy of 1.
        cmw = normalization * cmw
        cmw_real = np.real(cmw)
        # Calculate the fft of the "atom"
        cmw_fft = np.fft.rfft(cmw_real, n_conv)

        # Convolution
        ba_list.append(cmw_fft)

    cdef cnp.ndarray[DCTYPE_t, ndim=2] ba = np.asarray(ba_list)

    return ba, n_conv, frex, half_wave


cdef int get_nproc():
    cdef int total_cpu = multiprocessing.cpu_count()
    cdef int nproc = total_cpu - 2 if total_cpu > 3 else total_cpu - 1
    nproc = max(nproc, 1)
    return nproc


cpdef  cnp.ndarray[DTYPE_t, ndim=1] ccwt_ifft(cnp.ndarray[DCTYPE_t, ndim=1] data, int n, float half_wave, int npts):
    cdef cnp.ndarray[DTYPE_t, ndim=1] cwt = np.fft.irfft(data, n=n)
    cwt = cwt - np.mean(cwt)
    cdef cnp.ndarray[DTYPE_t, ndim=1] d = np.diff(np.log10(np.abs(cwt[<int>(half_wave + 1):npts + <int>(half_wave + 1)])))
    return d


cpdef cnp.ndarray[DTYPE_t, ndim=1] ccwt_ba_fast(cnp.ndarray data, tuple param, bool parallel=False):
    cdef cnp.ndarray[DCTYPE_t, ndim=2] ba
    cdef int nConv
    cdef cnp.ndarray[DTYPE_t, ndim=1] frex
    cdef float half_wave
    ba, nConv, frex, half_wave = param
    cdef int npts = len(data)

    # FFT data
    cdef cnp.ndarray[DCTYPE_t, ndim=1] data_fft = np.fft.rfft(data, n=nConv)
    data_fft = data_fft - np.mean(data_fft)
    cdef cnp.ndarray[DCTYPE_t, ndim=2] m = np.multiply(ba, data_fft)

    parallel = parallel if len(frex) > 1 else False
    cdef list tf_list = []
    cdef int nproc
    cdef cnp.ndarray[DCTYPE_t, ndim=1] row

    if parallel:
        nproc = get_nproc()
        nproc = min(nproc, len(frex))
        pool = ThreadPool(processes=nproc)
        results = [pool.apply_async(ccwt_ifft, args=(row, nConv, half_wave, npts)) for row in m]
        tf_list = [p.get() for p in results]
        pool.close()
    else:
        for row in m:
            tf_list.append(ccwt_ifft(row, nConv, half_wave, npts))

    tf = np.asarray(tf_list)  # convert to array
    cdef cnp.ndarray[DTYPE_t, ndim=1] sc = np.mean(tf, axis=0, dtype=np.float64)

    return sc
