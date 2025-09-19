# rec_filter.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

import numpy as np          # para crear/manipular arrays en modo Python
cimport numpy as cnp        # para tipado C de arrays

# Requerido por Cython al usar NumPy tipado
cnp.import_array()

def recursive_filter(cnp.ndarray[cnp.double_t, ndim=1] signal,
                     double C_HP,
                     double C_LP=-1,
                     int npoles=2,
                     cnp.ndarray[cnp.double_t, ndim=1] filterH=None,
                     cnp.ndarray[cnp.double_t, ndim=1] filterL=None,
                     double prev_sample_value=0.0,
                     int memory_sample=-1):
    """
    Versión mínima del filtro recursivo en Cython.
    Devuelve (filt_signal, filterH, filterL, prev_sample_value_actualizado)
    """
    cdef Py_ssize_t npts = signal.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=1] filt_signal = np.zeros(npts, dtype=np.float64)

    if filterH is None:
        filterH = np.zeros(npoles, dtype=np.float64)
    if filterL is None:
        filterL = np.zeros(npoles, dtype=np.float64)

    # Memoryviews tipados (rápidos)
    cdef double[:] x = signal
    cdef double[:] y = filt_signal
    cdef double[:] fH = filterH
    cdef double[:] fL = filterL

    # Buffers de trabajo locales
    cdef cnp.ndarray[cnp.double_t, ndim=1] _H  = np.empty(npoles, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] _H0 = np.empty(npoles, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] _L  = np.empty(npoles, dtype=np.float64)

    cdef double[:] H  = _H
    cdef double[:] H0 = _H0
    cdef double[:] L  = _L

    cdef Py_ssize_t i
    cdef int n
    cdef double s0, s1, prev = prev_sample_value

    # Inicializa estado local
    for n in range(npoles):
        H[n]  = fH[n]
        H0[n] = 0.0
        L[n]  = fL[n]

    if memory_sample < 0 or memory_sample >= npts:
        memory_sample = npts - 1

    # Bucle principal
    for i in range(npts):
        # cadena de HP
        for n in range(npoles):
            if n == 0:
                s0 = prev
                s1 = x[i]
            else:
                s0 = H0[n-1]
                s1 = H[n-1]
            H0[n] = H[n]
            H[n]  = C_HP * (H[n] + s1 - s0)

        if C_LP < 0:
            # solo high-pass
            y[i] = H[npoles-1]
        else:
            # cadena de LP
            for n in range(npoles):
                if n == 0:
                    s0 = H[npoles-1]
                else:
                    s0 = L[n-1]
                L[n] = L[n] + C_LP * (s0 - L[n])
            y[i] = L[npoles-1]

        prev = x[i]

        # guardar estado en el punto pedido
        if i == memory_sample:
            for n in range(npoles):
                fH[n] = H[n]
                fL[n] = L[n]
            prev_sample_value = prev
    return filt_signal
#    return filt_signal, filterH, filterL, prev_sample_value