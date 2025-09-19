# lib_rec_hos.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport pow

# Requerido al usar NumPy tipado
cnp.import_array()

def recursive_hos(cnp.ndarray[cnp.double_t, ndim=1] signal,
                  double sigma_min,
                  double C_WIN,
                  int order,
                  double mean=0.0,
                  double var=0.0,
                  double hos=0.0,
                  int memory_sample=-1,
                  int initialize=0):
    """
    HOS recursivo con ventana exponencial.
    Devuelve: (hos_signal, mean, var, hos) actualizados en memory_sample.
    """
    # ---- Declaraciones (todas aquí) ----
    cdef Py_ssize_t npts = signal.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=1] hos_signal = np.zeros(npts, dtype=np.float64)
    cdef double[:] x = signal
    cdef double[:] y = hos_signal

    cdef int n_win
    cdef double _mean, _var, _hos
    cdef double power
    cdef Py_ssize_t i, kmax
    cdef double var_temp, diff

    # ---- Inicializaciones ----
    if C_WIN != 0.0:
        n_win = <int>(1.0 / C_WIN)
    else:
        n_win = 0

    _mean = mean
    _var  = var
    _hos  = hos

    # Igual que en el C original: división ENTERA (p/2). Si quieres real, usa 2.0
    power = order / 2

    i = 0
    kmax = 0
    var_temp = 0.0
    diff = 0.0

    if memory_sample < 0 or memory_sample >= npts:
        memory_sample = npts - 1

    # ---- Pre-roll opcional ----
    if initialize and n_win > 0:
        kmax = n_win if n_win < npts else npts
        for i in range(kmax):
            _mean = C_WIN * x[i] + (1.0 - C_WIN) * _mean
            _var  = C_WIN * pow(x[i] - _mean, 2.0) + (1.0 - C_WIN) * _var

    # ---- Bucle principal ----
    for i in range(npts):
        _mean = C_WIN * x[i] + (1.0 - C_WIN) * _mean
        diff = x[i] - _mean

        var_temp = C_WIN * (diff * diff) + (1.0 - C_WIN) * _var
        if var_temp > sigma_min:
            _var = var_temp
        else:
            _var = sigma_min

        # HOS normalizada: E[(x-mean)^order / var^(order/2)]
        _hos = C_WIN * (pow(diff, order) / pow(_var, power)) + (1.0 - C_WIN) * _hos

        y[i] = _hos

        if i == memory_sample:
            mean = _mean
            var  = _var
            hos  = _hos

    return hos_signal, mean, var, hos
