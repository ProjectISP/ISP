# lib_rec_rms.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport pow, sqrt

cnp.import_array()

def recursive_rms(cnp.ndarray[cnp.double_t, ndim=1] signal,
                  double C_WIN,
                  double mean_sq=0.0,
                  int memory_sample=-1,
                  int initialize=0):
    """
    RMS recursivo con ventana exponencial equivalente (C_WIN ~ 1/N).

    Parámetros
    ----------
    signal : 1D float64 array
    C_WIN : double in (0,1]
    mean_sq : double
        Estado inicial (raíz de la media cuadrática previa).
    memory_sample : int
        Índice donde se guarda el estado (si inválido -> npts-1).
    initialize : int (0/1)
        Si 1, hace un pre-roll de n_win=int(1/C_WIN) muestras: sqrt(mean(x^2)).

    Devuelve
    --------
    rms_signal : 1D float64 array
    mean_sq    : double actualizado en memory_sample
    """
    # ---- Declaraciones ----
    cdef Py_ssize_t npts = signal.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=1] rms_signal = np.zeros(npts, dtype=np.float64)
    cdef double[:] x = signal
    cdef double[:] y = rms_signal

    cdef int n_win
    cdef double _mean_sq
    cdef Py_ssize_t i, j, jmax
    cdef double acc

    # ---- Inicializaciones ----
    if C_WIN != 0.0:
        n_win = <int>(1.0 / C_WIN)
    else:
        n_win = 0  # evita div/0; no habrá pre-roll

    _mean_sq = mean_sq
    i = 0
    j = 0
    jmax = 0
    acc = 0.0

    if memory_sample < 0 or memory_sample >= npts:
        memory_sample = npts - 1

    # ---- Pre-roll opcional (igual que el C) ----
    if initialize and n_win > 0:
        jmax = n_win if n_win < npts else npts
        # Nota: el C original acumula sobre _mean_sq SIN reiniciarlo a 0.
        # Mantenemos la semántica:
        for j in range(jmax):
            _mean_sq = _mean_sq + pow(x[j], 2.0)
        _mean_sq = sqrt(_mean_sq / jmax)  # en C: /n_win, pero si jmax<n_win usamos jmax real

    # ---- Bucle principal ----
    for i in range(npts):
        # _mean_sq nuevo = sqrt( C_WIN * x[i]^2 + (1 - C_WIN) * (_mean_sq)^2 )
        _mean_sq = sqrt(C_WIN * pow(x[i], 2.0) + (1.0 - C_WIN) * pow(_mean_sq, 2.0))
        y[i] = _mean_sq

        if i == memory_sample:
            mean_sq = _mean_sq

    return rms_signal, mean_sq
