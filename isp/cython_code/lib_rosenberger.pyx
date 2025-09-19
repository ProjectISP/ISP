# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef inline char _to_char_flag(object v):
    if isinstance(v, (int, np.integer)):
        return <char>int(v)
    if isinstance(v, bytes):
        if len(v) != 1:
            raise ValueError("bytes flag must be length 1")
        return <char>v[0]
    if isinstance(v, str):
        if len(v) != 1:
            raise ValueError("str flag must be length 1")
        return <char>ord(v)
    raise TypeError("flag must be int, 1-char str, or 1-byte bytes")

def rosenberger(cnp.ndarray[cnp.double_t, ndim=1] dataX,
                cnp.ndarray[cnp.double_t, ndim=1] dataY,
                cnp.ndarray[cnp.double_t, ndim=1] dataZ,
                double lam,
                double delta,   # aceptado por compatibilidad; no usado directamente
                proj,
                rl_filter):
    """
    Polarización 'Rosenberger-like' sin dependencias C:
    - Estima covarianza recursiva 3x3 con decaimiento lam in (0,1).
    - Eigen-decomp (eigh) para dirección principal y medida de rectilineidad.
    - Si proj: salida = proyección de v en la dirección principal
      Si no:   salida = (λ1 - λ2) / (λ1 + eps)
    - Si rl_filter: suaviza la salida con el mismo lam (IIR de 1 polo).

    Parámetros
    ----------
    dataX, dataY, dataZ : arrays 1D float64 (misma longitud)  -> he (E), hn (N), vz (Z)
    lam   : float en (0,1)    (lam alto = memoria larga)
    delta : float             (no usado; se mantiene por compatibilidad)
    proj  : 0/1, '0'/'1', b'0'/b'1'
    rl_filter : 0/1, '0'/'1', b'0'/b'1'

    Devuelve
    --------
    polarization : array 1D float64
    """
    cdef Py_ssize_t npts = dataX.shape[0]
    if dataY.shape[0] != npts or dataZ.shape[0] != npts:
        raise ValueError("dataX, dataY y dataZ deben tener la misma longitud")
    if lam <= 0.0 or lam >= 1.0:
        raise ValueError("lam debe estar en (0,1)")

    cdef char c_proj = _to_char_flag(proj)
    cdef char c_rl   = _to_char_flag(rl_filter)

    # Salida
    cdef cnp.ndarray[cnp.double_t, ndim=1] polarization = np.zeros(npts, dtype=np.float64)
    cdef double[:] p = polarization

    # Vectores de entrada (orden físico: e, n, z), vector "v" usará orden [z, n, e]
    # para ser consistente con documentación clásica {ZZ, NS, EW}
    cdef double[:] ex = dataX
    cdef double[:] ny = dataY
    cdef double[:] zz = dataZ

    # Covarianza recursiva 3x3
    cdef cnp.ndarray[cnp.double_t, ndim=2] R = np.zeros((3,3), dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=2] outer = np.empty((3,3), dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] v = np.empty(3, dtype=np.float64)

    cdef double[:] Rv0 = R[0]
    cdef double[:] Rv1 = R[1]
    cdef double[:] Rv2 = R[2]

    cdef double[:] ov0 = outer[0]
    cdef double[:] ov1 = outer[1]
    cdef double[:] ov2 = outer[2]

    cdef double alpha = 1.0 - lam
    cdef double eps = 1e-12

    cdef Py_ssize_t i
    cdef double y_inst = 0.0
    cdef double y_filt = 0.0

    # Bucle principal
    for i in range(npts):
        # Construye vector v = [z, n, e]
        v[0] = zz[i]
        v[1] = ny[i]
        v[2] = ex[i]

        # outer = v v^T (manual para evitar asignaciones adicionales)
        ov0[0] = v[0]*v[0]; ov0[1] = v[0]*v[1]; ov0[2] = v[0]*v[2]
        ov1[0] = v[1]*v[0]; ov1[1] = v[1]*v[1]; ov1[2] = v[1]*v[2]
        ov2[0] = v[2]*v[0]; ov2[1] = v[2]*v[1]; ov2[2] = v[2]*v[2]

        # R = lam*R + (1-lam)*outer
        Rv0[0] = lam*Rv0[0] + alpha*ov0[0]
        Rv0[1] = lam*Rv0[1] + alpha*ov0[1]
        Rv0[2] = lam*Rv0[2] + alpha*ov0[2]
        Rv1[0] = lam*Rv1[0] + alpha*ov1[0]
        Rv1[1] = lam*Rv1[1] + alpha*ov1[1]
        Rv1[2] = lam*Rv1[2] + alpha*ov1[2]
        Rv2[0] = lam*Rv2[0] + alpha*ov2[0]
        Rv2[1] = lam*Rv2[1] + alpha*ov2[1]
        Rv2[2] = lam*Rv2[2] + alpha*ov2[2]

        # Autovalores/autovectores (R simétrica): eigh devuelve ascendente
        # (Necesita GIL; estamos en modo normal)
        w, U = np.linalg.eigh(R)     # w ascendente, U columnas = autovectores

        # Toma modo principal (λ1 = mayor, u1 = columna asociada)
        lam1 = float(w[2]); lam2 = float(w[1])
        u1 = U[:, 2]                 # shape (3,)

        if c_proj != 0:
            # proyección de la muestra actual sobre la dirección principal
            # y_inst = <v, u1>
            y_inst = v[0]*u1[0] + v[1]*u1[1] + v[2]*u1[2]
        else:
            # medida de rectilineidad
            y_inst = (lam1 - lam2) / (lam1 + eps)

        if c_rl != 0:
            # filtro recursivo (mismo lam)
            y_filt = lam * y_filt + alpha * y_inst
            p[i] = y_filt
        else:
            p[i] = y_inst

    return polarization