#!/usr/bin/env python3

# raydec_cli.py
# RayDec (single station) with ObsPy + Numba stacking
# Config from YAML + command-line runner
#
# Author: Roberto Cabieces 2025

from __future__ import annotations
import argparse
import glob
import os
import sys
import math
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
from scipy.signal import cheb1ord, cheby1, lfilter
from obspy import Stream, Trace, UTCDateTime

# -------- YAML loader --------
try:
    import yaml  # pip install pyyaml
except Exception as exc:
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None

# -------- Optional numba --------
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    njit = None



"""
typical config.yaml

fmin: 0.05
fmax: 10.0
fsteps: 100
cycles: 10.0
dfpar: 0.2
win_seconds: 600.0
taper_frac: 0.01
target_fs: null
allow_short_windows: true
verbose: true
"""




# ==============================
#        NUMBA HOT LOOP
# ==============================

if _HAS_NUMBA:
    @njit(cache=False, fastmath=True)
    def _raydec_stack_numba(vv_f, nn_f, ee_f, wl, shift, start_idx):
        """
        Stack around negative->positive zero-crossings on vv_f, with N/E shifted by 'shift'.
        wl, shift, start_idx are precomputed outside numba to:
          - match MATLAB more closely (ceil/floor handled outside)
          - avoid recomputation inside hot loop
        Returns (v_amp, h_amp) or (nan, nan).
        """
        Nw = vv_f.shape[0]
        end_idx = (Nw - 1) - wl
        if start_idx >= end_idx:
            return math.nan, math.nan

        vertsum = np.zeros(wl, dtype=np.float64)
        horsum = np.zeros(wl, dtype=np.float64)
        dvcount = 0

        for idx in range(start_idx, end_idx):
            # sign with zeros treated as +1
            s0 = 1.0 if vv_f[idx] >= 0.0 else -1.0
            s1 = 1.0 if vv_f[idx + 1] >= 0.0 else -1.0

            # negative -> positive crossing
            if 0.5 * (s1 - s0) == 1.0:
                i0h = idx - shift
                i1h = i0h + wl
                if i0h < 0 or i1h > Nw:
                    continue

                # azimuth integrals
                integral1 = 0.0
                integral2 = 0.0
                for k in range(wl):
                    vz = vv_f[idx + k]
                    integral1 += vz * ee_f[i0h + k]
                    integral2 += vz * nn_f[i0h + k]

                theta = math.atan2(integral1, integral2)
                if integral2 < 0.0:
                    theta += math.pi
                theta = (theta + math.pi) % (2.0 * math.pi)

                sth = math.sin(theta)
                cth = math.cos(theta)

                # correlation
                sum_v2 = 0.0
                sum_h2 = 0.0
                sum_vh = 0.0
                for k in range(wl):
                    vz = vv_f[idx + k]
                    hz = sth * ee_f[i0h + k] + cth * nn_f[i0h + k]
                    sum_v2 += vz * vz
                    sum_h2 += hz * hz
                    sum_vh += vz * hz

                denom = math.sqrt(sum_v2 * sum_h2)
                if denom <= 0.0:
                    continue

                corr = sum_vh / denom
                if corr >= -1.0:
                    wgt = corr * corr
                    for k in range(wl):
                        vz = vv_f[idx + k]
                        hz = sth * ee_f[i0h + k] + cth * nn_f[i0h + k]
                        vertsum[k] += wgt * vz
                        horsum[k] += wgt * hz

                dvcount += 1

        if dvcount == 0:
            return math.nan, math.nan

        sum_v = 0.0
        sum_h = 0.0
        for k in range(wl):
            sum_v += vertsum[k] * vertsum[k]
            sum_h += horsum[k] * horsum[k]

        return math.sqrt(sum_v), math.sqrt(sum_h)


def _raydec_stack_py(vv_f, nn_f, ee_f, wl, shift, start_idx) -> Tuple[float, float]:
    Nw = vv_f.size
    end_idx = (Nw - 1) - wl
    if start_idx >= end_idx:
        return np.nan, np.nan

    vertsum = np.zeros(wl, dtype=float)
    horsum = np.zeros(wl, dtype=float)
    dvcount = 0

    for idx in range(start_idx, end_idx):
        s0 = 1.0 if vv_f[idx] >= 0.0 else -1.0
        s1 = 1.0 if vv_f[idx + 1] >= 0.0 else -1.0
        if 0.5 * (s1 - s0) == 1.0:
            i0h = idx - shift
            i1h = i0h + wl
            if i0h < 0 or i1h > Nw:
                continue

            vseg = vv_f[idx:idx + wl]
            eseg = ee_f[i0h:i1h]
            nseg = nn_f[i0h:i1h]

            integral1 = float(np.sum(vseg * eseg))
            integral2 = float(np.sum(vseg * nseg))
            theta = math.atan2(integral1, integral2)
            if integral2 < 0.0:
                theta += math.pi
            theta = (theta + math.pi) % (2.0 * math.pi)

            hseg = math.sin(theta) * eseg + math.cos(theta) * nseg

            denom = math.sqrt(float(np.sum(vseg * vseg) * np.sum(hseg * hseg)))
            if denom <= 0.0:
                continue

            corr = float(np.sum(vseg * hseg)) / denom
            if corr >= -1.0:
                wgt = corr * corr
                vertsum += wgt * vseg
                horsum += wgt * hseg
            dvcount += 1

    if dvcount == 0:
        return np.nan, np.nan

    return float(np.sqrt(np.sum(vertsum ** 2))), float(np.sqrt(np.sum(horsum ** 2)))


def _raydec_stack(vv_f, nn_f, ee_f, wl, shift, start_idx) -> Tuple[float, float]:
    vv_f = np.asarray(vv_f, dtype=np.float64)
    nn_f = np.asarray(nn_f, dtype=np.float64)
    ee_f = np.asarray(ee_f, dtype=np.float64)

    if _HAS_NUMBA:
        return _raydec_stack_numba(vv_f, nn_f, ee_f, int(wl), int(shift), int(start_idx))
    return _raydec_stack_py(vv_f, nn_f, ee_f, int(wl), int(shift), int(start_idx))


# ==============================
#             CONFIG
# ==============================

_DEFAULT_CFG: Dict[str, Any] = {
    "fmin": 0.2,
    "fmax": 10.0,
    "fsteps": 100,
    "cycles": 10.0,
    "dfpar": 0.20,
    "nwind": 1,
    "win_seconds": None,
    "taper_frac": 0.01,
    "cheb_gpass": 1.0,
    "cheb_gstop": 5.0,
    "cheb_order_min": 2,
    "cheb_order_max": 12,
    "target_fs": None,
    "fs_atol": 1e-6,
    "allow_short_windows": True,
    "verbose": True,
}


def load_config(yaml_path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError(f"PyYAML is required to load YAML configs: {_YAML_IMPORT_ERROR}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError("YAML root must be a mapping/dict of parameters.")

    cfg = dict(_DEFAULT_CFG)
    cfg.update(user_cfg)

    # Basic validation + types
    cfg["fmin"] = float(cfg["fmin"])
    cfg["fmax"] = float(cfg["fmax"])
    cfg["fsteps"] = int(cfg["fsteps"])
    cfg["cycles"] = float(cfg["cycles"])
    cfg["dfpar"] = float(cfg["dfpar"])
    cfg["nwind"] = int(cfg["nwind"])
    cfg["win_seconds"] = None if cfg["win_seconds"] is None else float(cfg["win_seconds"])
    cfg["taper_frac"] = float(cfg["taper_frac"])
    cfg["cheb_gpass"] = float(cfg["cheb_gpass"])
    cfg["cheb_gstop"] = float(cfg["cheb_gstop"])
    cfg["cheb_order_min"] = int(cfg["cheb_order_min"])
    cfg["cheb_order_max"] = int(cfg["cheb_order_max"])
    cfg["target_fs"] = None if cfg["target_fs"] is None else float(cfg["target_fs"])
    cfg["fs_atol"] = float(cfg["fs_atol"])
    cfg["allow_short_windows"] = bool(cfg["allow_short_windows"])
    cfg["verbose"] = bool(cfg["verbose"])

    if cfg["fsteps"] < 2:
        raise ValueError("fsteps must be >= 2")
    if cfg["cycles"] <= 0:
        raise ValueError("cycles must be > 0")
    if cfg["dfpar"] <= 0:
        raise ValueError("dfpar must be > 0")

    return cfg


# ==============================
#           UTILITIES
# ==============================

def _cosine_taper(N: int, frac: float) -> np.ndarray:
    n_tap = max(1, int(round(N * frac)))
    if n_tap * 2 >= N:
        n_tap = max(1, N // 4)
    ramp = (1 - np.cos(np.linspace(0, np.pi, n_tap))) / 2.0
    mid = np.ones(N - 2 * n_tap, dtype=float)
    return np.concatenate([ramp, mid, ramp[::-1]])


def _detrend_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones(n)]).T
    m, c = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - (m * t + c)


def _log_freqs(fstart: float, fend: float, fsteps: int) -> np.ndarray:
    constlog = (fend / fstart) ** (1.0 / (fsteps - 1))
    out = np.zeros(fsteps, dtype=float)
    f = fstart
    for i in range(fsteps):
        out[i] = f
        f *= constlog
    return out


def _safe_makedirs(path: Optional[str]) -> None:
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


# ==============================
#        STREAM MANAGEMENT
# ==============================

class StreamManager:
    _sets = [
        ("Z", "N", "E"),
        ("Z", "1", "2"),
        ("Z", "Y", "X"),
    ]

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _pick_three(self, st: Stream) -> Dict[str, Trace]:
        by_last: Dict[str, List[Trace]] = {}
        for tr in st:
            code = (tr.stats.channel or "").strip()
            last = code[-1] if code else ""
            by_last.setdefault(last.upper(), []).append(tr)

        for Zc, Nc, Ec in self._sets:
            if Zc in by_last and Nc in by_last and Ec in by_last:
                return {"Z": by_last[Zc][0], "N": by_last[Nc][0], "E": by_last[Ec][0]}

        raise ValueError("Could not find a valid 3C set: need (Z,N,E) or (Z,1,2) or (Z,Y,X).")

    def _get_id(self, tr: Trace) -> Dict[str, Any]:
        return {
            "network": tr.stats.network,
            "station": tr.stats.station,
            "starttime": tr.stats.starttime,
            "channel": tr.stats.channel,
        }

    def _common_overlap(self, Z: Trace, N: Trace, E: Trace) -> Tuple[UTCDateTime, UTCDateTime]:
        t0 = max(Z.stats.starttime, N.stats.starttime, E.stats.starttime)
        t1 = min(Z.stats.endtime,   N.stats.endtime,   E.stats.endtime)
        if t1 <= t0:
            raise ValueError("No overlapping time interval among the three components.")
        return t0, t1

    def _resample_if_needed(self, Z: Trace, N: Trace, E: Trace) -> Tuple[Trace, Trace, Trace]:
        cfg = self.cfg
        if cfg["target_fs"] is not None:
            fs = float(cfg["target_fs"])
            for tr in (Z, N, E):
                if abs(tr.stats.sampling_rate - fs) > cfg["fs_atol"]:
                    tr.resample(fs, no_filter=True)
            return Z, N, E

        fsZ = float(Z.stats.sampling_rate)
        for tr in (N, E):
            if abs(tr.stats.sampling_rate - fsZ) > cfg["fs_atol"]:
                tr.resample(fsZ, no_filter=True)
        return Z, N, E

    def _equalize_length(self, Z: Trace, N: Trace, E: Trace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nmin = min(len(Z.data), len(N.data), len(E.data))
        v = np.asarray(Z.data[:nmin], dtype=float)
        n = np.asarray(N.data[:nmin], dtype=float)
        e = np.asarray(E.data[:nmin], dtype=float)
        fs = float(Z.stats.sampling_rate)
        t = np.arange(nmin, dtype=float) / fs
        return v, n, e, t

    def prepare_arrays(self, st: Stream) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
        picks = self._pick_three(st)
        Z, N, E = picks["Z"].copy(), picks["N"].copy(), picks["E"].copy()

        t0, t1 = self._common_overlap(Z, N, E)
        for tr in (Z, N, E):
            tr.trim(t0, t1, pad=False)

        Z, N, E = self._resample_if_needed(Z, N, E)

        t0, t1 = self._common_overlap(Z, N, E)
        for tr in (Z, N, E):
            tr.trim(t0, t1, pad=False)

        v, n, e, t = self._equalize_length(Z, N, E)
        fs = float(Z.stats.sampling_rate)
        identifier = self._get_id(Z)

        return v, n, e, t, fs, identifier


# ==============================
#           ANALYZER
# ==============================

def _design_filter_for_f(
    f: float,
    fstart: float,
    fend: float,
    fnyq: float,
    dfpar: float,
    cheb_gpass: float,
    cheb_gstop: float,
    order_min: int,
    order_max: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Precompute (b,a) for a given center frequency f.
    Returns None if invalid band.
    """
    df = dfpar * f
    f1 = max(fstart, f - 0.5 * df)
    f2 = min(fnyq,   f + 0.5 * df)
    if not (f2 > f1 > 0.0):
        return None

    fp1 = f1 + (f2 - f1) / 10.0
    fp2 = f2 - (f2 - f1) / 10.0
    fs1 = max(fstart * 0.5, f1 - (f2 - f1) / 10.0)
    fs2 = min(fnyq * 0.999, f2 + (f2 - f1) / 10.0)

    Wp = np.clip([fp1 / fnyq, fp2 / fnyq], 1e-6, 0.999999)
    Ws = np.clip([fs1 / fnyq, fs2 / fnyq], 1e-6, 0.999999)
    if not (Wp[0] < Wp[1] and Ws[0] < Ws[1] and Wp[1] < 1.0):
        return None

    try:
        na, wn = cheb1ord(Wp, Ws, cheb_gpass, cheb_gstop)
        na = max(order_min, min(int(na), order_max))
        b, a = cheby1(na, rp=0.5, Wn=wn, btype="bandpass")
        return b, a
    except Exception:
        try:
            b, a = cheby1(6, rp=0.5, Wn=Wp, btype="bandpass")
            return b, a
        except Exception:
            return None


class RayDecAnalyzer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(
        self,
        v: np.ndarray,
        n: np.ndarray,
        e: np.ndarray,
        t: np.ndarray,
        fs: float
    ) -> Tuple[np.ndarray, np.ndarray]:

        cfg = self.cfg

        if v.size != n.size or v.size != e.size or v.size != t.size:
            raise ValueError("v, n, e, t must have equal lengths.")

        tau = 1.0 / fs
        fnyq = 0.5 * fs

        fstart = float(cfg["fmin"])
        fend = min(float(cfg["fmax"]), fnyq)
        if fstart >= fend:
            raise ValueError("Invalid frequency band (check fmin/fmax and sampling).")

        N = v.size

        # windowing
        if cfg["win_seconds"] is not None:
            K = int(np.floor(float(cfg["win_seconds"]) * fs))
            if K <= 0:
                raise ValueError("win_seconds too small (window has 0 samples).")
            nwind_eff = int(np.floor(N / K))
            if nwind_eff < 1:
                raise ValueError("Record too short for win_seconds.")
        else:
            nwind_eff = int(cfg["nwind"])
            if nwind_eff < 1:
                raise ValueError("nwind must be >= 1.")
            K = int(np.floor(N / nwind_eff))

        if (K < int(fs)) and (not cfg["allow_short_windows"]):
            raise ValueError("Sub-window too short. Increase win_seconds or reduce nwind.")

        fsteps = int(cfg["fsteps"])
        V = np.zeros((fsteps, nwind_eff), dtype=float)
        W = np.full((fsteps, nwind_eff), np.nan, dtype=float)

        freqs = _log_freqs(fstart, fend, fsteps)

        # ---- PRECOMPUTE FILTERS (big speed win) ----
        filters: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * fsteps
        for fi, f0 in enumerate(freqs):
            filters[fi] = _design_filter_for_f(
                f=float(f0),
                fstart=fstart,
                fend=fend,
                fnyq=fnyq,
                dfpar=float(cfg["dfpar"]),
                cheb_gpass=float(cfg["cheb_gpass"]),
                cheb_gstop=float(cfg["cheb_gstop"]),
                order_min=int(cfg["cheb_order_min"]),
                order_max=int(cfg["cheb_order_max"]),
            )

        if cfg["verbose"]:
            print(f"[RayDec] fs={fs:.1f} Hz | {nwind_eff} windows | {K/fs:.1f} s/window | {fsteps} freqs | Numba={'ON' if _HAS_NUMBA else 'OFF'}")

        for w in range(nwind_eff):
            i0 = w * K
            i1 = i0 + K

            vv = _detrend_linear(v[i0:i1])
            nn = _detrend_linear(n[i0:i1])
            ee = _detrend_linear(e[i0:i1])

            Nw = vv.size
            taper = _cosine_taper(Nw, float(cfg["taper_frac"]))

            vertical_amp = np.full(fsteps, np.nan, dtype=float)
            horizontal_amp = np.full(fsteps, np.nan, dtype=float)

            for fi, f0 in enumerate(freqs):
                V[fi, w] = f0
                filt = filters[fi]
                if filt is None:
                    continue

                b, a = filt

                vv_f = lfilter(b, a, taper * vv)
                nn_f = lfilter(b, a, taper * nn)
                ee_f = lfilter(b, a, taper * ee)

                # Precompute wl/shift/start_idx outside hot loop (closer to MATLAB)
                DT = float(cfg["cycles"]) / float(f0)
                wl = int(DT / tau + 0.5)  # round
                if wl < 4:
                    continue

                shift = int(1.0 / (4.0 * float(f0) * tau))              # floor
                start_idx = int(math.ceil(1.0 / (4.0 * float(f0) * tau))) + 1  # ceil + 1 (MATLAB style)

                v_amp, h_amp = _raydec_stack(vv_f, nn_f, ee_f, wl, shift, start_idx)
                vertical_amp[fi] = v_amp
                horizontal_amp[fi] = h_amp

            if cfg["verbose"]:
                nan_ratio = float(np.mean(~np.isfinite(vertical_amp)))
                print(f"[RayDec] Window {w+1}/{nwind_eff} (nan freq ratio={nan_ratio:.2f})")

            with np.errstate(invalid="ignore", divide="ignore"):
                W[:, w] = horizontal_amp / vertical_amp

        return V, W


# ==============================
#            RESULT
# ==============================

class RayDecResult:
    def __init__(self, identifier: Dict[str, Any], freqs: np.ndarray, ellipticity: np.ndarray, fs: float, t_total: float):
        self.id = identifier
        self.freqs = freqs
        self.ellipticity = ellipticity
        self.fs = fs
        self.t_total = t_total

    def combine_windows(self, how: str = "median") -> Tuple[np.ndarray, np.ndarray]:
        if how not in {"median", "mean"}:
            raise ValueError("how must be 'median' or 'mean'")
        f = self.freqs[:, 0]
        if how == "median":
            ell = np.nanmedian(self.ellipticity, axis=1)
        else:
            ell = np.nanmean(self.ellipticity, axis=1)
        return f, ell

    def stats_across_windows(self):
        f = self.freqs[:, 0]
        mu = np.nanmean(self.ellipticity, axis=1)
        med = np.nanmedian(self.ellipticity, axis=1)
        sd = np.nanstd(self.ellipticity, axis=1, ddof=1)
        mad = 1.4826 * np.nanmedian(np.abs(self.ellipticity - med[:, None]), axis=1)
        n_eff = np.sum(np.isfinite(self.ellipticity), axis=1)
        return f, mu, med, sd, mad, n_eff


# ==============================
#           PIPELINE
# ==============================

class RayDecPipeline:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.manager = StreamManager(self.cfg)
        self.analyzer = RayDecAnalyzer(self.cfg)

    def fit(self, st: Stream) -> RayDecResult:
        v, n, e, t, fs, identifier = self.manager.prepare_arrays(st)
        V, W = self.analyzer.run(v, n, e, t, fs)
        t_total = float(t[-1] - t[0]) if t.size > 1 else 0.0
        return RayDecResult(identifier=identifier, freqs=V, ellipticity=W, fs=fs, t_total=t_total)


# ==============================
#       OUTPUT HELPERS
# ==============================

def save_npz(result: RayDecResult, out_dir: str, aggregate: str = "median") -> str:
    """
    Saves a compact, portable file: .npz with arrays + id.
    """
    _safe_makedirs(out_dir)

    f, mu, med, sd, mad, n_eff = result.stats_across_windows()
    f2, ell = result.combine_windows(how=aggregate)
    assert np.allclose(f, f2, equal_nan=True)

    t1 = result.id.get("starttime", None)
    sta = result.id.get("station", "STA")
    net = result.id.get("network", "NET")
    y = getattr(t1, "year", "YYYY")
    j = getattr(t1, "julday", "JJJ")

    base = f"{net}.{sta}.{y}.{j}.ellipticity"
    path = os.path.join(out_dir, base)

    # avoid overwrite
    if os.path.exists(path + ".npz"):
        k = 1
        while os.path.exists(f"{path}_{k}.npz"):
            k += 1
        path = f"{path}_{k}"

    path = path + ".npz"

    np.savez_compressed(
        path,
        frequency=f,
        ell=ell,
        mu=mu,
        med=med,
        sd=sd,
        mad=mad,
        n_eff=n_eff,
        fs=np.array([result.fs], dtype=float),
        t_total=np.array([result.t_total], dtype=float),
        id=np.array([str(result.id)], dtype=object),
    )
    return path


def plot_png(result: RayDecResult, png_dir: str, aggregate: str = "median", xscale: str = "log") -> str:
    import matplotlib.pyplot as plt

    _safe_makedirs(png_dir)

    f, mu, med, sd, mad, n_eff = result.stats_across_windows()
    f2, ell = result.combine_windows(how=aggregate)
    assert np.allclose(f, f2, equal_nan=True)

    t1 = result.id.get("starttime", None)
    sta = result.id.get("station", "STA")
    net = result.id.get("network", "NET")
    y = getattr(t1, "year", "YYYY")
    j = getattr(t1, "julday", "JJJ")

    base = f"{net}.{sta}.{y}.{j}"
    path = os.path.join(png_dir, base)

    if os.path.exists(path + ".png"):
        k = 1
        while os.path.exists(f"{path}_{k}.png"):
            k += 1
        path = f"{path}_{k}"
    path = path + ".png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(f, med - sd, med + sd, alpha=0.2, label="±1σ (across windows)")
    ax.plot(f, ell, lw=1.0, label=f"{net}.{sta} ({aggregate})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H/V ellipticity")
    ax.set_xscale(xscale)
    ax.grid(True, which="both", ls=":")
    ax.set_title(f"RayDec ellipticity {y}.{j}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ==============================
#              CLI
# ==============================

def _read_stream_from_inputs(inputs: List[str]) -> Stream:
    """
    Inputs can be:
      - multiple file paths
      - one or more glob patterns
    """
    from obspy import read

    files: List[str] = []
    for item in inputs:
        expanded = glob.glob(item)
        if expanded:
            files.extend(expanded)
        else:
            # keep as-is, might be a direct filename
            files.append(item)

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No input files found (check your paths/globs).")

    st = Stream()
    for fp in files:
        st += read(fp)

    return st


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="raydec_cli",
        description="RayDec (single station) ellipticity from 3C data using YAML config.",
    )
    p.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config file.",
    )
    p.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="Input waveform files or globs (e.g., CHZ* CHN* CHE* or *.mseed).",
    )
    p.add_argument(
        "--out-npz",
        default=None,
        help="Output folder for .npz result (optional).",
    )
    p.add_argument(
        "--out-png",
        default=None,
        help="Output folder for .png plot (optional).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate plots even if --out-png is given.",
    )
    p.add_argument(
        "--preprocess",
        choices=["none", "basic"],
        default="basic",
        help="Preprocess before RayDec: 'basic' = merge+detrend+taper; 'none' = keep as-is.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively (in addition to saving) if matplotlib backend allows.",
    )

    args = p.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Config load failed: {e}", file=sys.stderr)
        return 2

    try:
        st = _read_stream_from_inputs(args.input)
    except Exception as e:
        print(f"[ERROR] Reading input failed: {e}", file=sys.stderr)
        return 2

    if args.preprocess == "basic":
        # keep this simple & robust; your runner can do more outside
        try:
            st.merge()
        except Exception:
            pass
        try:
            st.detrend("linear")
        except Exception:
            pass
        try:
            st.taper(max_percentage=0.05)
        except Exception:
            pass

    pipe = RayDecPipeline(config_path=args.config)

    try:
        res = pipe.fit(st)
    except Exception as e:
        print(f"[ERROR] RayDec failed: {e}", file=sys.stderr)
        return 1

    # outputs
    if args.out_npz:
        try:
            path = save_npz(res, args.out_npz, aggregate="median")
            print(f"[OK] Wrote NPZ: {path}")
        except Exception as e:
            print(f"[ERROR] Failed writing NPZ: {e}", file=sys.stderr)

    if (args.out_png and not args.no_plot) or args.show:
        try:
            if args.out_png and not args.no_plot:
                png_path = plot_png(res, args.out_png, aggregate="median", xscale="log")
                print(f"[OK] Wrote PNG: {png_path}")
            if args.show:
                import matplotlib.pyplot as plt
                # quick replot for showing
                f, mu, med, sd, mad, n_eff = res.stats_across_windows()
                f2, ell = res.combine_windows("median")
                plt.figure(figsize=(8, 5))
                plt.fill_between(f, med - sd, med + sd, alpha=0.2)
                plt.plot(f, ell, lw=1.0)
                plt.xscale("log")
                plt.grid(True, which="both", ls=":")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("H/V ellipticity")
                plt.title("RayDec ellipticity")
                plt.show()
        except Exception as e:
            print(f"[ERROR] Plot failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
