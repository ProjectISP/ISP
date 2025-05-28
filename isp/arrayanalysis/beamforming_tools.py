#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

beamforming_tools, designed for MUSIC algorythm

"""

import math
from scipy.linalg import eigh
import numpy as np
from obspy.core import AttribDict
from scipy.signal import hilbert


def find_nearest(array, value):
    idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
    return idx, val

def extract_coordinates(st, inv):
    R = 6371.0  # Earth radius in km
    n = len(st)
    coords_def = {}
    lats = []
    lons = []

    for i in range(n):
        coords = inv.get_coordinates(st[i].id, st[i].stats.starttime)
        lat = coords['latitude']
        lon = coords['longitude']

        st[i].stats.coordinates = AttribDict({
            'latitude': lat,
            'elevation': coords['elevation'],
            'longitude': lon
        })

        coords_def[st[i].id] = [lat, lon]
        lats.append(lat)
        lons.append(lon)

    # Compute barycenter
    barycenter_lat = np.mean(lats)
    barycenter_lon = np.mean(lons)

    # Convert lat/lon to relative x, y in km
    relative_coords_km = {}
    for key, (lat, lon) in coords_def.items():
        delta_lat = math.radians(lat - barycenter_lat)
        delta_lon = math.radians(lon - barycenter_lon)
        mean_lat_rad = math.radians((lat + barycenter_lat) / 2.0)

        dx = R * delta_lon * math.cos(mean_lat_rad)  # East-West (x)
        dy = R * delta_lat  # North-South (y)

        relative_coords_km[key] = (dx, dy)

    return relative_coords_km


def get_covariance_matrix(data_array):

    R = np.dot(data_array, data_array.T.conj()) / data_array.shape[1]
    print("Covariance Matrix (Hilbert):", R)
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        raise ValueError("Covariance matrix unstable")
    return R

def convert_to_array(st, inv):

    relative_coords_km = extract_coordinates(st, inv)
    n = len(st)
    if n == 0:
        raise ValueError("No hay trazas para convertir")
    npts = st[0].stats.npts
    data_array = np.zeros((n, npts), dtype=complex)
    index_list = []
    for i, tr in enumerate(st):
        if len(tr.data) != npts:
            raise ValueError(f"Todas las trazas deben tener igual longitud. Traza {tr.id} tiene {len(tr.data)} muestras")
        data_array[i, :] = hilbert(tr.data)
        index_list.append(tr.id)
    print(f"Array convertido: {data_array.shape} (sensores x muestras)")

    sensor_positions = []
    for tr_id in index_list:
        x, y = relative_coords_km[tr_id]
        sensor_positions.append([x, y])
        print(f"Sensor {tr_id}: Este={x:.3f} km, Norte={y:.3f} km")

    return data_array, sensor_positions

def get_noise_subspace(R, n_signals):
    eigvals, eigvecs = eigh(R)
    print("Eigen_Values:", eigvals)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    signal_vec = eigvecs[:, :n_signals]
    print(f"Vector(es) propio(s) dominante(s) (subespacio de señal) para n_signals={n_signals}:", signal_vec)
    En = eigvecs[:, n_signals:]
    print("Shape of En:", En.shape)
    return En, signal_vec

def make_steering_cartesian(positions, slowness, freq):

    """
    positions: (N_sensors, 2)
    slowness_vec: (2,) - [sx, sy]
    freq: scalar frequency in Hz
    """
    delays = positions @ slowness

    return np.exp(-2j * np.pi * freq * delays)

def compute_music_spectrum(positions,  En, slow_lim, sinc=100, freq_range = (0.8, 1.2)):

    # Define Cartesian slowness grid (sx, sy)
    sx = np.arange(-1*slow_lim, slow_lim, sinc)[np.newaxis]
    sy = np.arange(-1*slow_lim, slow_lim, sinc)[np.newaxis]
    sx_grid, sy_grid = np.meshgrid(sx, sy)
    music_map = np.zeros_like(sx_grid)

    # Loop over frequencies and sum MUSIC maps
    for freq in freq_range:
        map_f = np.zeros_like(sx_grid)
        for i in range(sx_grid.shape[0]):
            for j in range(sy_grid.shape[1]):
                s_vec = np.array([sx_grid[i, j], sy_grid[i, j]])
                steering = make_steering_cartesian(positions, s_vec, freq)[:, np.newaxis]
                steering = steering / np.linalg.norm(steering)
                proj = steering.T.conj() @ En @ En.T.conj() @ steering
                map_f[i, j] = 1 / (np.abs(proj.item()) + 1e-10)
        music_map += map_f

    # Normalize or average the map
    music_map /= len(freq_range)

    return music_map

def regularize_covariance(R, loading_factor=1E-3):
    M = R.shape[0]
    delta = loading_factor * np.trace(R) / M
    R = R + delta * np.eye(M)
    try:
        P = np.linalg.pinv(R)
    except:
        print("Unstable Matrix Inversion")
        P = np.zeros(M)
    return P

