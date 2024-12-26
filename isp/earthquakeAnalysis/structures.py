#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structures

"""
from dataclasses import dataclass
from isp.Structures import BaseDataClass


@dataclass
class GridConfiguration(BaseDataClass):
    latitude: float
    longitude: float
    depth: float
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    geo_transformation: str
    grid_type: str
    path_to_1d_model: str
    path_to_3d_model: str
    path_to_picks: str
    p_wave_type: bool
    s_wave_type: bool
    model: str


@dataclass
class TravelTimesConfiguration(BaseDataClass):
    distance_limit: float
    grid: str


@dataclass
class LocationParameters(BaseDataClass):
    search: str
    method: str


@dataclass
class NLLConfig(BaseDataClass):

    grid_configuration: GridConfiguration
    travel_times_configuration: TravelTimesConfiguration
    location_parameters: LocationParameters