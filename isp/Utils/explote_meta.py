#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from obspy import Inventory


def find_coords(metadata: Inventory):

    sta_names = []
    latitudes = []
    longitudes = []
    networks = {}

    for network in metadata:
        for station in network.stations:
            if station.code not in sta_names:
                sta_names.append(network.code+"."+station.code)
                latitudes.append(station.latitude)
                longitudes.append(station.longitude)
            else:
                pass
        networks[network.code] = [sta_names, longitudes, latitudes]
    print(networks)

    return networks
