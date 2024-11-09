#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

mseed2sac

"""


import os
from obspy import read, read_inventory
from obspy.geodetics import gps2dist_azimuth
from obspy.io.sac import SACTrace


def sac_mapping(folder_path, output_path, hipo: dict, metadata_path:str):

    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    latitude = hipo["latitude"]
    longitude = hipo["longitude"]
    depth_km = hipo["depth_km"]
    inventory = read_inventory(metadata_path)
    for file in file_list:
        try:
            tr = read(file)[0]
            inv = inventory.select(network=tr.stats.network, station=tr.stats.station, channel=tr.stats.channel)
            station_coordinates = inv.get_coordinates(tr.id)
            lat_sta = station_coordinates['latitude']
            lon_sta = station_coordinates['longitude']
            dist, bazim, azim = gps2dist_azimuth(lat_sta, lon_sta, latitude,
                                                 longitude)

            # SAC header requires start time components
            starttime = tr.stats.starttime
            year = starttime.year
            julday = starttime.julday
            hour = starttime.hour
            minute = starttime.minute
            second = starttime.second
            msec = int(starttime.microsecond / 1000)  # Convert microseconds to milliseconds

            N = len(tr.data)
            header = {
                'b': 0, 'npts': N, 'kstnm': tr.stats.station,
                'kcmpnm': tr.stats.channel,
                'stla': lat_sta, 'stlo': lon_sta,
                'evla': latitude, 'evlo': longitude,
                'evdp': depth_km,
                'delta': tr.stats.delta, 'dist': dist * 1E-3,
                'baz': bazim, 'az': azim,
                'nzyear': year, 'nzjday': julday,
                'nzhour': hour, 'nzmin': minute,
                'nzsec': second, 'nzmsec': msec
            }

            # Create a unique SAC file name
            name = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}.{julday}.{year}"
            path = os.path.join(output_path, name + ".sac")

            # Create SACTrace object and write to file
            sac = SACTrace(data=tr.data, **header)
            sac.write(path, byteorder='little', flush_headers=True)

        except:
            pass


metadata_path = '/Volumes/LaCie/UPFLOW_5HZ/metadata_upflow/meta.xml'
files_path = '/Volumes/LaCie/surface_waves/FK_test/E2_full'
output_path = '/Volumes/LaCie/surface_waves/FK_test/Ev2_sac'

# coords ={}
# coords["latitude"] = 55.36
# coords["longitude"] = -157.89
# coords["depth_km"] = 35.0


coords ={}

coords["latitude"] = -58.45
coords["longitude"] = -25.33
coords["depth_km"] = 55.73


sac_mapping(files_path, output_path, coords, metadata_path)
