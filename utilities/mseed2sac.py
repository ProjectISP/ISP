#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

mseed2sac

"""

import os
from obspy import read, read_inventory, UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.io.sac import SACTrace


def sac_mapping(folder_path, output_path, hipo: dict, metadata_path: str, origin_time: UTCDateTime):
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
            dist, bazim, azim = gps2dist_azimuth(lat_sta, lon_sta, latitude, longitude)

            # Calculate time offsets
            starttime = tr.stats.starttime
            origin_offset = (starttime - origin_time)

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
                'nzsec': second, 'nzmsec': msec,
                'o': origin_offset  # Add origin time offset
            }

            # Create a unique SAC file name
            name = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}.{julday}.{year}"
            path = os.path.join(output_path, name + ".sac")

            # Create SACTrace object and write to file
            sac = SACTrace(data=tr.data, **header)
            sac.write(path, byteorder='little', flush_headers=True)

        except Exception as e:
            print(f"Failed to process {file}: {e}")


# Parameters
metadata_path = '/Volumes/LaCie/UPFLOW_5HZ/metadata_upflow/meta.xml'
files_path = '/Volumes/LaCie/UPFLOW_5HZ/Tele_events/29_07_2021_M8'
output_path = '/Volumes/LaCie/UPFLOW_5HZ/Tele_events/29_07_2021_M8/SAC'

coords = {
    "latitude": 55.3635,
    "longitude": -157.8876,
    "depth_km": 35.0
}

# Define the event origin time (replace with your event's actual origin time)
origin_time = UTCDateTime("2021-07-29T06:15:49")

# Call the function
sac_mapping(files_path, output_path, coords, metadata_path, origin_time)