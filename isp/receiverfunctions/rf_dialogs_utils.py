# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:42:41 2020

@author: olivar
"""
import os
import math
import pickle
import pathlib
import numpy as np

import obspy
import obspy.core
import obspy.taup
import obspy.clients.fdsn
import obspy.geodetics.base

def map_data(path, quick=True):
    
    data_map = {}
    
    for top_dir, sub_dir, files in os.walk(path):
        for file in files:
            if file.endswith('.mseed'):
                full_path_to_file = os.path.join(top_dir, file)
                
                if quick:
                    year = int(file.split('.')[0])
                    jday = int(file.split('.')[1])
                    stnm = file.split('.')[3]
                    chnm = file.split('.')[5]
                else:
                    record_info = obspy.io.mseed.util.get_record_information(full_path_to_file)
                    year = record_info['starttime'].year
                    jday = record_info['starttime'].julday
                    stnm = record_info['station']
                    chnm = record_info['channel']

                if os.stat(full_path_to_file).st_size > 0:
                    data_map.setdefault(stnm, {})
                    data_map[stnm].setdefault(year, {})
                    data_map[stnm][year].setdefault(jday, {})
                    data_map[stnm][year][jday].setdefault(chnm, full_path_to_file)
            
    return data_map

def get_catalog(starttime, endtime, client="IRIS", min_magnitude=5):

    client = obspy.clients.fdsn.client.Client("IRIS")
    catalog_starttime = obspy.UTCDateTime(starttime)
    catalog_endtime = obspy.UTCDateTime(endtime)

    catalog = client.get_events(starttime=catalog_starttime,
                                endtime=catalog_endtime,
                                minmagnitude=min_magnitude)
    
    return catalog

def taup_arrival_times(catalog, stationxml, phase="P", earth_model="iasp91",
                       min_distance_degrees=30, max_distance_degrees=90):
    
    EARTH_RADIUS = 6378137.0
    
    # Read station latitude, longitude and elevation from the xml file
    inv = obspy.core.inventory.read_inventory(stationxml)
    stations = {}
    for ntwk in inv:
        for stn in ntwk:
            stnm = stn.code
            stla = stn.latitude
            stlo = stn.longitude
            stev = stn.elevation
            stations.setdefault(stnm, {'latitude':stla, 'longitude':stlo,
                                       'elevation':stev})
    
    # Compute arrival times using TauPy
    arrivals = {}
    arrivals["stations"] = {}
    arrivals["events"] = {}
    model = obspy.taup.TauPyModel(model=earth_model)    
    for i, event in enumerate(catalog.events):
        ev_origin_info = event.origins[0]
        otime = ev_origin_info.time
        evdp = ev_origin_info.depth/1000
        evlo = ev_origin_info.longitude
        evla = ev_origin_info.latitude
        
        for stnm in stations.keys():
            stlo = stations[stnm]['longitude']
            stla = stations[stnm]['latitude']
            stev = stations[stnm]['elevation']
            
            # Distance, azimuth and back_azimuth for event:
            m_dist, az, back_az = obspy.geodetics.base.gps2dist_azimuth(evla, evlo,
                                                                        stla, stlo)
            #deg_dist = math.degrees(m_dist/EARTH_RADIUS)
            deg_dist = obspy.geodetics.base.kilometers2degrees(m_dist/1000)
            
            arrivals["stations"][stnm] = {"lon":stlo, "lat":stla, "elev":stev}
            
            # Check that deg_dist is inside the desired distance range
            if deg_dist >= min_distance_degrees and deg_dist <= max_distance_degrees:
                atime = model.get_travel_times(source_depth_in_km=evdp,
                                               distance_in_degree=deg_dist,
                                               phase_list=[phase],
                                               receiver_depth_in_km=0.0)
                arrivals["events"].setdefault(i, {})
                arrivals["events"][i].setdefault('event_info', {'latitude':evla, 'longitude':evlo,
                                                      'depth':evdp, 'origin_time':otime})
                arrivals["events"][i].setdefault('arrivals', {})
                arrivals["events"][i]['arrivals'].setdefault(stnm, {})
                arrivals["events"][i]['arrivals'][stnm]["arrival_time"] = atime[0].time
                arrivals["events"][i]['arrivals'][stnm]["distance"] = deg_dist
                arrivals["events"][i]['arrivals'][stnm]["incident_angle"] = atime[0].incident_angle
                arrivals["events"][i]['arrivals'][stnm]["azimuth"] = az
                arrivals["events"][i]['arrivals'][stnm]["back_azimuth"] = back_az
                arrivals["events"][i]['arrivals'][stnm]["ray_parameter"] = atime[0].ray_param

    return arrivals

def cut_earthquakes(data_map, arrivals, time_before, time_after, min_snr,
                    stationxml, output_dir, noise_wlen=300,
                    remove_response=True, pre_filt=[1/200, 1/100, 45, 50],
                    rotation='LQT'):

    inv = obspy.core.inventory.read_inventory(stationxml)
    
    for eq in arrivals["events"].keys():
        otime = arrivals["events"][eq]['event_info']['origin_time']
        year = otime.year

        for stnm in data_map.keys():#arrivals["events"][eq]['arrivals'].keys():
            
            if not stnm in arrivals["events"][eq]['arrivals'].keys():
                continue

            # Check if there is data for that year; else continue with next station
            if not year in data_map[stnm].keys():
                continue
            
            atime = otime + arrivals["events"][eq]['arrivals'][stnm]['arrival_time']
            inc = arrivals["events"][eq]['arrivals'][stnm]['incident_angle']
            baz = arrivals["events"][eq]['arrivals'][stnm]['back_azimuth']  

            # Determine start and end times for trimming
            stime = atime - time_before
            etime = atime + time_after            
            
            # Attempt to retrieve data
            stime_jday = stime.julday
            etime_jday = etime.julday
            
            # Check if there is data for this julday; else continue with next station
            if not stime_jday in data_map[stnm][year].keys() or not etime_jday in data_map[stnm][year].keys():
                continue
            
            stream = obspy.core.stream.Stream()
            if stime_jday == etime_jday:
                channels = data_map[stnm][year][stime_jday]
                for chn in channels.keys():
                    stream += obspy.read(channels[chn])
            else:
                channels1 = data_map[stnm][year][stime_jday]
                channels2 = data_map[stnm][year][etime_jday]
                for chn in channels1.keys():
                    stream += obspy.read(channels1[chn])
                for chn in channels2.keys():
                    stream += obspy.read(channels2[chn])
            
            stream.merge(fill_value=0)
            noise = stream.copy()
            
            # Trim data and a noise window
            if stime < stream[0].stats.starttime or etime > stream[0].stats.endtime:
                continue
            
            noise.trim(starttime=stime - noise_wlen, endtime=stime)
            noise.detrend(type="demean")
            noise.detrend(type="linear")           
            stream.trim(starttime=stime, endtime=etime)
            stream.detrend(type="demean")
            stream.detrend(type="linear")
            
            # Compute variance and check if the SNR criterium is fulfilled
            noise_var = np.var(noise.select(component="Z")[0].data)
            signal_var = np.var(stream.select(component="Z")[0].data)
            snr = (signal_var - noise_var)/noise_var
            
            if snr < min_snr:
                continue
            
            # Remove response and rotate
            if remove_response:
                try:
                    stream.remove_response(inventory=inv, pre_filt=pre_filt, output="DISP")
                except ValueError:
                    continue

            stream.rotate('->ZNE', inventory=inv)
            stream.rotate('ZNE->{}'.format(rotation), back_azimuth=baz, inclination=inc)
            
            # Write file to disk
            stn_output_dir = os.path.join(output_dir, stnm)
            pathlib.Path(stn_output_dir).mkdir(parents=True, exist_ok=True)
            stream.write(os.path.join(stn_output_dir,"EQ{}_{}.mseed".format(eq, stnm)), format="MSEED")

