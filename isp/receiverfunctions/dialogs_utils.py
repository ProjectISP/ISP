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

def map_data(path):
    # not implemented
    pass
    
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
            m_dist, az, back_az = obspy.geodetics.base.gps2dist_azimuth(stla, stlo,
                                                                        evla, evlo) # MAL SEGUN IRIS, hay que ponerlo al revés
            deg_dist = math.degrees(m_dist/EARTH_RADIUS)
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

def cut_earthquakes(data_map, arrivals, inventory, time_before=10,
                    time_after=90, min_snr=2.5, noise_window_length=300,
                    output_dir="earthquakes"):
    
    inv = obspy.core.inventory.read_inventory(inventory)
    
    for eq in arrivals.keys():
        otime = arrivals["events"][eq]['event_info']['origin_time']
        year = otime.year
        jday = otime.julday

        for stnm in arrivals["events"][eq]['arrivals'].keys():
        # Check data availability:
            try:
                channels = data_map[stnm][year][jday]
            except KeyError:
                print("Warning: no data available for station {}. Event ".format(stnm) +
                      "origin time {}".format(otime))
                continue
            atime = otime + arrivals["events"][eq]['arrivals'][stnm]['arrival_time']
            inc = arrivals["events"][eq]['arrivals'][stnm]['incident_angle']
            az = arrivals["events"][eq]['arrivals'][stnm]['azimuth']
            # Determine start and end times for trimming
            cut_stime = atime - time_before
            cut_etime = atime + time_after
            # We will cut a segment of noise to estimate SNR
            noise_stime = cut_stime - noise_window_length
            noise_etime = cut_stime
            # Read the data for all channels
            stream = obspy.core.stream.Stream()
            for chnm in channels.keys():
                stream += obspy.read(channels[chnm], format="MSEED")
                stream_stime = stream[0].stats.starttime
                stream_etime = stream[-1].stats.endtime

                try:
                    if cut_stime < stream_stime:
                            stream2 = obspy.read(data_map[stnm][year][jday-1][chnm])
                            stream = stream2 + stream                     
                    elif cut_etime > stream_etime:
                        stream3 = obspy.read(data_map[stnm][year][jday+1][chnm])
                        stream = stream + stream3
                except KeyError:
                    print("Warning: data only partially available for station {}. Event ".format(stnm) +
                          "origin time {}".format(otime))
                    continue
            
            # Trim earthquake and ambient noise
            earthquake = stream.copy().trim(starttime=cut_stime, endtime=cut_etime)
            noise = stream.copy().trim(starttime=noise_stime, endtime=noise_etime)
            earthquake.merge(fill_value=0)
            earthquake.detrend(type='demean')
            earthquake.detrend(type='linear')
            noise.merge(fill_value=0)
            noise.detrend(type='demean')
            noise.detrend(type='linear')
            
            # Compute noise and earthquake variances
            try:
                var_eq = np.var(earthquake.select(component="Z")[0].data)
                var_noise = np.var(noise.select(component="Z")[0].data)
            except IndexError:
                    print("Warning: no Z channel data available for station {}. Event ".format(stnm) +
                          "origin time {}".format(otime))     
                    continue

            try:
                snr = (var_eq - var_noise)/var_noise
            except FloatingPointError:
                continue
            
            if snr > min_snr:
                pre_filt = [1/200, 1/100, 45, 50]
                try:
                    earthquake.remove_response(inventory=inv, pre_filt=pre_filt, output="DISP")
                except ValueError:
                    print("No matching response information found for station {}, event origin time {}".format(
                        stnm, otime))
                    continue
                try:
                    earthquake.rotate(method='ZNE->LQT', back_azimuth=az, inclination=inc) # COMO ESTAMOS AL REVÉS SEGÚN IRIS, HAY QUE ARREGLARLO Y PONER BAZ
                except ValueError:
                    print("Components of station {} have different time spans after trimming, event".format(stnm)+
                          " origin time {}".format(otime))
                    continue
                stn_output_dir = os.path.join(output_dir, stnm)
                pathlib.Path(stn_output_dir).mkdir(parents=True, exist_ok=True)
                if len(earthquake) == 3:
                    earthquake.write(os.path.join(stn_output_dir,"EQ{}_{}.mseed".format(eq, stnm)), format="MSEED")
            else:
                print("Station {} does not fullfill SNR criterium for event with origin time {}".format(
                    stnm, otime))

