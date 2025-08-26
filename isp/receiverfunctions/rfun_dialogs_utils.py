# -*- coding: utf-8 -*-
"""
This file is part of Rfun, a toolbox for the analysis of teleseismic receiver
functions.

Copyright (C) 2020-2025 Andrés Olivar-Castaño

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
For questions, bug reports, or to suggest new features, please contact me at
olivar.ac@gmail.com.
"""

import os
import numpy as np

from pathlib import Path

import obspy
import obspy.core
import obspy.taup
import obspy.clients.fdsn
import obspy.geodetics.base
import obspy.io.mseed.util
import math
import obspy.taup.taup_create

import h5py

from isp.receiverfunctions.definitions import ROOT_DIR, CONFIG_PATH
from PyQt5.QtWidgets import QApplication

def map_data(top_directory, format_):
    
    data_map = {}

    # TODO: añadir cuadros de dialogo describiendo posibles errores. Por ejemplo
    # TODO: si los ficheros no tienen extension, no los lee
    # TODO: outputear un log de error
    for path in Path(top_directory).rglob("*"):
        str_path = str(path.absolute())
        try:
            record_info = obspy.read(str_path, headonly=True)
        except Exception as e:
            print(e)
            print("The format of file {} could not be recognized.".format(str_path))
            continue
        year = record_info[0].stats.starttime.year
        jday = record_info[0].stats.starttime.julday
        stnm = record_info[0].stats.station
        chnm = record_info[0].stats.channel
        
        if record_info[0].stats.npts > 0:
            data_map.setdefault(stnm, {})
            data_map[stnm].setdefault(year, {})
            data_map[stnm][year].setdefault(jday, {})
            data_map[stnm][year][jday].setdefault(chnm, str_path)
            
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
        
        mags = []
        for mag in event.magnitudes:
            if "mw" in mag.magnitude_type.lower():
                mags = mag.mag
                break
            else:
                mags.append(mag.mag)
        
        # If there is no Mw magnitude take the biggest one
        if type(mags) == list:
            mags = np.max(mags)

        try:
            evdp = ev_origin_info.depth/1000
        except TypeError:
            evdp = 0
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
                                                      'depth':evdp, 'origin_time':otime,
                                                      'magnitude':mags})
                arrivals["events"][i].setdefault('arrivals', {})
                arrivals["events"][i]['arrivals'].setdefault(stnm, {})
                arrivals["events"][i]['arrivals'][stnm]["arrival_time"] = atime[0].time
                arrivals["events"][i]['arrivals'][stnm]["distance"] = deg_dist
                arrivals["events"][i]['arrivals'][stnm]["incident_angle"] = atime[0].incident_angle
                arrivals["events"][i]['arrivals'][stnm]["azimuth"] = az
                arrivals["events"][i]['arrivals'][stnm]["back_azimuth"] = back_az
                arrivals["events"][i]['arrivals'][stnm]["ray_parameter"] = atime[0].ray_param

    return arrivals

def parse_event_picks(event):
    picks = {}
    for pick in event.picks:
        picks.setdefault(pick.phase_hint.lower(), {})
        picks[pick.phase_hint.lower()][pick.waveform_id.station_code] = pick.time
    if not picks:
        picks = {"p":{}, "s":{}}
    return picks

def azimuth(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = [np.radians(x) for x in [lat1, lon1, lat2, lon2]]
    londiff = lon2 - lon1
    az = np.arctan2(np.sin(londiff) * np.cos(lat2), np.cos(lat1)*np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(londiff))
    az = (az*180/math.pi + 360) % 360
    return az

def distance_deg(stla, stlo, evla, evlo):
    stla, stlo, evla, evlo = [np.radians(x) for x in [stla, stlo, evla, evlo]]
    difflat = evla - stla
    difflon = evlo - stlo
    a = np.sin(difflat/2)**2 + np.cos(stla) * np.cos(evla) * np.sin(difflon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return np.degrees(c)

def parse_taup_arrivals(arrivals):
    phases = {}
    for arrival in arrivals:
        phases[arrival.name.lower()] = {"time":arrival.time, "incident_angle":arrival.incident_angle, "ray_param":arrival.ray_param_sec_degree}
    return phases

def cut_earthquakes(data_map, catalog, time_before, time_after, min_snr,
                    min_magnitude, min_distance_degrees, max_distance_degrees, min_depth, max_depth,
                    stationxml, output_dir, earth_model, custom_earth_models, remove_response, units_output,
                    corner_frequencies, water_level, noise_wlen=300, noise_before_P=10,
                    pre_filt=[1/200, 1/100, 45, 50],
                    phase="p", use_picks=False, pbar=None):
    
    pbar.setRange(0, len(catalog))
    pbar.setValue(0)
    
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

    # Instance of TauPyModel
    if earth_model in custom_earth_models:
        try:
            model = obspy.taup.TauPyModel(model=earth_model)
        except FileNotFoundError:
            obspy.taup.taup_create.build_taup_model(ROOT_DIR + "/earth_models/{}.tvel".format(earth_model))
            model = obspy.taup.TauPyModel(model=earth_model)
    else:
        model = obspy.taup.TauPyModel(model=earth_model)
    
    inv = obspy.core.inventory.read_inventory(stationxml)
    
    # Fetch station locations from the inventory
    stations = {}
    for ntwk in inv:
        for stn in ntwk:
            stnm = stn.code
            stla = stn.latitude
            stlo = stn.longitude
            stev = stn.elevation
            stations.setdefault(stnm, {'latitude':stla, 'longitude':stlo,
                                       'elevation':stev})
    
    # Initialize h5py file
    if os.path.isfile(output_dir):
        mode = "a"
    else:
        mode = "w"
    hdf5 = h5py.File(output_dir, mode)   

    
    # Loop over events
    for i, event in enumerate(catalog):
        pbar.setValue(i)
        QApplication.processEvents()
        event_magnitude = event.preferred_magnitude()
        summary_card = str(event).split("\n")[0].split("Event:")[1]
        if type(event_magnitude) is type(None):
            # If there is no preferred magnitude, we take the first one
            try:
                event_magnitude = event.magnitudes[0]
            except IndexError:
                print("No magnitude defined for event {}, skipping...".format(summary_card))
                continue

        mag = event_magnitude.mag
        if mag < min_magnitude:
            print("Magnitude of event {} below threshold, skipping...".format(summary_card))
            continue

        # Try to get preferred origin info
        event_origin_info = event.preferred_origin()
        if type(event_origin_info) is type(None):
            # If there is no preferred origin, we take the first one
            try:
                event_origin_info = event.origins[0]
            except IndexError:
                print("No origin defined for event {}, skipping...".format(summary_card))
                continue # If there is no origin info, skip to the next event

        depth = event_origin_info.depth
        summary_card = summary_card + " | Depth: {} km".format(depth)
        if min_depth == max_depth == 0:
            if i == 0:
                print("Both min_depth and max_depth are set to 0. No depth constraints are applied...")
        else:
            if min_depth > max_depth and i==0:
                print("WARNING: min_depth is higher than max_depth, switching the values around...")
                max_depth, min_depth = min_depth, max_depth
            
            if depth < min_depth or depth > max_depth:
                print("Event {} outside of allowed depth range. Skipping...".format(summary_card))
                continue
        
        # Fetch location info
        evla, evlo, evdp, otime = (event_origin_info.latitude, event_origin_info.longitude,
                                   event_origin_info.depth, event_origin_info.time)

        for stnm in data_map.keys():
            # Retrieve station information
            stla, stlo, stel = (stations[stnm]["latitude"], stations[stnm]["longitude"],
                                stations[stnm]["elevation"])
            back_az = azimuth(stla, stlo, evla, evlo)
            deg_dist = distance_deg(stla, stlo, evla, evlo)
            
            # Check event distance
            if deg_dist < min_distance_degrees or deg_dist > max_distance_degrees:
                print("Event {} outside of distance threshold for station {}, skipping...".format(summary_card, stnm))
                continue

            # We always trace the rays with TauP, even if we are using the picks
            # in the catalog, because we need an estimation for the ray parameter
            # and inclination angles
            arrivals = []
            
            p_arrivals = model.get_travel_times(source_depth_in_km=evdp/1000,
                                                distance_in_degree=deg_dist,
                                                phase_list=["P", "p"],
                                                receiver_depth_in_km=0)
            if len(p_arrivals) == 0:
                print("TauP traveltime computation failed for phase p, station {}, event {}. Skipping...".format(stnm, summary_card))
                continue
            else:
                arrivals.append(p_arrivals[0])
            
            if phase == "s":
                s_arrivals = model.get_travel_times(source_depth_in_km=evdp/1000,
                                                    distance_in_degree=deg_dist,
                                                    phase_list=["S", "s"],
                                                    receiver_depth_in_km=0)
                if len(s_arrivals) == 0:
                    print("TauP traveltime computation failed for phase s, station {}, event {}. Skipping...".format(stnm, summary_card))
                    continue
                else:
                    arrivals.append(s_arrivals[0]) # take the first arrival
                

            phases = parse_taup_arrivals(arrivals)
            # We need the p arrival incident angle in order to be able to rotate to P-SV
            incident_angle = phases["p"]["incident_angle"]
            ray_param = phases["p"]["ray_param"]
            
            # Now we get the start and end times for the rf and nosie segments
            starttime = otime + phases[phase]["time"] - time_before
            endtime = otime + phases[phase]["time"] + time_after
            atime = phases[phase]["time"]
            
            noise_starttime = otime + phases["p"]["time"] - noise_before_P - noise_wlen
            noise_endtime = otime + phases["p"]["time"] - noise_before_P
            
            # Now we read the data
            try:
                st = obspy.core.stream.Stream()
                if noise_starttime.julday == endtime.julday:
                    channels = data_map[stnm][noise_starttime.year][noise_starttime.julday]
                    for chn in channels.keys():
                        st += obspy.read(channels[chn])
                else:
                    channels1 = data_map[stnm][noise_starttime.year][noise_starttime.julday]
                    channels2 = data_map[stnm][endtime.year][endtime.julday]
                    for chn in channels1.keys():
                        st += obspy.read(channels1[chn])
                    for chn in channels2.keys():
                        st += obspy.read(channels2[chn])
            except KeyError:
                print("No available data for station {} and event {}, skipping...".format(stnm, summary_card))
                continue
            
            # Detrend, merge
            st.detrend(type="constant")
            st.detrend(type="linear")
            st.merge(fill_value=0)
            st.taper(max_percentage=0.05)
            
            if st[0].stats.starttime > starttime or st[0].stats.endtime < endtime:
                print("Incomplete data for station {} and event {}, skipping...".format(stnm, summary_card))
                continue
            
            # Trim
            signal = st.copy()
            signal.trim(starttime=starttime, endtime=endtime)
            noise = st.copy()
            noise.trim(starttime=noise_starttime, endtime=noise_endtime)
            
            # Remove response/rotate
            for segment in [signal, noise]:
                if remove_response:
                    try:
                        print(pre_filt, units_output, water_level)
                        segment.remove_response(inventory=inv, pre_filt=pre_filt, output=units_output, water_level=water_level)
                    except:
                        print("WARNING: No response information found for station {}, event {}. Skipping...".format(stnm, summary_card))
                        continue
                
                # In case the station is not rotated to ZNE, try using the inventory
                components = sorted([tr.stats.channel[-1] for tr in st])
                if components != ["E", "N", "Z"]:
                    try:
                        segment.rotate(method="->ZNE", inventory=inv)
                    except:
                        print("Could not rotate station {} to ZNE. Skipping...".format(stnm))
                        continue
            
            # Check the length of the traces
            expected_len_signal = int((time_before + time_after)/signal[0].stats.delta)
            expected_len_noise = int((noise_wlen)/signal[0].stats.delta)
            
            for tr in signal:
                tr.detrend(type="constant")
                tr.detrend(type="linear")
                if len(tr.data) < expected_len_signal:
                    tr.data = np.pad(tr.data, (0, expected_len_signal - len(tr.data)))
                elif len(tr.data) > expected_len_signal:
                    tr.data = tr.data[:expected_len_signal]
            
            for tr in noise:
                tr.detrend(type="constant")
                tr.detrend(type="linear")
                if len(tr.data) < expected_len_noise:
                    tr.data = np.pad(tr.data, (0, expected_len_noise - len(tr.data)))
                elif len(tr.data) > expected_len_noise:
                    tr.data = tr.data[:expected_len_noise]
            
            # Check that there are no traces full of zeroes
            all_zeroes = []
            for tr in signal:
                if np.all(tr.data == 0):
                    all_zeroes.append(True)
                else:
                    all_zeroes.append(True)
            
            if np.any(all_zeroes):
                print("One or more traces contains only zeros. Station {}, event {}. Skipping...".format(stnm, summary_card))
            
            # Compare RMS
            rms_signal = np.sqrt(np.mean(np.square(signal.select(component="Z")[0].data)))
            rms_noise = np.sqrt(np.mean(np.square(noise.select(component="Z")[0].data)))
            RMS_ratio = rms_signal/rms_noise
            if RMS_ratio < min_snr:
                print("RMS ratio below threshold for station {}, event {}".format(stnm, summary_card))
            
            # RMS computation. It does not perform very well for discarding low quality data
            # Use the vertical channel
            signal_rms = np.sqrt(np.mean(np.square(signal.copy().select(component="Z")[0].data)))
            noise_rms = np.sqrt(np.mean(np.square(noise.copy().select(component="Z")[0].data)))
            snr = signal_rms/noise_rms            
            
            if snr <= min_snr:
                print("Signal to noise ratio for station {} and event {} below threshold, skipping...".format(stnm, summary_card))
                continue



            # it = iter(eq_data)
            # the_len = len(next(it))
            # if not all(len(l) == the_len for l in it):
            #     print("Warning: mismatch in the number of samples of the different components. Event {}, station {}. Correct by trimming or padding with zeros...".format(summary_card, stnm))
            #     for tr in eq_data:
            #         if len(tr) < the_len:
            #             tr = np.pad(tr, (0, the_len - len(tr)))
            #         else:
            #             tr = tr[:the_len]

            eq_data = [tr.data for tr in signal]
            time = np.arange(0, len(eq_data[0])*signal[0].stats.delta, signal[0].stats.delta)
            eq_data.append(time)

            g = hdf5.require_group(stnm)
            # These should not be modified if the group already exists, to be changed
            g.attrs["stla"] = stations[stnm]["latitude"]
            g.attrs["stlo"] = stations[stnm]["longitude"]
            g.attrs["stel"] = stations[stnm]["elevation"]
            d = g.create_dataset(str(i), data=np.array(eq_data))
            d.attrs["data_structure"] = [tr.stats.channel[-1] for tr in signal] + ["time"]
            d.attrs["otime"] = float(otime)
            d.attrs["atime"] = float(atime)
            d.attrs["time_before_phase_onset"] = time_before
            d.attrs["dist_degrees"] = deg_dist
            try:
                d.attrs["mag"] = event.preferred_magnitude().mag
            except AttributeError:
                d.attrs["mag"] = event.magnitudes[0].mag
            d.attrs["evlo"] = evlo
            d.attrs["evla"] = evla
            d.attrs["evdp"] = evdp
            d.attrs["baz"] = back_az
            d.attrs["incident_angle"] = incident_angle
            d.attrs["ray_param"] = ray_param
            d.attrs["phase"] = phase
            d.attrs["SNR"] = snr

    if isinstance(hdf5, h5py.File):   # Just HDF5 files
        try:
            hdf5.close()
        except:
            pass

