#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
readNLL



:param : 
:type : 
:return: 
:rtype: 
"""
from obspy import read_events, read_inventory


def get_station_location_dict(origin, inventory):
    """
    Creates a dictionary mapping station names to their latitude and longitude.

    Parameters:
    origin (obspy.core.event.origin.Origin): The Origin object containing pick information.
    inventory (obspy.core.inventory.inventory.Inventory): The Inventory object with station metadata.

    Returns:
    dict: A dictionary where keys are station names, and values are [latitude, longitude].
    """
    station_dict = {}

    # Extract station codes from the Origin's picks
    for pick in origin.picks:
        if pick.waveform_id and pick.waveform_id.station_code:
            station_code = pick.waveform_id.station_code

            # Search for the station in the Inventory
            for network in inventory:
                for station in network:
                    if station.code == station_code:
                        # Add the station's name as key, and [latitude, longitude] as value
                        station_dict[station_code] = [station.latitude, station.longitude]
                        break  # Stop searching once the station is found
    print(station_dict)
    return station_dict


path = "/Users/admin/Documents/iMacROA/ISP/Test/test_data/last.hyp"
inv_path = "/Users/admin/Documents/iMacROA/ISP/isp/Metadata/xml/metadata.xml"
catalog = read_events(path)
inventory = read_inventory(inv_path)
for event in catalog:
    #origin = event.preferred_origin() or event.origins[0]
    get_station_location_dict(event, inventory)