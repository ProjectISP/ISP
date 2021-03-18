#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Feb  6 12:03:39 2018
    @author: robertocabieces
"""
import obspy

from isp.Gui.Frames import MessageDialog
from isp.Gui.Utils.pyqt_utils import convert_qdatetime_utcdatetime


class retrieve:

    def __init__(self):
        """
                Dedicater to support retrieve data


                :param No params required to initialize the class
        """

    def get_inventory(self, url, starttime, endtime, networks, stations, FDSN=True, use_networks = False,
                      use_stations = False, **kwargs):

        print(FDSN, use_networks, starttime, endtime, networks)
        try:

            if FDSN:

                client = obspy.clients.fdsn.Client(url)

                if len(networks) == 0:

                    inventory = client.get_stations(network="*", station="*", starttime=starttime,
                                                endtime=endtime)
                elif len(networks) > 0:

                    inventory = client.get_stations(network=networks, station="*", starttime=starttime,
                                                    endtime=endtime)
                    print(inventory)

            else:
                ip_address = kwargs.pop('ip')
                port = kwargs.pop('port')
                client = obspy.clients.earthworm.Client(ip_address, int(port))

                if len(networks) == 0 and len(stations) == 0:

                    inventory = client.get_stations(network="*", station="*", starttime=starttime,
                                                    endtime=endtime)
                elif len(networks) > 0 and use_networks:

                    inventory = client.get_stations(network=networks, station="*", starttime=starttime,
                                                    endtime=endtime)

                elif len(networks) > 0 and use_networks and use_stations:

                    inventory = client.get_stations(network=networks, station=stations, starttime=starttime,
                                                    endtime=endtime)
        except:

            md = MessageDialog(self)
            md.set_info_message("Please check your internet conection")
        return inventory, client


    def get_inventory_coordinates(self, inv):
        num_nets = len(inv)
        coordinates = {}

        for net in range(num_nets):

            net_ids = []
            sta_ids = []
            latitude = []
            longitude = []
            net = inv[net]
            num_sta = len(net)
            for sta in range(num_sta):
                sta = net[sta]
                net_ids.append(net.code)
                sta_ids.append(sta.code)
                latitude.append(sta.latitude)
                longitude.append(sta.longitude)
            net_content = [net_ids,sta_ids,latitude,longitude]
            # coordinates.update(net.code,net_content)
            coordinates[net.code] = net_content

        return coordinates

    def get_station_id(self, lon, lat, inv):

        import math
        net_ids = []
        sta_ids = []
        distance = []
        num_nets = len(inv)

        for net in range(num_nets):

            net = inv[net]
            num_sta = len(net)

            for sta in range(num_sta):
                sta = net[sta]
                lat0 = sta.latitude
                lon0 = sta.longitude
                y = (lat-lat0)**2
                x = (lon-lon0)**2
                m = math.sqrt(x+y)
                if m < 3.0:
                    net_ids.append(net.code)
                    sta_ids.append(sta.code)
                    distance.append(m)

        #if len(distance)>0:
        index = distance.index(min(distance))


        #if len(net_ids)>0 and len(sta_ids)>0:

        results = [net_ids[index],sta_ids[index]]

        #else:
        #    results = [net_ids, sta_ids]

        return results

