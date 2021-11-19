#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Feb  6 12:03:39 2018
    @author: robertocabieces
"""
import obspy
import nvector as nv
from isp.Gui.Frames import MessageDialog


class retrieve:

    def __init__(self):
        """
                Dedicater to support retrieve data


                :param No params required to initialize the class
        """

    def get_inventory(self, url, starttime, endtime, networks, stations, FDSN=True, use_networks = False,
                      use_stations = False, **kwargs):
        inventory = None
        client = None
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

            print("Coundn't be conected, Please check your internet connection and try another time")

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

    @staticmethod
    def get_circle(lat1,lon1):
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat30 = []
        lon30 = []
        lat90 = []
        lon90 = []

        frame = nv.FrameE(a=6371e3)
        pointA = frame.GeoPoint(latitude=lat1, longitude=lon1, degrees=True)

        for az in range(360):

            pointB, _azimuthb = pointA.displace(distance=30*112000, azimuth=az, degrees=True)
            pointC, _azimuthC = pointA.displace(distance=90*112000, azimuth=az, degrees=True)
            lat2, lon2 = pointB.latitude_deg, pointB.longitude_deg
            lat3, lon3 = pointC.latitude_deg, pointC.longitude_deg

            lat30.append(lat2)
            lat90.append(lat3)
            lon30.append(lon2)
            lon90.append(lon3)

        return lat30, lon30, lat90, lon90
