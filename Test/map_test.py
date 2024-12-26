#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_test



:param : 
:type : 
:return: 
:rtype: 
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from owslib.wms import WebMapService


# Define a function to get WMS information and plot it
def plot_wms_layer():
    # URL of the WMS service
    wms_url = "https://gis.ngdc.noaa.gov/arcgis/services/gebco08_hillshade/MapServer/WMSServer"  # Example WMS server

    # Connect to the WMS service
    wms = WebMapService(wms_url)

    # Display available layers (optional)
    print("Available layers:")
    for layer in wms.contents:
        print(f" - {layer}: {wms[layer].title}")

        # Set up the map
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-10, -6, 32, 34])  # Set map bounds (global view)
        ax.coastlines()  # Add coastlines for reference

        # Add the WMS layer to the map
        ax.add_wms(
            wms_url,
            layer)

        # Set a title and show the map
        plt.title("WMS Layer from Ahocevar Geoserver")
        plt.show()


# Call the function
plot_wms_layer()
