#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mass_donwloader



:param : 
:type : 
:return: 
:rtype: 
"""

from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader
from datetime import datetime, timedelta

# Parameters
networks = ["IV", "IU"]  # List of networks
stations = ["ATMI", "ATFO"]  # List of stations
channels = ["HHN", "HHE", "HHZ"]  # List of channels
start_date = "2023-01-01"  # Start date (YYYY-MM-DD)
end_date = "2023-01-07"  # End date (YYYY-MM-DD)

# Convert start_date and end_date to datetime objects
start_date = datetime.strptime(start_date, "%Y-%m-%d")
end_date = datetime.strptime(end_date, "%Y-%m-%d")

# Loop through each day
current_date = start_date
while current_date <= end_date:
    next_date = current_date + timedelta(days=1)

    # Define the time window for this day
    starttime = current_date
    endtime = next_date

    # Loop over each network and station
    for network in networks:
        for station in stations:
            # Set up the domain and restrictions
            domain = CircularDomain(latitude=0, longitude=0, minradius=0, maxradius=180)  # Global search
            restrictions = Restrictions(
                starttime=starttime,
                endtime=endtime,
                network=network,
                station=station,
                channel=",".join(channels),
                location="*",  # Include all location codes
                reject_channels_with_gaps=False,
                minimum_length=0.0,
                sanitize=True,
                filename_template=f"{network}_{station}_%(channel)s_{current_date.strftime('%Y-%m-%d')}.mseed"
            )

            # Initialize MassDownloader
            mdl = MassDownloader()

            # Download data
            try:
                mdl.download(domain, restrictions, mseed_storage="waveforms", stationxml_storage="metadata")
                print(f"Successfully downloaded data for {network}.{station} on {current_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"Failed to download data for {network}.{station} on {current_date.strftime('%Y-%m-%d')}: {e}")

    # Increment the date
    current_date = next_date