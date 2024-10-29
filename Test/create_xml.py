import pandas as pd
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy import UTCDateTime

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("stations.csv")

# Create an empty inventory
inventory = Inventory(networks=[], source="Your Organization")

# Loop through each row in the DataFrame
for _, row in df.iterrows():
    # Parse station metadata from each row
    network_code = row["Network"]
    station_code = row["Station"]
    channel_code = row["Channel"]
    latitude = row["Latitude"]
    longitude = row["Longitude"]
    elevation = row["Elevation"]

    # Check if the network already exists in the inventory
    network = next((net for net in inventory if net.code == network_code), None)
    if not network:
        # If the network doesn't exist, create it and add to inventory
        network = Network(code=network_code)
        inventory.networks.append(network)

    # Create the station object
    station = Station(
        code=station_code,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        site=Site(name=station_code),
        creation_date=UTCDateTime.now(),
    )

    # Create the channel object
    channel = Channel(
        code=channel_code,
        location_code="00",  # Default location code
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        depth=0.0,  # Depth of 0 if it's a surface station
        sample_rate=40.0  # Sample rate, customize as needed
    )

    # Attach the channel to the station and the station to the network
    station.channels.append(channel)
    network.stations.append(station)

# Write the inventory to a StationXML file
inventory.write("stations.xml", format="stationxml")
print("StationXML file created successfully as 'stations.xml'")
