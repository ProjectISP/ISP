from obspy import Inventory
from obspy.core.event import Origin



class StationUtils:
    """
    Utility class for handling station data and metadata.
    """
    @staticmethod
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
                            station_dict[station_code] = [station.longitude, station.latitude]
                            break  # Stop searching once the station is found

        return station_dict