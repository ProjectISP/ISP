from obspy.geodetics import gps2dist_azimuth
from isp.Structures.structures import StationCoordinates
import numpy as np
class Indmag:
    def __init__(self, st_deconv, st_wood, pick_info, event_info, inventory):
        self.st_deconv = st_deconv
        self.st_wood = st_wood
        self.inventory = inventory
        self.pick_info = pick_info
        self.event_info = event_info
        self.ML = []

    def cut_waveform(self):
        pickP_time = None
        for item in self.pick_info:
            if item[0] == "P":
                pickP_time = item[1]

        self.st_deconv.trim(starttime=pickP_time-5, endtime=pickP_time+300)
        self.st_wood.trim(starttime=pickP_time-5, endtime=pickP_time + 300)

    def extrac_coordinates_from_station_name(self, inventory, name):
        selected_inv = inventory.select(station=name)
        cont = selected_inv.get_contents()
        coords = selected_inv.get_coordinates(cont['channels'][0])
        return StationCoordinates.from_dict(coords)

    def magnitude_local(self):
        print("Calculating Local Magnitude")
        tr_E = self.st_wood.select(component="E")
        tr_E = tr_E[0]
        tr_N = self.st_deconv.select(component="N")
        tr_N = tr_N[0]
        coords = self.extrac_coordinates_from_station_name(self.inventory, self.st_deconv[0].stats.station)
        dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event_info[1], self.event_info[2])
        dist = dist / 1000
        max_amplitude_N = np.max(tr_N.data)*1e3 # convert to  mm --> nm
        max_amplitude_E = np.max(tr_E.data) * 1e3  # convert to  mm --> nm
        max_amplitude = max([max_amplitude_E, max_amplitude_N])
        ML_value = np.log10(max_amplitude)+1.11*np.log10(dist)+0.00189*dist-2.09
        print(ML_value)
        self.ML.append(ML_value)
