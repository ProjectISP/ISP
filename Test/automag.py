import os
import numpy as np
from datetime import datetime
import pandas as pd
from obspy import read, read_events, UTCDateTime, Stream
from obspy.geodetics import gps2dist_azimuth

from isp import ALL_LOCATIONS
from isp.Utils import MseedUtil
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Structures.structures import StationCoordinates

class Automag:

    def __init__(self, project, inventory_path, working_chnnels):
        self.project = project
        self.all_traces = []
        self.st = None
        self.ML = []
        self.ML_std = []
        self.inventory_path = inventory_path
        self.working_channels = working_chnnels

    def load_metadata(self):

        try:

            self.__metadata_manager = MetadataManager(self.inventory_path)
            self.inventory = self.__metadata_manager.get_inventory()
            print(self.inventory)
        except:
            raise FileNotFoundError("The metadata is not valid")



    def make_stream(self):

         for file in self.files_path:
             try:
                 st = read(file)
                 st = self.fill_gaps(st, 60)
                 tr = self.ensure_24(st[0])
                 self.all_traces.append(tr)
             except:
                 pass
    #
         self.st = Stream(traces=self.all_traces)
    #     return st

    def check_gaps(self, gaps, tol):
        time_gaps = []
        for i in gaps:
            time_gaps.append(i[6])

        sum_total = sum(time_gaps)

        if sum_total > tol:
            check = True
        else:
            check = False

        return check

    def fill_gaps(self, st, tol):
        gaps = st.get_gaps()

        if len(gaps) > 0 and self.check_gaps(gaps, tol):
            st.print_gaps()
            st = []

        elif len(gaps) > 0 and self.check_gaps(gaps, tol) == False:
            st.print_gaps()
            st.merge(fill_value="interpolate", interpolation_samples=-1)

        elif len(gaps) == 0 and self.check_gaps(gaps, tol) == False:
            pass
        return st

    def ensure_24(self, tr):
        # Ensure that this trace is set to have 24h points padding with zeros the starttime and endtime
        # take random numbers to ensure the day
        random_list = np.random.choice(len(tr), 100)
        times_posix = tr.times(type="timestamp")
        days_prob = times_posix[random_list.tolist()]
        days_prob_max = days_prob.tolist()
        max_prob = max(set(days_prob_max), key=days_prob_max.count)
        year = int(datetime.utcfromtimestamp(max_prob).strftime('%Y'))
        month = int(datetime.utcfromtimestamp(max_prob).strftime('%m'))
        day = int(datetime.utcfromtimestamp(max_prob).strftime('%d'))

        check_starttime = UTCDateTime(year=year, month=month, day=day, hour=00, minute=00, microsecond=00)
        check_endtime = check_starttime + 24 * 3600
        tr.detrend(type="simple")
        tr.trim(starttime=check_starttime, endtime=check_endtime, pad=True, nearest_sample=True, fill_value=0)
        return tr

    def remove_response(self, st):

        st_deconv = []
        st_wood = []

        for tr in st:
            f1 = 0.05
            f2 = 0.08
            f3 = 0.3*tr.stats.sampling_rate
            f4 = 0.4*tr.stats.sampling_rate
            pre_filt = (f1, f2, f3, f4)

            # print("Deconvolving")
            try:
                tr_test = tr.copy()
                tr_test.remove_response(inventory=self.inventory, pre_filt=pre_filt, output="displacement",
                                        water_level=90)
                st_deconv.append(tr)
            except:
                print("Coudn't deconvolve", tr.stats)
                tr.data = np.array([])

                # print("Simulating Wood Anderson Seismograph")
            resp = self.inventory.get_response(tr.id, tr.stats.starttime)
            resp = resp.response_stages[0]
            paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
                      'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

            paz_mine = {'sensitivity': resp.stage_gain * resp.normalization_factor, 'zeros': resp.zeros,
                        'gain': resp.stage_gain, 'poles': resp.poles}

            try:
                tr_test = tr.copy()
                tr_test.simulate(paz_remove=paz_mine, paz_simulate=paz_wa, water_level=90)
                st_wood.append(tr_test)
            except:
                print("Coudn't deconvolve", tr.stats)
                tr.data = np.array([])

        st_deconv = Stream(traces=st_deconv)
        st_wood = Stream(traces=st_wood)
        del self.st

        return st_deconv, st_wood



    def get_now_files(self, date, station):

        selection = [".", station, "."]

        _, self.files_path = MseedUtil.filter_project_keys(self.project, net=selection[0], station=selection[1],
                                                       channel=selection[2])
        start = date.split(".")
        start = UTCDateTime(year=int(start[0]), julday=int(start[1]), hour=00, minute=00, second=00)+3600
        end = start+23*3600
        self.files_path = MseedUtil.filter_time(list_files=self.files_path, starttime=start, endtime=end)
        print(self.files_path)

    def filter_station(self, station):

        filtered_list = []

        for file in self.files_path:
            header = read(file, headlonly=True)
            sta = header[0].stats.station
            if station == sta:
                filtered_list.append(file)

        return filtered_list

    def scan_folder(self):
        obsfiles1 = []
        dates = {}
        for top_dir, _, files in os.walk(ALL_LOCATIONS):

            for file in files:
                try:
                    file_hyp = os.path.join(top_dir, file)
                    cat = read_events(file_hyp, format="NLLOC_HYP")
                    ev = cat[0]
                    date = ev.origins[0]["time"]
                    date = str(date.year)+"."+str(date.julday)

                    obsfiles1.append(file_hyp)
                    if date not in dates:
                        dates[date] = [file_hyp]
                    else:
                        dates[date].append(file_hyp)
                except:
                    pass

        self.dates=dates

    def info_event(self):
        events_picks = {}
        for date in self.dates:
            events = self.dates[date]
            self.get_now_files(date, ".")
            self.make_stream()
            #TODO search mseeds for this date and cut it ensure 24 and deconv, return a stream
            for event in events:
                cat = read_events(event, format="NLLOC_HYP")
                picks = cat[0].picks
                focal_parameters=[cat[0].origins[0]["time"],cat[0].origins[0]["latitude"],cat[0].origins[0]["longitude"],
                cat[0].origins[0]["depth"]]
                print(focal_parameters)
                for pick in picks:
                    if pick.waveform_id["station_code"] not in events_picks.keys():
                        events_picks[pick.waveform_id["station_code"]]=[[pick.phase_hint, pick.time]]
                    else:
                        events_picks[pick.waveform_id["station_code"]].append([pick.phase_hint, pick.time])
                print("end event")
                for key in events_picks:
                    pick_info = events_picks[key]
                    st2 = self.st.select(station=key)
                    st_deconv, st_wood = self.remove_response(st2)
                    mag = Indmag(st_deconv,st_wood, pick_info, focal_parameters, self.inventory)
                    mag.magnitude_local()
                #TODO: In this point send info to process magnitudes

class Indmag:
     def __init__(self, st_deconv, st_wood, pick_info, event_info, inventory):
         self.st_deconv = st_deconv
         self.st_wood = st_wood
         self.inventory = inventory
         self.pick_info = pick_info
         self.event_info = event_info
         self.ML = []
     def extrac_coordinates_from_station_name(self, inventory, name):
         selected_inv = inventory.select(station=name)
         cont = selected_inv.get_contents()
         coords = selected_inv.get_coordinates(cont['channels'][0])
         return StationCoordinates.from_dict(coords)

     def magnitude_local(self):
         tr_N = self.st_wood.select(component="N")
         tr_E = self.st_wood.select(component="E")
         pickP_time = None
         for item in self.pick_info:
             if item[0] == "P":
                 pickP_time = item[1]

         tr_N.trim(starttime=pickP_time-15, endtime = pickP_time+300)
         tr_E.trim(starttime=pickP_time-15, endtime=pickP_time + 300)
        #
        #
         coords = self.extrac_coordinates_from_station_name(self.inventory, self.st_wood[0].stats.station)
         dist, _, _ = gps2dist_azimuth(coords.Latitude, coords.Longitude, self.event_info[1], self.event_info[2])
         dist = dist / 1000
         max_amplitude_N = np.max(tr_N.data) * 1e3  # convert to  mm --> nm
         max_amplitude_E = np.max(tr_E.data) * 1e3
         max_amplitude = max([max_amplitude_E, max_amplitude_N])
         ML_value = np.log10(max_amplitude) + 1.11 * np.log10(dist) + 0.00189 * dist - 2.09
         self.ML.append(ML_value)
        #  MLs = np.array(ML)
        #  ML_mean = MLs.mean()
        #  ML_deviation = MLs.std()

if __name__ == "__main__":
    project_path = "/home/rcabdia/Documentos/magnitudes_test/data/alboran"
    inv_path = "/home/rcabdia/Documentos/ISP/isp/Metadata/xml/metadata.xml"
    df = pd.read_pickle(project_path)
    project = MseedUtil.load_project(project_path)
    mg = Automag(project, inv_path, ["HHE, HHN, HHZ"])
    mg.load_metadata()
    mg.scan_folder()
    mg.info_event()
    #mg.get_now_files()