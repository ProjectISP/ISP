import os
import numpy as np
from datetime import datetime
import pandas as pd
from obspy import read, read_events, UTCDateTime, Stream
from obspy.geodetics import gps2dist_azimuth
from isp import LOCATION_OUTPUT_PATH, ALL_LOCATIONS
from isp.Gui.Frames.open_magnitudes_calc import get_coordinates_from_metadata
from isp.Utils import MseedUtil


class Automag:

    def __init__(self, project):
        self.project = project
        self.all_traces = []
        self.st = None

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


    def remove_response(self, st, f1, f2, f3, f4, water_level, units):

        pass
        # done = True
        #
        # try:
        #     st.remove_response(inventory=self.inventory, pre_filt=(f1, f2, f3, f4), output=units, water_level=water_level)
        # except:
        #     print("Coudn't deconvolve", print(st))
        #     done = False
        #
        # return tr, done

    def get_now_files(self, date, station):

        selection = [".",station, "."]

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
            net = header[0].stats.network
            sta = header[0].stats.station
            chn = header[0].stats.channel
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
                        events_picks[pick.waveform_id["station_code"]]=[pick.phase_hint, pick.time]
                    else:
                        events_picks[pick.waveform_id["station_code"]].append([pick.phase_hint, pick.time])
                print("end event")
                for key in events_picks:
                    pick_info = events_picks[key]
                    st2 = self.st.select(station=key)
                #TODO: In this point send info to process magnitudes

class Indmag:
    def __init__(self, st, pick_info, event_info, inventory):
        self.st = st
        self.inventory = inventory
        self.pick_info = pick_info
        self.event_info = event_info

    def magnitude_local(self):

        pass


if __name__ == "__main__":
    project_path = "/Users/robertocabieces/Documents/Alboran"
    df = pd.read_pickle(project_path)
    project = MseedUtil.load_project(file=project_path)
    mg = Automag(project)
    mg.scan_folder()
    mg.info_event()
    #mg.get_now_files()