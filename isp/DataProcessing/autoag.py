import os
from collections import ChainMap
import numpy as np
from datetime import datetime
from obspy import read, read_events, UTCDateTime, Stream
from isp import ALL_LOCATIONS
from isp.DataProcessing.automag_tools import preprocess_tools
#from isp.DataProcessing.automag_tools import Indmag
from isp.Utils import MseedUtil
from isp.DataProcessing.metadata_manager import MetadataManager
from isp.Utils import read_nll_performance

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

    def preprocess_stream(self, st, pick_info, regional=True):

        #TODO CHECK IS VALID THE WAVEFORM
        if regional:
            st.trim(starttime=pick_info[0][1]-60, endtime=pick_info[0][1]+3*60)
        else:
            #Teleseism time window
            st.trim(starttime=pick_info[0][1] - 1300, endtime=pick_info[0][1] + 3600)

        st.detrend(type="simple")
        st.taper(type="blackman", max_percentage=0.05)
        f1 = 0.05
        f2 = 0.08
        f3 = 0.35 * st[0].stats.sampling_rate
        f4 = 0.40 * st[0].stats.sampling_rate
        pre_filt = (f1, f2, f3, f4)

        st_deconv = []
        st_wood = []
        paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
                  'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

        for tr in st:

            tr_deconv = tr.copy()
            tr_wood = tr.copy()

            try:
                print("Removing Instrument")
                tr_deconv.remove_response(inventory=self.inventory, pre_filt=pre_filt, output="DISP", water_level=90)
                print(tr_deconv)
                # tr_deconv.plot()
                st_deconv.append(tr_deconv)
            except:
                print("Coudn't deconvolve", tr.stats)
                tr.data = np.array([])

            print("Simulating Wood Anderson Seismograph")

            try:
                resp = self.inventory.get_response(tr.id, tr.stats.starttime)
                resp = resp.response_stages[0]
                paz_mine = {'sensitivity': resp.stage_gain * resp.normalization_factor, 'zeros': resp.zeros,
                            'gain': resp.stage_gain, 'poles': resp.poles}
                tr_wood.simulate(paz_remove=paz_mine, paz_simulate=paz_wa, water_level=90)
                # tr_wood.plot()
                st_wood.append(tr_wood)
            except:
                print("Coudn't deconvolve", tr.stats)
                tr.data = np.array([])

        print("Finished Deconvolution")
        st_deconv = Stream(traces=st_deconv)
        st_wood = Stream(traces=st_wood)
        st_deconv.plot()
        st_wood.plot()
        return st_deconv, st_wood

    def statistics(self):
        MLs =np.array(self.ML)
        self.ML_mean = MLs.mean()
        self.ML_deviation = MLs.std()
        print("Local Magnitude", str(self.ML_mean)+str(self.ML_deviation))

    def get_arrival(self, arrivals, sta_name):
        arrival_return = []
        for arrival in arrivals:
            if arrival.station == sta_name:
                arrival_return.append(arrival)
        return arrival_return

    def get_now_files(self, date, station):

        selection = [".", station, "."]

        _, self.files_path = MseedUtil.filter_project_keys(self.project, net=selection[0], station=selection[1],
                                                       channel=selection[2])
        start = date.split(".")
        start = UTCDateTime(year=int(start[1]), julday=int(start[0]), hour=00, minute=00, second=00)+3600
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
                    date = str(date.julday)+ "."+ str(date.year)

                    obsfiles1.append(file_hyp)
                    if date not in dates:
                        dates[date] = [file_hyp]
                    else:
                        dates[date].append(file_hyp)
                except:
                    pass

        self.dates=dates

    def info_event(self, config):

        for date in self.dates:
            events = self.dates[date]
            self.get_now_files(date, ".") #this is just for test
            self.make_stream()
            #TODO search mseeds for this date and cut it ensure 24 and deconv, return a stream
            for event in events:
                events_picks = {}
                #cat = read_events(event, format="NLLOC_HYP")
                cat = read_nll_performance.read_nlloc_hyp_ISP(event)
                event = cat[0]
                arrivals = event["origins"][0]["arrivals"]
                arrival = self.get_arrival(arrivals, "WMELI")
                picks = cat[0].picks
                focal_parameters = [cat[0].origins[0]["time"], cat[0].origins[0]["latitude"], cat[0].origins[0]["longitude"],
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
                    inv_selected = self.inventory.select(station=key)
                    #st_deconv, st_wood = self.preprocess_stream(st2, pick_info, arrival, focal_parameters)
                    pt = preprocess_tools(st2, pick_info, focal_parameters, arrival, inv_selected)
                    pt.deconv_waveform(config['gap_max'], config['overlap_max'], config['rmsmin'],
                                       config['clipping_sensitivity'])
                    #geom_spread_model, geom_spread_n_exponent,
                    #geom_spread_cutoff_distance, rho, spectral_smooth_width_decades
                    pt.compute_spectrum(config['geom_spread_model'], config['geom_spread_n_exponent'],
                            config['geom_spread_cutoff_distance'], config['rho'], config['spectral_smooth_width_decades'])
                    #self.ML.append(mag.magnitude_local())
                #self.statistics()


if __name__ == "__main__":
    time_window_params = {"channels": ["HHE, HHN, HHZ"], "max_epi_dist": 300, "vp_tt": None, "vs_tt": None,
                          "p_arrival_tolerance": 4.0,
                          "s_arrival_tolerance": 4.0, "noise_pre_time": 15.0, "signal_pre_time": 1.0,
                          "win_length": 10.0}

    spectrum_params = {"wave_type": "S", "time_domain_int": False, "ignore_vertical": False, "taper_halfwidth": 0.05,
                       "spectral_win_length": 20.0, "spectral_smooth_width_decades": 0.2, "residuals_filepath": None,
                       "bp_freqmin_acc": 1.0,
                       "bp_freqmax_acc": 50.0, "bp_freqmin_shortp": 1.0, "bp_freqmax_shortp": 40.0,
                       "bp_freqmin_broadb": 0.4,
                       "bp_freqmax_broadb": 40.0, "freq1_acc": 1, "freq2_acc": 30.0, "freq1_shortp": 1.0,
                       "freq2_shortp": 30.0,
                       "freq1_broadb": 0.5, "freq2_broadb": 10.0}

    signal_noise_ratio_params = {"rmsmin": 0.0, "sn_min": 1.0, "clip_max_percent": 5.0, "gap_max": None,
                                 "overlap_max": None,
                                 "spectral_sn_min": 0.0, "spectral_sn_freq_range": (0.1, 2.0), "clipping_sensitivity": 3}

    source_model_parameters = {"vp_source": 6.0, "vs_source": 3.5, "vp_stations": None, "vs_stations": None,
                               "rho": 2500.0,
                               "rpp": 0.52, "rps": 0.62, "rp_from_focal_mechanism": False,
                               "geom_spread_model": "r_power_n",
                               "geom_spread_n_exponent": 1.0, "geom_spread_cutoff_distance": 100.0}

    spectral_model_params = {"weighting": "noise", "f_weight": 7.0, "weight": 10.0, "t_star_0": 0.045,
                             "invert_t_star_0": False,
                             "t_star_0_variability": 0.1, "Mw_0_variability": 0.1, "inv_algorithm": "TNC",
                             "t_star_min_max": (0.0, 0.1),
                             "Qo_min_max": None}

    postinversion_params = {"pi_fc_min_max": None, "pi_bsd_min_max": None, "pi_misfit_max": None}

    radiated_energy_params = {"max_freq_Er": None}
    avarage_params = {"nIQR": 1.5}

    config = ChainMap(time_window_params, spectrum_params, signal_noise_ratio_params, source_model_parameters,
                      spectral_model_params, postinversion_params, radiated_energy_params, avarage_params)

    #project_path = "/Users/admin/Documents/test_data/alboran_project"
    project_path = "/Users/admin/Documents/test_meli/test_meli_work"
    inv_path = "/Users/admin/Documents/test_meli/metadata/meli.xml"
    project = MseedUtil.load_project(project_path)
    mg = Automag(project, inv_path, ["HHE, HHN, HHZ"])
    mg.load_metadata()
    mg.scan_folder()
    mg.info_event(config)
    #mg.get_now_files()