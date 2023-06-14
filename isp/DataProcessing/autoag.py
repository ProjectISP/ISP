import os
import numpy as np
from datetime import datetime
from obspy import read, read_events, UTCDateTime, Stream
from isp import ALL_LOCATIONS
from isp.DataProcessing.automag_processing_tools import ssp_inversion
from isp.DataProcessing.automag_statistics import compute_summary_statistics, SourceSpecOutput
from isp.DataProcessing.automag_tools import preprocess_tools
from isp.DataProcessing.radiated_energy import Energy
from isp.Utils import MseedUtil
#from isp.Utils import read_nll_performance

class Automag:

    def __init__(self, origin, event, project, inventory):
        self.origin = origin
        self.event = event
        self.project = project
        self.inventory = inventory
        self.all_traces = []
        self.st = None
        self.ML = []

    def make_stream(self):

         for file in self.files_path:
             try:
                 st = read(file[0])
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
        tr.trim(starttime=check_starttime, endtime=check_endtime, pad=False, nearest_sample=True, fill_value=None)
        return tr

    def ML_statistics(self):
        self.ML = list(filter(lambda item: item is not None, self.ML))
        MLs = np.array(self.ML)
        ML_mean = MLs.mean()
        ML_deviation = MLs.std()

        return ML_mean, ML_deviation

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
        #start = date.split(".")
        #start = UTCDateTime(year=int(start[1]), julday=int(start[0]), hour=00, minute=00, second=00)+3600
        #end = start+23*3600
        #self.files_path = MseedUtil.filter_time(list_files=self.files_path, starttime=start, endtime=end)
        print(date, self.files_path)

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

    def scan_from_origin(self, origin):

        date = origin["time"]
        self.dates = str(date.julday) + "." + str(date.year)


    def _get_stations(self, arrivals):
        stations = []
        for pick in arrivals:
            if pick.station not in stations:
                stations.append(pick.station)

        return stations

    def _get_info_in_arrivals(self, station, arrivals, min_residual_threshold):
        data = {}
        geodetics = {}
        for arrival in arrivals:
            if station == arrival.station:
                geodetics["distance_km"] = arrival.distance_km
                geodetics["distance_degrees"] = arrival.distance_degrees
                geodetics["azimuth"] = arrival.azimuth
                geodetics["takeoff_angle"] = arrival.takeoff_angle
                if arrival.phase[0] == "P":
                    geodetics["travel_time"] = float(arrival.travel_time)
                if arrival.phase in data.keys():
                    data[arrival.phase]["time_weight"].append(arrival.time_weight)
                    data[arrival.phase]["date"].append(arrival.date)
                else:
                    data[arrival.phase] = {}
                    data[arrival.phase]["time_weight"] = []
                    data[arrival.phase]["date"] = []
                    data[arrival.phase]["time_weight"].append(arrival.time_weight)
                    data[arrival.phase]["date"].append(arrival.date)

        output = {}
        output[station] = []
        for key, value in data.items():

             residual_min = list(map(abs, data[key]["time_weight"]))
             residual_min = max(residual_min)
             residual_min_index = data[key]["time_weight"].index(residual_min)
             if data[key]["date"][residual_min_index] >= min_residual_threshold:
                output[station].append([key, data[key]["date"][residual_min_index]])

        return output, geodetics


    def estimate_magnitudes(self, config):
        magnitude_mw_statistics = {}
        magnitude_ml_statistics = {}
        # extract info from config:
        gap_max = config['gap_max']
        overlap_max = config['overlap_max']
        rmsmin = config['rmsmin']
        clipping_sensitivity = config['clipping_sensitivity']
        geom_spread_model = config['geom_spread_model']
        geom_spread_n_exponent = config['geom_spread_n_exponent']
        geom_spread_cutoff_distance = config['geom_spread_cutoff_distance']
        rho = config['rho']
        spectral_smooth_width_decades = config['spectral_smooth_width_decades']
        spectral_sn_min = config['spectral_sn_min']
        spectral_sn_freq_range = config['spectral_sn_freq_range']
        t_star_0_variability = config["t_star_0_variability"]
        invert_t_star_0 = config["invert_t_star_0"]
        t_star_0 = config["t_star_0"]
        inv_algorithm = config["inv_algorithm"]
        pi_misfit_max = config["pi_misfit_max"]
        pi_t_star_min_max = config["pi_t_star_min_max"]
        pi_fc_min_max = config["pi_fc_min_max"]
        pi_bsd_min_max = config["pi_bsd_min_max"]
        max_freq_Er = config["max_freq_Er"]
        min_residual_threshold = config["min_residual_threshold"]
        scale = config["scale"]
        max_win_duration = config["win_length"]
        a = config["a_local_magnitude"]
        b = config["b_local_magnitude"]
        c = config["c_local_magnitude"]
        bound_config = {"Qo_min_max": config["Qo_min_max"], "t_star_min_max": config["t_star_min_max"],
                        "wave_type": config["wave_type"], "fc_min_max": config["fc_min_max"]}
        statistics_config = config.maps[7]

        #for date in self.dates:
        #events = self.dates[date]
        self.get_now_files(self.dates, ".")
        self.make_stream()
        #for event in events:
        sspec_output = SourceSpecOutput()

        #cat = read_nll_performance.read_nlloc_hyp_ISP(event)
        focal_parameters = [self.event.origins[0]["time"], self.event.origins[0]["latitude"],
                            self.event.origins[0]["longitude"],
                            self.event.origins[0]["depth"] * 1E-3]
        sspec_output.event_info.event_id = "Id_Local"
        sspec_output.event_info.longitude = self.event.origins[0]["longitude"]
        sspec_output.event_info.latitude = self.event.origins[0]["latitude"]
        sspec_output.event_info.depth_in_km = self.event.origins[0]["depth"] * 1E-3
        sspec_output.event_info.origin_time = self.event.origins[0]["time"]

        arrivals = self.event["origins"][0]["arrivals"]
        stations = self._get_stations(arrivals)

        for station in stations:

            events_picks, geodetics = self._get_info_in_arrivals(station, arrivals, min_residual_threshold)
            pick_info = events_picks[station]
            st2 = self.st.select(station=station)
            if st2.count() > 0:
                inv_selected = self.inventory.select(station=station)
                pt = preprocess_tools(st2, pick_info, focal_parameters, geodetics, inv_selected, scale)
                pt.deconv_waveform(gap_max, overlap_max, rmsmin, clipping_sensitivity, max_win_duration)
                pt.st_deconv = pt.st_deconv.select(component="Z")
                if pt.st_deconv.count() > 0 and pt.st_wood.count() > 0:

                    self.ML.append(pt.magnitude_local(a, b, c))
                    spectrum_dict = pt.compute_spectrum(geom_spread_model, geom_spread_n_exponent,
                                    geom_spread_cutoff_distance, rho, spectral_smooth_width_decades,
                                    spectral_sn_min, spectral_sn_freq_range)

                    if spectrum_dict is not None:
                        ssp = ssp_inversion(spectrum_dict, t_star_0_variability, invert_t_star_0, t_star_0,
                            focal_parameters, geodetics, inv_selected, bound_config, inv_algorithm, pi_misfit_max,
                                            pi_t_star_min_max, pi_fc_min_max, pi_bsd_min_max)

                        magnitudes = ssp.run_estimate_all_traces()
                        for chn in magnitudes:
                            sspec_output.station_parameters[chn._id] = chn
                            # for now just for vertical component
                            for keyId, trace_dict in spectrum_dict.items():
                                spec = trace_dict["amp_signal_moment"]
                                specnoise = trace_dict["amp_signal_moment"]
                                freq_signal = trace_dict["freq_signal"]
                                freq_noise = trace_dict["freq_noise"]
                                full_period_signal = trace_dict["full_period_signal"]
                                full_period_noise = trace_dict["full_period_noise"]
                                vs = trace_dict["vs"]
                            # compute and implement energy
                            sspec_output.station_parameters[chn._id] = Energy.radiated_energy(chn._id, spec,
                                specnoise, freq_signal, freq_noise, full_period_signal, full_period_noise,
                                chn.fc.value, vs, max_freq_Er, rho, chn.t_star.value, chn)

        magnitude_mw_statistics = compute_summary_statistics(statistics_config, sspec_output)
        ML_mean, ML_std = self.ML_statistics()
        magnitude_ml_statistics["ML_mean"] = ML_mean
        magnitude_ml_statistics["ML_std"] = ML_std
        print(magnitude_mw_statistics, ML_mean, ML_std)
        
        return magnitude_mw_statistics, magnitude_ml_statistics