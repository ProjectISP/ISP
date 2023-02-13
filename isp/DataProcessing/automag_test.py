#from isp.DataProcessing.automag_tools import Indmag
from isp.Utils import MseedUtil
from collections import ChainMap
from obspy import read, Stream
from isp.Utils import read_nll_performance
time_window_params ={"channels":["HHE, HHN, HHZ"], "max_epi_dist":300, "vp_tt":None, "vs_tt":None, "p_arrival_tolerance":4.0,
                         "s_arrival_tolerance": 4.0, "noise_pre_time": 15.0, "signal_pre_time": 1.0, "win_length": 10.0}

spectrum_params = {"wave_type":"S", "time_domain_int":False, "ignore_vertical":False, "taper_halfwidth":0.05,
    "spectral_win_length": 20.0, "spectral_smooth_width_decades":0.2, "residuals_filepath":None, "bp_freqmin_acc":1.0,
    "bp_freqmax_acc":50.0, "bp_freqmin_shortp":1.0, "bp_freqmax_shortp":40.0, "bp_freqmin_broadb":0.4,
    "bp_freqmax_broadb":40.0, "freq1_acc":1, "freq2_acc":30.0, "freq1_shortp":1.0, "freq2_shortp":30.0,
                       "freq1_broadb":0.5, "freq2_broadb":10.0}


signal_noise_ratio_params = {"rmsmin":0.0, "sn_min":1.0, "clip_max_percent":5.0, "gap_max":None, "overlap_max":None,
    "spectral_sn_min":0.0, "spectral_sn_freq_range":(0.1, 2.0)}

source_model_parameters = {"vp_source":6.0, "vs_source":3.5, "vp_stations": None, "vs_stations":None, "rho": 2500.0,
                               "rpp":0.52, "rps":0.62, "rp_from_focal_mechanism":False, "geom_spread_model": "r_power_n",
                               "geom_spread_n_exponent":1.0, "geom_spread_cutoff_distance":100.0}

spectral_model_params = {"weighting":"noise", "f_weight":7.0, "weight":10.0, "t_star_0":0.045, "invert_t_star_0":False,
    "t_star_0_variability":0.1, "Mw_0_variability":0.1, "inv_algorithm":"TNC", "t_star_min_max":(0.0, 0.1),
                             "Qo_min_max":None}

postinversion_params = {"pi_fc_min_max":None, "pi_bsd_min_max":None, "pi_misfit_max":None}

radiated_energy_params = {"max_freq_Er":None}
avarage_params = {"nIQR": 1.5}

config = ChainMap(time_window_params ,spectrum_params, signal_noise_ratio_params, source_model_parameters,
                  spectral_model_params, postinversion_params, radiated_energy_params, avarage_params)


project_path = "/Users/admin/Desktop/MELI_Project"
inv_path = "/Users/admin/Documents/ISP/isp/Metadata/xml/metadata.xml"
project = MseedUtil.load_project(project_path)
files = project["WM.WMELI.HHE"][0] + project["WM.WMELI.HHN"][0]+project["WM.WMELI.HHZ"][0]
# read files
stream_complete = []
for file in files:
    print(file)
    try:
        st = read(file)
        stream_complete.append(st[0])
    except:
        pass

stream = Stream(traces = stream_complete)
path_hyp = ('/Users/admin/Desktop/24_12_2022_1828.hyp')
cat = read_nll_performance.read_nlloc_hyp_ISP(path_hyp)
picks = cat[0].picks
focal_parameters = [cat[0].origins[0]["time"],cat[0].origins[0]["latitude"],cat[0].origins[0]["longitude"],
cat[0].origins[0]["depth"]]
print(focal_parameters)

#mag = Indmag(stream,stream, pick_info, event_info, inventory))


#print(picks)
# read the corresponding.hyp file


# mg = Automag(project, inv_path, ["HHE, HHN, HHZ"])
# mg.load_metadata()
#     mg.scan_folder()
#mg.info_event()