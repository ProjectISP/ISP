import os
import pickle
from collections import ChainMap
from isp import MAGNITUDE_DICT_PATH
#import pandas as pd

time_window_params = {"max_epi_dist": 300, "vp_tt": None, "vs_tt": None,
                      "p_arrival_tolerance": 4.0,
                      "s_arrival_tolerance": 4.0, "noise_pre_time": 15.0, "signal_pre_time": 1.0,
                      "win_length": 10.0}

spectrum_params = {"wave_type": "P", "time_domain_int": False, "ignore_vertical": False, "taper_halfwidth": 0.05,
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
                         "t_star_min_max": (0.0, 0.1), "fc_min_max" : (1.0, 50.0), "Qo_min_max": None}

postinversion_params = {"pi_fc_min_max": None, "pi_bsd_min_max": None, "pi_misfit_max": None,
                        "pi_t_star_min_max": None}

radiated_energy_params = {"max_freq_Er": None}

statistics = {"reference_statistics": 'weighted_mean', "n_sigma": 1, "lower_percentage": 15.9, "mid_percentage": 50,
              "upper_percentage": 84.1, "nIQR": 1.5}

config = ChainMap(time_window_params, spectrum_params, signal_noise_ratio_params, source_model_parameters,
                      spectral_model_params, postinversion_params, radiated_energy_params, statistics)

file = os.path.join(MAGNITUDE_DICT_PATH, "automag_config")
file_to_store = open(file, "wb")
pickle.dump(config, file_to_store)
#df = pd.read_pickle(file)
#print(df)