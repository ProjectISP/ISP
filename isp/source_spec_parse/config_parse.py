#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass

@dataclass
class SourceSpecConfig:
    """
    config_parse
    :param :
    :type :
    :return:
    :rtype:
    """

    author_name: str = "SurfQuakeCore"
    author_email: str = "https://projectisp.github.io/ISP_tutorial.github.io/"
    agency_full_name: str = "Spanish Navy Observatory"
    agency_short_name: str = "ROA"
    mis_oriented_channels: str = "Z, 1, 2"
    instrument_code_acceleration: str = None
    instrument_code_velocity: str = None
    traceid_mapping_file: str = None
    ignore_traceids: str = None
    use_traceids: str = None
    epi_dist_ranges: str = "0, 400.0"
    sensitivity: str = None
    correct_instrumental_response: bool = True
    trace_units: str = "auto"
    vp_tt: str = None
    vs_tt: str = None
    NLL_time_dir: str = None
    p_arrival_tolerance: float = 4.0
    s_arrival_tolerance: float = 4.0
    noise_pre_time: float = 15.0
    signal_pre_time: float = 1.0
    win_length: float = 10.0
    wave_type: str = "S"
    time_domain_int: bool = False
    ignore_vertical: bool = False
    taper_halfwidth: float = 0.05
    spectral_win_length: float = 20.0
    spectral_smooth_width_decades: float = 0.2
    residuals_filepath: str = None
    bp_freqmin_acc: float = 1.0
    bp_freqmax_acc: float = 50.0
    bp_freqmin_shortp: float = 1.0
    bp_freqmax_shortp: float = 40.0
    bp_freqmin_broadb: float = 0.4
    bp_freqmax_broadb: float = 40.0
    freq1_acc: float = 1.0
    freq2_acc: float = 30.0
    freq1_shortp: float = 1.0
    freq2_shortp: float = 30.0
    freq1_broadb: float = 0.5
    freq2_broadb: float = 10.0
    rmsmin: float = 0.0
    sn_min: float = 1.0
    clip_max_percent: float = 5.0
    gap_max: str = None
    overlap_max: str = None
    spectral_sn_min: float = 0.0
    spectral_sn_freq_range: str = "0.1, 2.0"
    vp_source: float = 6.0
    vs_source: float = 3.5
    vp_stations: str = None
    vs_stations: str = None
    NLL_model_dir: str = None
    rho_source: float = 2500.0
    rpp: float = 0.52
    rps: float = 0.62
    rp_from_focal_mechanism: bool = False
    geom_spread_model: str = "r_power_n"
    geom_spread_n_exponent: float = 1.0
    geom_spread_cutoff_distance: float = 100.0
    weighting: str = "noise"
    f_weight: float = 7.0
    weight: float = 10.0
    t_star_0: float = 0.045
    invert_t_star_0: bool = False
    t_star_0_variability: float = 0.1
    Mw_0_variability: float = 0.1
    inv_algorithm: str = "TNC"
    fc_min_max: str = None
    t_star_min_max: str = "0.0, 0.1"
    Qo_min_max: str = None
    pi_fc_min_max: str = None
    pi_t_star_min_max: str = None
    pi_bsd_min_max: str = None
    pi_misfit_max: str = None
    max_freq_Er: str = None
    compute_local_magnitude: bool = True
    a: float = 1.0
    b: float = 0.00301
    c: float = 3.0
    ml_bp_freqmin: float = 0.1
    ml_bp_freqmax: float = 20.0
    nIQR: float = 1.5
    plot_show: bool = False
    plot_save: bool = True
    plot_save_format: str = "png"
    plot_spectra_no_attenuation: bool = False
    plot_spectra_no_fc: bool = False
    plot_spectra_maxrows: int = 3
    plot_traces_maxrows: int = 3
    plot_traces_ignored: bool = True
    plot_spectra_ignored: bool = True
    plot_station_map: bool = False
    plot_station_names_on_map: bool = True
    plot_station_text_size: float = 8.0
    plot_coastline_resolution: str = None
    plot_map_tiles_zoom_level: str = None
    html_report: bool = True
    event_url: str = "https://projectisp.github.io/ISP_tutorial.github.io//$EVENTID"
    set_preferred_magnitude: bool = False
    smi_base: str = "smi:local"
    smi_strip_from_origin_id: str = ""
    smi_magnitude_template: str = "$SMI_BASE/Magnitude/Origin/$ORIGIN_ID#sourcespec"
    smi_station_magnitude_template: str = "$SMI_MAGNITUDE_TEMPLATE#$WAVEFORM_ID"
    smi_moment_tensor_template: str = "$SMI_BASE/MomentTensor/Origin/$ORIGIN_ID#sourcespec"
    smi_focal_mechanism_template: str = "$SMI_BASE/FocalMechanism/Origin/$ORIGIN_ID#sourcespec"

"""
# Create an instance of the data class with your desired configuration
config = SourceSpecConfig()

# Convert the data class instance to a dictionary
config_dict = dataclasses.asdict(config)

# Write the dictionary to the configuration file using configparser
config_parser = configparser.ConfigParser()
config_parser.read_dict({"source_spec": config_dict})

# Write the configuration to a file
with open("source_spec.conf", "w") as config_file:
    config_parser.write(config_file)

print("source_spec.conf file has been generated.")
"""