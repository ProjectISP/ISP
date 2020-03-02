#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:09:28 2020

@author: olivar
"""

from isp.mti.src.class_isola import *

def perform_mti():

    isola = ISOLA(location_unc = 3000, # m
    	depth_unc = 3000, # m
    	time_unc = 0, # s
    	deviatoric = True, 		# True = force isotropic component to be zero
    	step_x = 50, # m
    	step_z = 50, # m
    	max_points = 500,
    	threads = 8,
    	circle_shape = True,
    	use_precalculated_Green = True,
    	rupture_velocity = 1000, # m/s
        logfile='mti/output/log.txt'
    )
    
    isola.read_event_info('mti/input/event.isl')
    isola.read_network_coordinates('mti/input/network.stn', network='GR', channelcode='HH')
    isola.read_crust('mti/input/crustal.dat')
    
    isola.add_SAC('mti/invert/sac/', prefix='r', pz_dir='mti/invert/pzfiles/', pz_separator='', pz_suffix='.pz') # reads SAC files and P&Z files
    
    isola.detect_mouse(figures='mti/output/mouse/')
    isola.correct_data()

    isola.set_parameters(0.15, 0.02)
    
    if not isola.calculate_or_verify_Green():
    	exit()
    
    isola.trim_filter_data(noise_slice=False)
    isola.decimate_shift()
    isola.run_inversion()
    isola.find_best_grid_point()
    
    # plotting solution
    isola.print_solution()
    isola.print_fault_planes()
    if len(isola.grid) > len(isola.depths):
    	isola.plot_maps('mti/output/map.png')
    if len(isola.depths) > 1:
    	isola.plot_slices('mti/output/slice.png')
    isola.plot_MT()
    isola.plot_seismo(outfile='mti/output/seismo-var_y_range.png')
    isola.plot_seismo(sharey=True)
    isola.plot_stations()
    isola.html_log(outfile='mti/output/index.html', h1='Example 1', plot_MT='centroid.png', plot_stations='stations.png', plot_seismo_sharey='seismo.png', mouse_figures='mouse/', plot_maps='map.png', plot_slices='slice.png')
