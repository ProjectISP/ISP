# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
import obspy
import numpy as np

def read_log(file):
    """
    Reads the ISOLA-ObsPy output log file.

    Parameters
    ----------
    file : string
        The full path to the log.txt file.

    Returns
    -------
    log_dict : dict
        A dictionary containing the results from the moment tensor inversion.

    """

    with open(file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line == "Centroid location:\n":
            ri = i
            break
    
    log_dict = {}
    
    #log_dict["Time"] = obspy.UTCDateTime('T'.join(lines[ri+1].strip('\n').split(' ')[4:]))
    log_dict["latitude"] = float(lines[ri+2].split(' ')[5])
    log_dict["longitude"] = float(lines[ri+2].split(' ')[11])
    log_dict["depth"] = float(lines[ri+2].split(' ')[16])
    
    log_dict["VR"] = float(lines[ri+6].strip('\n').split(':')[1].strip(' %'))
    log_dict["CN"] = float(lines[ri+7].strip('\n').split(':')[1].strip(' '))
    
    MT_exp = float(lines[ri+9].split('*')[1].strip(' \n'))
    MT = np.array([float(x) for x in lines[ri+9].split('*')[0].strip('[ ]').split(' ') if x != ''])
    MT *= MT_exp
    
    mrr, mtt, mpp, mrt, mrp, mtp = MT
    log_dict["mrr"] = mrr
    log_dict["mtt"] = mtt
    log_dict["mpp"] = mpp
    log_dict["mrt"] = mrt
    log_dict["mrp"] = mrp
    log_dict["mtp"] = mtp
    
    log_dict["mo"] = float(lines[ri+11].split('M0')[1].split('Nm')[0].strip(' ='))
    log_dict["mw_mt"] = float(lines[ri+11].split('M0')[1].split('Nm')[1].strip('\n ( ) Mw = '))
    
    regex = re.compile('[a-zA-Z =:%,\n]')
    
    dc, clvd, iso = [float(x) for x in re.sub(regex, ' ', lines[ri+12]).split(' ') if x != '']
    log_dict["dc"] = dc
    log_dict["clvd"] = clvd
    log_dict["iso"] = iso
    
    fp1_strike, fp1_dip, fp1_rake = [float(x) for x in re.sub(regex, ' ', lines[ri+13].split(':')[1]).split(' ') if x != '' and x != '-']
    fp2_strike, fp2_dip, fp2_rake = [float(x) for x in re.sub(regex, ' ', lines[ri+14].split(':')[1]).split(' ') if x != '' and x != '-']
    log_dict["strike_mt"] = fp1_strike
    log_dict["dip_mt"] = fp1_dip
    log_dict["rake_mt"] = fp1_rake
    #log_dict["fp2_strike"] = fp2_strike
    #log_dict["fp2_dip"] = fp2_dip
    #log_dict["fp2_rake"] = fp2_rake
    
    return log_dict
