#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
script.py
"""

from obspy import Stream, UTCDateTime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
def run_process(st: Stream, starttime: UTCDateTime, endtime: UTCDateTime, hypo_lat: float, hypo_lon: float,
                hypo_depth_km: float, hypo_origin_time: UTCDateTime):

    # example of hw to design your running script

    try:

        data = st[0].data + st[1].data

        fig, ax1 = plt.subplots(1, 1, layout='constrained')
        ax1.plot(st[0].times(), data)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Trace1 + Trace')
        ax1.grid(True)
        plt.show()

    except Exception as e:
        # Handle any exception and print the error message
        print("An error occurred:", str(e))




