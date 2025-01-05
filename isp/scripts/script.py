#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
script.py
"""

from obspy import Stream, UTCDateTime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
def run_process(st: Stream, chop: dict, starttime: UTCDateTime, endtime: UTCDateTime, hypo_lat: float,
                hypo_lon: float, hypo_depth_km: float, hypo_origin_time: UTCDateTime):


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


"""
chop = {'Body waves': {}, 'Surf Waves': {}, 'Coda': {}, 'Noise': {}}: dict
id = {id: [metadata, t, s, xmin_index, xmax_index, t_start_utc, t_end_utc]}
metadata = [dic_metadata['net'], dic_metadata['station'], dic_metadata['location'], dic_metadata['channel'],
            dic_metadata['starttime'], dic_metadata['endtime'], dic_metadata['sampling_rate'],
            dic_metadata['npts']]

# example of chop_full_dict_input = {'Body waves':{"WM.SFS..HHZ": [[WM, SFS,,HHZ,...], time_amplitudes, amplitudes,...
  
def search_chop(chop, st, wave_type = 'Body waves'):
    if st:
        try:
            n = len(st)
            self.kind_wave = self.ChopCB.currentText()
            for j in range(n):
                tr = st[j]
                if tr.id in self.chop[wave_type]:
                    print(chop[self.kind_wave][tr.id])

        except Exception:
            raise Exception('Nothing to clean')
"""