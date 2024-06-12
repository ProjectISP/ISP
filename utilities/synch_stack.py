#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from obspy.io.sac import SACTrace
from stackmaster.core import stack

class SynchEnergy():
    def __init__(self, path_ref, path_to_polynom, output_path, plots):

        self.obsfiles_daily = []
        self.obsfiles =[]
        self.input_path = path_ref
        self.path_to_polynom = path_to_polynom
        self.output_path = output_path
        self.plots = plots
        self.list_dir()
        self.get_polynom()
        self.model = None

    def get_polynom(self):
        df = pd.read_pickle(self.path_to_polynom)
        print(df)
        return df

    def plot_polynom(self, dates, model):

        fig, ax = plt.subplots()
        ax.plot(dates, model(dates))
        #plt.show()

    def list_dir(self):

        for top_dir, sub_dir, files in os.walk(self.input_path):
            for file in files:
                if os.path.basename(file) != ".DS_Store":
                    extension = file.split("_")
                    if len(extension)==3:
                        self.obsfiles_daily.append(os.path.join(top_dir, file))
                    else:
                        self.obsfiles.append(os.path.join(top_dir, file))
        self.obsfiles_daily.sort()
        self.obsfiles.sort()
        print(self.obsfiles_daily)
        print(self.obsfiles)

    def _fill_matrix(self, matrix, st):

        for i, item in enumerate(st["stream"]):
            matrix[i, :] = st["stream"][i].data
        return matrix


    def _plot_matrix(self, matrix, st, tr, stack, correct):
        plt.rcParams['figure.constrained_layout.use'] = True
        fs = tr[0].stats.sampling_rate

        t = st["stream"][0].times()
        t_shift = np.roll(t, int(len(t)/2))
        t = t - t_shift
        times = np.arange(np.min(t), np.max(t), 1/fs)
        matrix = matrix/np.max(matrix)
        tr[0].data = tr[0].data/np.max(tr[0].data)
        stack = stack / np.max(stack)
        extent = np.min(t), np.max(t), np.min(st["dates"]), np.max(st["dates"]),
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})
        if correct:
            ax0.plot(times, tr[0].data, color="black", linewidth=0.75, alpha=0.5,
                     label=tr[0].id)
            ax0.plot(times, stack, color="green", linewidth=0.75)
        else:
            ax0.plot(times, tr[0].data / np.max(tr[0].data), color="black", linewidth=0.75,
                     label=tr[0].id)
        ax0.set_xlim([np.min(t), np.max(t)])
        c = ax1.imshow(matrix, cmap=plt.cm.seismic, interpolation='gaussian', extent=extent, vmin=-1, vmax=1)
        ax0.legend()
        fig.colorbar(c, orientation="horizontal")
        c.set_label('Amplitud')
        file_name = os.path.join(self.plots, tr[0].id)+".pdf"

        plt.savefig(file_name, dpi=250)
        #plt.subplots_adjust(hspace=0)
        plt.show()


    def mapping(self, tr1, output_path, format="H5"):

        print("Writting file")
        #tr1.plot()
        N = len(tr1.data)
        if (N % 2) == 0:

            # print(“The number is even”)
            c = int(np.ceil(N / 2.) + 1)
        else:
            # print(“The providednumber is odd”)
            c = int(np.ceil((N + 1) / 2))



        tr_causal = tr1.copy()
        tr_acausal = tr1.copy()
        fs = tr1.stats.sampling_rate
        # take into acount causality

        "Causal"
        starttime = tr_causal.stats.starttime
        endtime = tr_causal.stats.starttime+(c/fs)
        tr_causal.trim(starttime=starttime, endtime=endtime)
        data_causal = np.flip(tr_causal.data)
        tr_causal.data = data_causal

        "Acausal"
        starttime = tr_acausal.stats.starttime + (c/fs)
        endtime = tr_acausal.stats.endtime
        tr_acausal.trim(starttime=starttime, endtime=endtime)
        N_cut = min(len(tr_causal.data), len(tr_acausal.data))
        "Both"
        tr_acausal.data = (tr_causal.data[0:N_cut]+tr_acausal.data[0:N_cut])/2
        #tr_acausal.plot()
        # header = {'b': -1*(c/fs), 'e': 1*(c/fs), 'npts':N, 'kstnm': tr.stats.station, 'kcmpnm': tr.stats.channel,
        #           'stla': tr.stats.mseed['coordinates'][0], 'stlo': tr.stats.mseed['coordinates'][1],
        #       'evla': tr.stats.mseed['coordinates'][2], 'evlo': tr.stats.mseed['coordinates'][3], 'evdp': 0,
        #           'delta': 1.0, 'dist': tr.stats.mseed['geodetic'][0]*1E-3, 'baz':tr.stats.mseed['geodetic'][1],
        #           'az': tr.stats.mseed['geodetic'][2]}

        header = {'b': 0, 'e': 1*(c/fs), 'npts': c, 'kstnm': tr_acausal.stats.station, 'kcmpnm': tr_acausal.stats.channel,
                   'stla': tr_acausal.stats.mseed['coordinates'][0], 'stlo': tr_acausal.stats.mseed['coordinates'][1],
               'evla': tr_acausal.stats.mseed['coordinates'][2], 'evlo': tr_acausal.stats.mseed['coordinates'][3], 'evdp': 0,
                   'delta': 1.0, 'dist': tr_acausal.stats.mseed['geodetic'][0]*1E-3, 'baz':tr_acausal.stats.mseed['geodetic'][1],
                   'az': tr_acausal.stats.mseed['geodetic'][2]}

        name = tr_acausal.stats.network + "." + tr_acausal.stats.station + "." + tr_acausal.stats.channel
        if format == "sac":
            sac = SACTrace(data=tr_causal.data, **header)
            path = os.path.join(output_path, name+"_"+"sac")
            sac.write(path, byteorder='little', flush_headers=True)

        elif format == "H5":

            path_mseed = os.path.join(output_path, name+"_"+"h5")
            tr1.write(path_mseed, format="H5")

    def create_matrix(self, filter=False, correct=True, plot=True, save=False, trim=False, stack_type="linear",
                      format="H5", f1=0.033, f2=0.333):

        df = self.get_polynom()
        model = df["model"]

        for file_stack, file in zip(self.obsfiles, self.obsfiles_daily):
            #try:
            tr = read(file_stack)
            d_rate = tr[0].stats.sampling_rate
            #print(tr.id)
            tr.detrend(type="simple")
            tr.taper(max_percentage=0.05)
            if filter:
                tr.filter(type="bandpass", freqmin=f1, freqmax=f2, zerophase=True)
                tr.detrend(type="simple")
                tr.taper(max_percentage=0.05)
                #tr.plot()

            st = pd.read_pickle(file)
            #st['stream'][0].plot()

            starttime = st["stream"][0].stats.starttime
            endtime = st["stream"][0].stats.endtime
            central_time = starttime+(endtime-starttime)/2
            if trim:
                starttime_new = central_time - 400
                endtime_new = central_time + 400

            st["stream"].detrend(type="simple")
            st["stream"].taper(max_percentage=0.05)
            #if filter:
            #    st["stream"].filter(type="bandpass", freqmin=0.05, freqmax=0.1, zerophase=False)
            if trim:
                st["stream"].trim(starttime=starttime_new, endtime = endtime_new)
            #self.plot_polynom(st["dates"], model)
            #st["stream"].pop(0)
            if correct == True:
                for i in range(len(st["stream"])):
                    #st["stream"][i].data = np.roll(st["stream"][i].data, self.model(st["dates"][i]))
                    date = st["dates"][i]
                    if st["dates"][0] <= 150:
                        shift = model(date+365)
                    else:
                        shift = model(date)

                    print("Clock shifted by ", shift, " s")
                    shift_int = int(shift*d_rate)

                    st["stream"][i].data = np.roll(st["stream"][i].data, -1*shift_int)
                    #st["stream"][i].plot()

            if filter:
                st["stream"].filter(type="bandpass", freqmin=f1, freqmax=f2, zerophase=True)
                st["stream"].detrend(type="simple")
                st["stream"].taper(max_percentage=0.05)

            if trim:
                st["stream"].trim(starttime=starttime_new, endtime=endtime_new)
                tr.trim(starttime=starttime_new, endtime=endtime_new)

            rows = len(st["dates"])
            columns = len(st["stream"][0].data)
            matrix = np.zeros([rows, columns])
            matrix = self._fill_matrix(matrix, st)
            matrix_stack = stack(matrix, method=stack_type)
            #matrix_stack = np.mean(matrix, axis=0)


            if plot:
                self._plot_matrix(matrix, st, tr, matrix_stack, correct=correct)

            if save:
                tr1 = tr[0]
                tr1.data = matrix_stack
                self.mapping(tr1, output_path=self.output_path, format=format)
            #except:
            #    print("Coudn't do ", file_stack)




if __name__ == "__main__":
    root_path = "/Volumes/LaCie/UPFLOW_5HZ/toy/final_stack/stack_pcc"
    path_to_polynom = "/Volumes/LaCie/UPFLOW_5HZ/toy/polynom/UP09_UP13_join"
    plots = "/Volumes/LaCie/UPFLOW_5HZ/toy/final_stack/plots"
    outputs = "/Volumes/LaCie/UPFLOW_5HZ/toy/final_stack/stack_synch_pcc"
    pe = SynchEnergy(root_path, path_to_polynom, output_path=outputs, plots=plots)
    #"linear", "pws", "robust", "acf", "nroot", "selective",
    #"cluster", "tfpws","tfpws-dost"
    pe.create_matrix(filter=False, correct=True, plot=True, save=True,
                     trim=False, stack_type="robust", f1=0.033, f2=0.333, format="H5")
