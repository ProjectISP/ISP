#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from obspy.io.sac import SACTrace
from stackmaster.core import stack
import glob
from matplotlib import font_manager as fm


class SynchEnergy():
    def __init__(self, path_ref, path_to_polynom, output_path, skew_path, plots):

        self.obsfiles_daily = []
        self.obsfiles =[]
        self.input_path = path_ref
        self.path_to_polynom = path_to_polynom
        self.output_path = output_path
        self.skews_path =skew_path
        self.plots = plots
        self.model = None
        self.black_list = ["UP26", "UP19", "OBS19", "OBS17"]

    def _convert_date_to_julday(self, date):

        # example of use date_end = '2022-08-11T00:00:00'
        date_jul = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').timetuple().tm_yday

        if date_jul > 365:
            date_jul = date_jul+365

        return date_jul

    def _retrieve_skews(self, sta):
        df_dates = pd.read_csv(self.skews_path, sep="\t", index_col="Station")

        skew = df_dates.loc[sta]['Skew']*1E-6
        date_ini = df_dates.loc[sta]['Deploy Date']
        date_end = df_dates.loc[sta]['Recovery']
        jul_day_ini = self._convert_date_to_julday(date_ini)
        jul_day_end = self._convert_date_to_julday(date_end)

        return skew, jul_day_ini, jul_day_end

    def retrive_poly1d(self, sta):

        skew, jul_day_ini, jul_day_end = self._retrieve_skews(sta)
        k = skew/(jul_day_end-jul_day_ini)
        xo = k*jul_day_ini
        poly = np.poly1d([k, -k*xo])

        return poly

    def list_dir(self, input_path):

        obsfiles_daily = []
        for top_dir, sub_dir, files in os.walk(input_path):
            for file in files:
                if os.path.basename(file) != ".DS_Store":
                    obsfiles_daily.append(os.path.join(top_dir, file))

        return obsfiles_daily

    def _fill_matrix(self, matrix, st):

        for i, item in enumerate(st["stream"]):
            matrix[i, :] = st["stream"][i].data
        return matrix

    def get_polynom(self, sta1, sta2):
        """
        Finds and reads pickled data files matching the specified patterns.

        Parameters:
        - sta1 (str): The pattern to match for the first file.
        - sta2 (str): The pattern to match for the second file.

        Returns:
        - tuple: A tuple containing two pandas DataFrames read from the matching files.
                 Returns (None, None) if no files are found for a pattern.
        """
        # Construct search patterns with wildcards
        pattern_sta1 = f"*{sta1}*"
        pattern_sta2 = f"*{sta2}*"

        # Use glob to find files matching the patterns
        matching_paths_sta1 = glob.glob(f'{self.path_to_polynom}/{pattern_sta1}')
        matching_paths_sta2 = glob.glob(f'{self.path_to_polynom}/{pattern_sta2}')

        # Initialize variables to hold dataframes
        model1 = np.poly1d([0])
        model2 = np.poly1d([0])

        if sta1 in self.black_list:
            model1 = self.retrive_poly1d(sta1)

        if sta2 in self.black_list:
            model2 = self.retrive_poly1d(sta2)

        # Check if files were found for sta1
        if matching_paths_sta1:
            matching_path_sta1 = matching_paths_sta1[0]
            df1 = pd.read_pickle(matching_path_sta1)
            model1 = df1["model"]
        # Check if files were found for sta2
        if matching_paths_sta2:
            matching_path_sta2 = matching_paths_sta2[0]
            df2 = pd.read_pickle(matching_path_sta2)
            model2 = df2["model"]



        model = model1-model2

        return model


    def run_all_synch(self, stack_type="linear", filter = False, plot=False, save=False):

        daily_files_list = self.list_dir(self.input_path)
        daily_files_list.sort()
        for file in daily_files_list:
            try:
                st = pd.read_pickle(file)
                filename = os.path.basename(file).split("_")
                sta1 = filename[0].split(".")[1]
                sta2 = filename[1].split(".")[0]
                print(sta1, sta2)
                model = self.get_polynom(sta1, sta2)
                self._create_matrix(st, model, filter=filter, plot=plot, save=save, stack_type=stack_type)
            except Exception as e:
                print(f"Error processing file {file}: {e}")


    def _create_matrix(self, st, model, trim=False, filter=False, plot=False, save=False, stack_type="robust"):

        starttime = st["stream"][0].stats.starttime
        endtime = st["stream"][0].stats.endtime
        d_rate = st["stream"][0].stats.sampling_rate

        central_time = starttime + (endtime - starttime) / 2

        if trim:
            starttime_new = central_time - 400
            endtime_new = central_time + 400

        st["stream"].detrend(type="simple")
        st["stream"].taper(max_percentage=0.05)

        st_raw = copy.deepcopy(st)

        for i in range(len(st["stream"])):
            # st["stream"][i].data = np.roll(st["stream"][i].data, self.model(st["dates"][i]))
            date = st["dates"][i]
            # if st["dates"][0] <= 150:
            #     shift = model(date + 365)
            # else:
            #     shift = model(date)
            shift = model(date)

            shift_int = int(shift * d_rate)
            st["stream"][i].data = np.roll(st["stream"][i].data, shift_int)

        if filter:
            st["stream"].filter(type="bandpass", freqmin=0.033, freqmax=0.333, zerophase=True)
            st["stream"].detrend(type="simple")
            st["stream"].taper(max_percentage=0.05)


            st_raw["stream"].filter(type="bandpass", freqmin=0.033, freqmax=0.333, zerophase=True)
            st_raw["stream"].detrend(type="simple")
            st_raw["stream"].taper(max_percentage=0.05)

        if trim:
            st["stream"].trim(starttime=starttime_new, endtime=endtime_new)
            st["stream"].detrend(type="simple")
            st["stream"].taper(max_percentage=0.05)


            st_raw["stream"].trim(starttime=starttime_new, endtime=endtime_new)
            st_raw["stream"].detrend(type="simple")
            st_raw["stream"].taper(max_percentage=0.05)


        rows = len(st["dates"])
        columns = len(st["stream"][0].data)

        matrix = np.zeros([rows, columns])
        matrix = self._fill_matrix(matrix, st)
        matrix_stack = stack(matrix, method=stack_type)

        matrix_raw = np.zeros([rows, columns])
        matrix_raw = self._fill_matrix(matrix_raw, st_raw)
        matrix_stack_raw = stack(matrix_raw, method="linear")

        if plot:
            self._plot_matrix(matrix_stack_raw, matrix_stack, matrix, st, d_rate)

        if save:
            tr1 = st["stream"][0]
            tr1.data = matrix_stack
            #self.mapping(tr1, output_path=self.output_path, format="H5")
            self.mapping(tr1, output_path=self.output_path, format="sac")


    def _plot_matrix(self, matrix_stack_raw, matrix_stack, matrix,  st, d_rate):
        plt.rcParams['figure.constrained_layout.use'] = True
        fs = d_rate

        t = st["stream"][0].times()
        station = st["stream"][0].stats.station
        channel = st["stream"][0].stats.channel
        name = station+"_"+channel

        t_shift = np.roll(t, int(len(t)/2))
        t = t - t_shift
        times = np.arange(np.min(t), np.max(t), 1/fs)

        extent = np.min(t), np.max(t), np.min(st["dates"]), np.max(st["dates"]),
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

        ax0.plot(times, matrix_stack_raw, color="blue", linewidth=0.75, alpha=0.75, label="RAW")
        #ax0.plot(times, stack, color="green", linewidth=0.75)

        ax0.plot(times, matrix_stack, color="green", linewidth=0.75, alpha=0.75, label="SYNCH")
        ax0.set_xlim([np.min(t), np.max(t)])
        c = ax1.imshow(matrix, cmap=plt.cm.seismic, interpolation='gaussian', extent=extent, vmin=-1, vmax=1)
        ax0.legend()

        # Adding custom text at the top-left of the first subplot
        ax0.text(0.05, 0.95, name, transform=ax0.transAxes,
                 fontsize=14, fontweight='bold', va='top', ha='left')
        # Adding a colorbar and setting its label
        colorbar = fig.colorbar(c, ax=ax1, orientation="horizontal")
        colorbar.set_label('Normalized Cross Correlation', fontsize=12, fontweight='bold')

        # Set the colorbar ticks to intervals of 0.5
        colorbar.set_ticks(np.arange(-1, 1.5, 0.5))
        # Define a bold font property
        bold_font = fm.FontProperties(weight='bold', size=12)

        # Manually set bold and larger tick labels for both x and y axes
        for label in ax0.get_xticklabels() + ax0.get_yticklabels():
            label.set_fontproperties(bold_font)

        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontproperties(bold_font)

        # Set bold tick labels for the colorbar (color palette)
        colorbar.ax.tick_params(labelsize=12)  # Set label size for colorbar
        for label in colorbar.ax.get_xticklabels():  # Access colorbar tick labels
            label.set_fontproperties(bold_font)  # Set bold font for colorbar tick labels

        # Adjust x-axis labels to be bold
        #ax0.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (s)', fontsize=14, fontweight='bold')

        # Adjust y-axis labels to be bold
        ax0.set_ylabel('Amplitude', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Dates', fontsize=14, fontweight='bold')

        file_name = os.path.join(self.plots, name)+".pdf"
        plt.savefig(file_name, dpi=250)

        # plt.show()

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
                   'delta': tr_acausal.stats.delta, 'dist': tr_acausal.stats.mseed['geodetic'][0]*1E-3, 'baz':tr_acausal.stats.mseed['geodetic'][1],
                   'az': tr_acausal.stats.mseed['geodetic'][2]}

        name = tr_acausal.stats.network + "." + tr_acausal.stats.station + "." + tr_acausal.stats.channel
        if format == "sac":
            sac = SACTrace(data=tr_causal.data, **header)
            path = os.path.join(output_path, name+"_"+"sac")
            sac.write(path, byteorder='little', flush_headers=True)

        elif format == "H5":

            path_mseed = os.path.join(output_path, name+"_"+"h5")
            tr1.write(path_mseed, format="H5")

if __name__ == "__main__":
    # best test using robust
    #root_path = "/Volumes/LaCie/UPFLOW_5HZ/matrix_z_upflow/stack_daily"
    root_path = "/Volumes/LaCie/UPFLOW_NEW_MATRIX/morocco_new/stack_daily"
    path_to_polynom = "/Volumes/LaCie/UPFLOW_5HZ/poly/"
    plots = "/Volumes/LaCie/UPFLOW_NEW_MATRIX/plots"
    outputs = "/Volumes/LaCie/UPFLOW_NEW_MATRIX/synch/morocco"
    skew_path = "/Volumes/LaCie/UPFLOW_5HZ/poly/skews.txt"

    stack_type = "robust"
    pe = SynchEnergy(root_path, path_to_polynom, output_path=outputs, skew_path=skew_path, plots=plots)
    #"linear", "pws", "robust", "acf", "nroot", "selective",
    #"cluster", "tfpws","tfpws-dost"
    #pe.create_matrix(filter=False, correct=True, plot=True, save=True,
    #                 trim=False, stack_type="robust", f1=0.033, f2=0.333, format="H5")
    pe.run_all_synch(stack_type=stack_type, filter=False, plot=False, save=True)