import os
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from isp.ant.signal_processing_tools import noise_processing


def list_all_dir(input_path):
    obsfiles = []
    for top_dir, sub_dir, files in os.walk(input_path):
        for file in files:
            if os.path.basename(file) != ".DS_Store":
                obsfiles.append(os.path.join(top_dir, file))

    return obsfiles
def list_stations(files):
    stations =[]
    for name in files:
        try:
            name = os.path.basename(name)
            info = name.split("_")
            sta1 = info[0]
            sta2 = info[1]
            if sta1 not in stations:
                stations.append(sta1)
            elif sta2 not in stations:
                stations.append(sta2)
        except:
            pass

    return stations
def plot_results(input_path, station_list):
    data = np.loadtxt(input_path, dtype=np.str)
    fig, ax = plt.subplots(figsize=(16, 12))
    station_list.sort()
    for i, sta in enumerate(station_list):
        sta_idx = i
        for k in range(len(data[:, 0])):

            name = data[k, 1].split("_")
            sta1 = name[0]
            sta2 = name[1]
            if sta == sta1 or sta == sta2:
                y = data[k, 4]
                y = np.abs(y.astype(np.float))
                z = data[k, 5]
                z = z.astype(np.float)
                cs = ax.scatter(sta_idx, y, c=z, marker='o', edgecolors='k', s=24, vmin=1.0, vmax=3.0,
                                alpha=0.5)

    major_ticks = np.arange(0, len(station_list), 1)
    #ax.xaxis.set_ticklabels(station_list, rotation=90)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(labels=station_list, rotation=90)
    ax.set_ylim([0, 0.5])
    plt.ylabel('Skew [s]')
    plt.xlabel('Station')
    fig.colorbar(cs, ax=ax, orientation='horizontal', fraction=0.05, ticks=np.arange(1, 4),
                 extend='both', pad=0.15, spacing='proportional', label='Order Polynomial')
    plt.show()
        #y = x.astype(np.float)

class CheckSkews():
    def __init__(self, input_path, obs_pair, dates_path, plots_output, output):

        """
        polynom = {clocks_station_name: p.tolist(), 'Dates': dates, 'Dates_selected': x, 'Drift': y, 'Ref': ref,
                   'R2': R2, 'resid': resid, 'chi2_red': chi2_red, 'std_err': std_err, 'cross_correlation': cc,
                   'skew': skew}
        """

        self.join_curve = None
        self.input_path = os.path.join(input_path,obs_pair)
        self.obspair = obs_pair
        self.dates_path = dates_path
        self.plots_output = plots_output
        self.output = output

    def find_nearest(self, a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"

        a = np.array(a)
        idx = np.abs(a - a0).argmin()
        return idx

    def convert_date_to_julday(self, date):

        # example of use date_end = '2022-08-11T00:00:00'
        date_jul = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').timetuple().tm_yday

        return date_jul

    def list_dir(self):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.input_path):
            for file in files:
                if os.path.basename(file) != ".DS_Store":
                    obsfiles.append(os.path.join(top_dir, file))

        self.obsfiles = obsfiles


    def retrive_dates(self):
        land_list = ["PGRA", "ADHB", "CALA", "ROSA", "SRBC", "PMOZ", "PMAR", "PMPS", "HORB", "PICO"]
        df_dates = pd.read_csv(self.dates_path, sep="\t", index_col="Station")
        obs_pair = self.obspair.split("_")
        sta1 = obs_pair[0]
        sta2 = obs_pair[1]
        date_ini_1 = self.convert_date_to_julday(df_dates.loc[sta1]['Deploy Date'])
        date_ini_2 = self.convert_date_to_julday(df_dates.loc[sta2]['Deploy Date'])

        date_end_1 = self.convert_date_to_julday(df_dates.loc[sta1]['Recovery'])
        date_end_2 = self.convert_date_to_julday(df_dates.loc[sta2]['Recovery'])

        if sta1 in land_list:
            date_ini_1 = date_ini_2
            date_end_1 = date_end_2
        elif sta2 in land_list:
            date_ini_2 = date_ini_1
            date_end_2 = date_end_1

        self.date_ini = max([date_ini_1, date_ini_2])
        self.date_end = min([date_end_1, date_end_2])+365

    def PolyCoefficients(self, x, coeffs):
        """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

        The coefficients must be in ascending order (``x**0`` to ``x**o``).
        """
        x = np.array(x)
        o = len(coeffs)
        #coeffs = np.flip(coeffs)
        # print(f'# This is a polynomial of order {o-1}.')
        y = 0
        for i in range(o):
            y += coeffs[i] * x ** i
        return y

    def estimate_error(self, revaluate=False, degree=3):
        """
        polynom = {clocks_station_name: p.tolist(), 'Dates': dates, 'Dates_selected': x, 'Drift': y, 'Ref': ref,
                           'R2': R2, 'resid': resid, 'chi2_red': chi2_red, 'std_err': std_err, 'cross_correlation': cc,
                           'skew': skew}
        """
        self.df_polynom = pd.read_pickle(self.input_path)
        self.order = degree
        # reavaluate polynom
        if revaluate:
            m, n, R2, p, y_model, model, c, t_critical, resid, chi2_red, std_err, ci, pi, x_new, y_new = \
                noise_processing.statisics_fit(self.df_polynom['Dates_selected'], self.df_polynom['Drift'], "Polynom", degree)
            self.df_polynom[self.obspair] = np.flip(p).tolist()


        self.polynom = self.df_polynom[self.obspair]
        default_ini = self.df_polynom['Drift'][0]
        self.skew_estimated_ini = self.PolyCoefficients(self.date_ini, self.polynom)-default_ini
        self.skew_estimated_end = self.PolyCoefficients(self.date_end, self.polynom)-default_ini
        self.err_skew_ini = self.skew_estimated_ini
        self.skew = self.df_polynom["skew"]
        #self.err_skew_end = self.skew_estimated_end-(self.df_polynom["skew"][1]-self.df_polynom["skew"][0]) - self.df_polynom['Drift'][0]
        try:
            self.skew2 = float(self.df_polynom["skew"][1])
        except:
            self.skew2 = 0.0

        try:
            self.skew1 = float(self.df_polynom["skew"][0])
        except:
            self.skew1 = 0.0

        self.err_skew_end = self.skew_estimated_end - (self.skew2-self.skew1)
        print(self.err_skew_ini, self.err_skew_end)

    def plot_polynom(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        title = "Skew drift " + self.obspair + "  // " + str(self.skew1) + " && " + str(self.skew2) + " = "+  str(self.skew2-self.skew1)
        fig.suptitle(title, fontsize=16)
        dates_selected = self.df_polynom["Dates_selected"]
        data = self.df_polynom['Drift'] - self.df_polynom['Drift'][0]-self.err_skew_ini
        cc = self.df_polynom['cross_correlation']
        ax.scatter(dates_selected, data, c=cc, marker='o', edgecolors='k', s=18, vmin=0.0, vmax=1.0,
                        label='ZZ')

        # End Points
        cs = ax.scatter(self.date_ini, 0, c="blue", marker='o', edgecolors='k', s=24, vmin=0.0, vmax=1.0, alpha= 0.5)
        cs = ax.scatter(self.date_end, self.skew2-self.skew1, c="blue", marker='o',
                        edgecolors='k', s=24, vmin=0.0, vmax=1.0, alpha=0.5)

        # Extrapolated
        cs = ax.scatter(self.date_ini, self.err_skew_ini-self.err_skew_ini, c="red", marker='o', edgecolors='k', s=24, vmin=0.0, vmax=1.0,
                        alpha= 0.5)
        cs = ax.scatter(self.date_end, self.skew_estimated_end-self.err_skew_ini, c="red", marker='o',
                        edgecolors='k', s=24, vmin=0.0, vmax=1.0, alpha = 0.5)


        # Plot Polynom
        polynom_points = self.PolyCoefficients(dates_selected, self.polynom)
        ax.plot(dates_selected, polynom_points - self.df_polynom['Drift'][0]-self.err_skew_ini, linewidth = 1.5, color="red", alpha = 0.5)
        skew_ini = "{:.4f}".format(self.err_skew_ini)
        skew_end = "{:.4f}".format(self.err_skew_end)
        skew_overall = "{:.4f}".format(self.err_skew_end - self.err_skew_ini)
        ax.annotate('Diff Skew Ini  ' + str(skew_ini), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(0.80, 0.19), textcoords='axes fraction', va='top', ha='left')
        ax.annotate('Diff Skew End  ' + str(skew_end), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(0.80, 0.15), textcoords='axes fraction', va='top', ha='left')

        ax.annotate('Diff Skew Overall  ' + str(skew_overall), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(0.80, 0.11), textcoords='axes fraction', va='top', ha='left')

        ax.annotate('Order Polynomial  ' + str(self.order), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(0.80, 0.05), textcoords='axes fraction', va='top', ha='left')
        #ax.legend(handler_map={cs: HandlerLine2D(numpoints=1)})
        fig.colorbar(cs, ax=ax, orientation='horizontal', fraction=0.05,
                                                       extend='both', pad=0.15, label='Normalized Cross Correlation')

        plt.ylabel('Skew [s]')
        plt.xlabel('Jul day')

        file_name = os.path.join(self.plots_output, self.obspair)+"_"+str(self.order)+".pdf"
        plt.savefig(file_name, dpi=150)
        line = 'Skew ' + self.obspair + " " + str(skew_ini) + " " + str(skew_end)+" " + str(skew_overall) + " " + str(self.order)
        output_file = os.path.join(self.output, "skews.txt")
        with open(output_file, 'a') as f:
            f.write(line)
            f.write('\n')
        #plt.show()

if __name__ == "__main__":
    input_path = "/Users/admin/Documents/Documentos - iMac de Admin/clock_dir_def/vertical_component"
    skews_path = "/Users/admin/Documents/Documentos - iMac de Admin/clock_dir_def/skews/skews.txt"
    plots_output = "/Users/admin/Documents/Documentos - iMac de Admin/clock_dir_def/plots"
    output = "/Users/admin/Documents/Documentos - iMac de Admin/clock_dir_def/output"
    # example
    #obs_pair = "UP16_UP18_ZZ"

    # example

    # list_of_files = list_all_dir(input_path)
    # for obs_pair in list_of_files:
    #     try:
    #         obs_pair = os.path.basename(obs_pair)
    #         cs = CheckSkews(input_path, obs_pair, skews_path, plots_output, output)
    #         cs.retrive_dates()
    #         cs.estimate_error(revaluate=True, degree=1)
    #         cs.plot_polynom()
    #     except:
    #         pass

    list_of_files = list_all_dir(input_path)
    stations_list = list_stations(list_of_files)
    path_skew_def = os.path.join(output, "skews.txt")
    plot_results(path_skew_def, stations_list)