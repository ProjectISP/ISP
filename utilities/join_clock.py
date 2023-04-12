import math
import os
import pickle
import re
from datetime import datetime
import pandas as pd
import numpy as np
import collections
from matplotlib import pyplot as plt
#from matplotlib.legend_handler import HandlerLine2D
from isp.ant.signal_processing_tools import noise_processing

class JoinClocks():
    def __init__(self, input_path, skews_path, output_path, obs_pair, components, degree = 3):

        """
        polynom = {clocks_station_name: p.tolist(), 'Dates': dates, 'Dates_selected': x, 'Drift': y, 'Ref': ref,
                   'R2': R2, 'resid': resid, 'chi2_red': chi2_red, 'std_err': std_err, 'cross_correlation': cc,
                   'skew': skew}
        """

        self.join_curve = None
        self.input_path = input_path
        self.obspair = obs_pair
        self.skews_path = skews_path
        self.output_path = output_path
        self.components = components
        self.order = degree

    def retrive_dates(self):

        land_list = ["PGRA", "ADHB", "CALA", "ROSA", "SRBC", "PMOZ", "PMAR", "PMPS", "HORB", "PICO"]
        df_dates = pd.read_csv(self.skews_path, sep="\t", index_col="Station")
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


    def retrieve_skews(self):
        df_dates = pd.read_csv(self.skews_path, sep="\t", index_col="Station")
        obs_pair = self.obspair.split("_")
        sta1 = obs_pair[0]
        sta2 = obs_pair[1]
        self.skew1 = df_dates.loc[sta1]['Skew']*1E-6
        self.skew2 = df_dates.loc[sta2]['Skew']*1E-6


    def estimate_error(self):

        self.retrive_dates()
        self.retrieve_skews()
        self.skew_estimated_ini = self.PolyCoefficients(self.date_ini, self.polynom)
        self.skew_estimated_end = self.PolyCoefficients(self.date_end, self.polynom)
        self.err_skew_ini = self.skew_estimated_ini
        self.err_skew_end = self.skew_estimated_end - (self.skew2 - self.skew1)


    def find_nearest(self, a, a0):

        "Element in nd array `a` closest to the scalar value `a0`"

        a = np.array(a)
        idx = np.abs(a - a0).argmin()
        return idx

    def list_dir(self):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(self.input_path):
            for file in files:
                if os.path.basename(file) != ".DS_Store":
                    obsfiles.append(os.path.join(top_dir, file))

        self.obsfiles = obsfiles

    def find_obs_pair(self):
        obs_pair = self.obspair.split("_")
        file_ini = []
        file_end = []
        for idx, file in enumerate(self.obsfiles):

            x = re.search(obs_pair[0], os.path.basename(file))

            if x:
                file_ini.append(file)

        for idx, file in enumerate(file_ini):
            match = obs_pair[1]
            finding = os.path.basename(file)
            x = re.search(match, finding)
            # print(x)
            if x:
                file_end.append(file)
        print(file_end)
        self.file_end = file_end
        df = pd.read_pickle(self.file_end[0])
        self.skew = df["skew"]

    def convert_date_to_julday(self, date):

        # example of use date_end = '2022-08-11T00:00:00'
        date_jul = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').timetuple().tm_yday

        return date_jul

    def PolyCoefficients(self, x, coeffs):
        """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

        The coefficients must be in ascending order (``x**0`` to ``x**o``).
        """
        x = np.array(x)
        o = len(coeffs)
        coeffs = np.flip(coeffs)
        # print(f'# This is a polynomial of order {o-1}.')
        y = 0
        for i in range(o):
            y += coeffs[i] * x ** i
        return y

    def create_grid(self, revaluate = False):
        min_dates = []
        max_dates = []
        join_curve = {}
        obs_pair = self.obspair.split("_")
        for file in self.file_end:
            name = os.path.basename(file)
            sta1 = name.split("_")[0]
            sta2 = name.split("_")[1]
            chn = name.split("_")[2]
            if chn in self.components:
                df = pd.read_pickle(file)
                min_dates.append(min(df["Dates_selected"]))
                max_dates.append(max(df["Dates_selected"]))
                dates_selected = df["Dates_selected"]
                drift = df['Drift']-df['Drift'][0]

                if sta2 == obs_pair[0] and sta1 == obs_pair[1]:
                    drift = -1*drift

                cc = df['cross_correlation']

                # fill dictionary
                for i, value in enumerate(dates_selected):
                    if str(value) not in join_curve:
                        join_curve[str(value)] = [[drift[i], cc[i], chn]]
                    else:
                        join_curve[str(value)].append([drift[i], cc[i], chn])

        #join_curve["min_max_dates"] = [max(min_dates), min(max_dates)]
        join_curve = collections.OrderedDict(sorted(join_curve.items()))
        self.join_curve = join_curve
        self.correct_grid()

    def create_join_polynom(self):
        avarage1 = []
        avarage2 = []
        all_dates = []

        #TODO correct for minimumdate

        # 1 Loop over days
        for key, value in self.join_curve.items():
            num1 = []
            den1 = []
            num2 = []
            den2 = []
            all_dates.append(float(key))
            # Loop over components daily skew
            for j_term in value:
                check = j_term[2]
                if check in self.components:
                    drif_j = j_term[0]
                    cc_j = j_term[1]**2
                    cc_jj = j_term[1]**3
                    num1.append(cc_j*drif_j)
                    den1.append(cc_j)
                    num2.append(cc_jj)
                    den2.append(cc_j)

            num1 = np.mean(np.array(num1))
            den1 = np.mean(np.array(den1))
            factor1 = num1/den1
            if math.isnan(factor1):
                factor = 0.0
            avarage1.append(factor1)

            num2 = np.mean(np.array(num2))
            den2 = np.mean(np.array(den2))
            factor2 = num2/den2
            avarage2.append(factor2)
        avarage1 = [x for _, x in sorted(zip(all_dates, avarage1))]
        all_dates.sort()
        m, n, R2, p, y_model, model, c, t_critical, resid, chi2_red, std_err,ci,pi, x, y = \
                                 noise_processing.statisics_fit(all_dates, avarage1, "Polynom", self.order)

        self.m = m
        self.n = n
        self.R2 = R2
        self.polynom = p
        self.y_model = y_model
        self.model = model
        self.c = c
        self.t_critical = t_critical
        self.resid = resid
        self.chi2_red = chi2_red
        self.ci = ci
        self.pi = pi
        self.std_err = std_err
        self.all_dates = all_dates
        self.avarage1 = avarage1
        self.estimate_error()
        self.plot_polynom_scatter(x, y_model, ci, pi)


    def extract_components(self):
        # First plot the points might be better in another method
        ZZ = []
        ZZ_dates = []
        ZZ_cross = []
        ZZ_drift = []
        RR = []
        RR_dates = []
        RR_cross = []
        RR_drift = []
        HH = []
        HH_dates = []
        HH_cross = []
        HH_drift = []
        TT = []
        TT_dates = []
        TT_cross = []
        TT_drift = []
        for key, value in self.join_curve.items():
            for j_term in value:
                if j_term[2] == "ZZ":
                    ZZ_dates.append(float(key))
                    ZZ_drift.append(j_term[0])
                    ZZ_cross.append(j_term[1])
                if j_term[2] == "HH":
                    HH_dates.append(float(key))
                    HH_drift.append(j_term[0])
                    HH_cross.append(j_term[1])
                if j_term[2] == "RR":
                    RR_dates.append(float(key))
                    RR_drift.append(j_term[0])
                    RR_cross.append(j_term[1])
                if j_term[2] == "TT":
                    TT_dates.append(float(key))
                    TT_drift.append(j_term[0])
                    TT_cross.append(j_term[1])

        if len(ZZ_dates)>0:
            ZZ.append([ZZ_dates, ZZ_drift, ZZ_cross])
        if len(HH_dates)>0:
            HH.append([HH_dates, HH_drift, HH_cross])
        if len(RR_dates)>0:
            RR.append([RR_dates, RR_drift, RR_cross])
        if len(TT_dates)>0:
            TT.append([TT_dates, TT_drift, TT_cross])



        return ZZ, HH, RR, TT

    def plot_polynom_scatter(self, x, y, ci, pi):

        fig, ax = plt.subplots(figsize=(12, 8))
        title = "Skew drift "+self.obspair+"  // "+str(-1*self.skew[0])+" "+str(-1*self.skew[1])
        fig.suptitle(title, fontsize=16)

        self.ZZ, self.HH, self.RR, self.TT = self.extract_components()

        # set last correction
        y = y - self.err_skew_ini
        if len(self.ZZ) > 0:
            self.ZZ = self.ZZ - self.err_skew_ini
        if len(self.HH) > 0:
            self.HH = self.HH - self.err_skew_ini
        if len(self.RR) > 0:
            self.RR = self.RR - self.err_skew_ini
        if len(self.TT) > 0:
            self.TT = self.TT - self.err_skew_ini

        ZZ, HH, RR, TT = self.ZZ, self.HH, self.RR, self.TT
        self.correct_grid()

        self.skew_overall = self.err_skew_end - self.err_skew_ini
        self.skew_estimated_end_original = self.skew_estimated_end
        self.skew_estimated_end = self.skew_estimated_end - self.err_skew_ini
        self.err_skew_ini_original = self.err_skew_ini
        self.err_skew_ini = self.err_skew_ini - self.err_skew_ini



        if len(self.ZZ) > 0:

            cs = ax.scatter(ZZ[0][0], ZZ[0][1], c=ZZ[0][2], marker='o', edgecolors='k', s=18, vmin=0.0, vmax=1.0,
                            label='ZZ')
        #
        if len(self.HH) > 0:
            cs = ax.scatter(HH[0][0], HH[0][1], c=HH[0][2], marker='*', edgecolors='k', s=18, vmin=0.0, vmax=1.0,
                            label='HH')

        if len(self.RR) > 0:
            cs = ax.scatter(RR[0][0], RR[0][1], c=RR[0][2], marker="v", edgecolors='k', s=18, vmin=0.0, vmax=1.0,
                            label='RR')

        if len(self.TT) > 0:
            cs = ax.scatter(TT[0][0], TT[0][1], c=TT[0][2], marker=">", edgecolors='k', s=18, vmin=0.0, vmax=1.0,
                            label='TT')


        ax.plot(x, y, linewidth = 1.5, color="red", alpha = 0.5)
        ax.fill_between(x, y + ci, y - ci, facecolor='#b9cfe7', zorder=0,label=r'95 % Confidence Interval')
        plt.plot(x, y - pi, '--', color='0.5', label=r'95 % Prediction Limits')
        plt.plot(x, y + pi, '--', color='0.5')

        # End Points
        ax.scatter(self.date_ini, 0, c="blue", marker='o', edgecolors='k', s=24, vmin=0.0, vmax=1.0, alpha=0.5)
        ax.scatter(self.date_end, self.skew2 - self.skew1, c="blue", marker='o',
                        edgecolors='k', s=24, vmin=0.0, vmax=1.0, alpha=0.5)

        # Extrapolated
        ax.scatter(self.date_ini, self.err_skew_ini, c="red", marker='o', edgecolors='k', s=24, vmin=0.0, vmax=1.0,
                        alpha=0.5)
        ax.scatter(self.date_end, self.skew_estimated_end, c="red", marker='o',
                        edgecolors='k', s=24, vmin=0.0, vmax=1.0, alpha=0.5)

        skew_ini = "{:.4f}".format(self.err_skew_ini_original)
        skew_end = "{:.4f}".format(self.err_skew_end)
        skew_overall = "{:.4f}".format(self.skew_overall)

        ax.annotate('Diff Skew Ini  ' + str(skew_ini), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(0.80, 0.19), textcoords='axes fraction', va='top', ha='left')
        ax.annotate('Diff Skew End  ' + str(skew_end), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(0.80, 0.15), textcoords='axes fraction', va='top', ha='left')

        ax.annotate('Diff Skew Overall  ' + str(skew_overall), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(0.80, 0.11), textcoords='axes fraction', va='top', ha='left')

        ax.annotate('Order Polynomial  ' + str(self.order), xy=(0.1, 0.1), xycoords='axes fraction',
                    xytext=(0.80, 0.05), textcoords='axes fraction', va='top', ha='left')

        #ax.legend(handler_map={cs: HandlerLine2D(numpoints=1)})
        ax.legend()
        fig.colorbar(cs, ax=ax, orientation='horizontal', fraction=0.05,
                                                       extend='both', pad=0.15, label='Normalized Cross Correlation')

        plt.ylabel('Skew [s]')
        plt.xlabel('Jul day')
        file_name = os.path.join(self.output_path, self.obspair)+".pdf"
        plt.savefig(file_name, dpi=150)
        plt.show()
        self.save_results()


    def check_components(self):

        ZZ, HH, RR, TT = self.ZZ, self.HH, self.RR, self.TT
        check_list = []
        min_list = []
        if len(ZZ)>0:
            min_date_z = min(ZZ[0][0])
            min_list.append(min_date_z)
            check_list.append("ZZ")

        if len(HH)>0:
            min_date_h = min(HH[0][0])
            min_list.append(min_date_h)
            check_list.append("HH")

        if len(RR)>0:
            min_date_r = min(RR[0][0])
            min_list.append(min_date_r)
            check_list.append("RR")

        if len(TT)>0:
            min_date_t = min(TT[0][0])
            min_list.append(min_date_t)
            check_list.append("TT")

        check_list = [x for _, x in sorted(zip(min_list, check_list))]
        min_list.sort()

        return check_list, min_list

    def correct_grid(self):

        components_check = []
        ref_hydro = 0
        ref_tt = 0
        ref_rr = 0
        for key, value in self.join_curve.items():
            for index, item in enumerate(value):
                if item[2] == "HH" and item[2] not in components_check:
                    ref_vertical = self.find_nearest_ref(key)
                    ref_hydro = ref_vertical
                    components_check.append("HH")

        for key, value in self.join_curve.items():
            for index, item in enumerate(value):
                if item[2] == "TT" and item[2] not in components_check:
                    ref_tt = self.find_nearest_ref(key)
                    components_check.append("TT")

        for key, value in self.join_curve.items():
            for index, item in enumerate(value):
                if item[2] == "RR" and item[2] not in components_check:
                    ref_rr = self.find_nearest_ref(key)
                    components_check.append("RR")

        for key, value in self.join_curve.items():
            for index, item in enumerate(value):
                if item[2] == "HH":
                    item[0] = item[0] + ref_hydro
                    self.join_curve[key][index] = item
                elif item[2] == "RR":
                    item[0] = item[0] + ref_rr
                    self.join_curve[key][index] = item
                elif item[2] == "TT":
                    item[0] = item[0] + ref_tt
                    self.join_curve[key][index] = item


    def find_nearest_ref(self, key_to_find):
        ref_vertical = 0
        for key, value in self.join_curve.items():
            #tolerance = float(key) - float(key_to_find)

            if key == key_to_find:
                for index, item in enumerate(value):
                    if item[2] == "ZZ":
                        ref_def = self.join_curve[key][index][0]
                    else:
                        ref_def = ref_vertical

            else:
                for index, item in enumerate(value):
                    if item[2] == "ZZ":
                        ref_vertical = self.join_curve[key][index][0]

        return ref_def


    def save_results(self):
        print("Do you want to save this curve?, y or n? ")
        answer = input()
        if answer == "y":
            name = self.obspair + "_" + "join"
            polynom = {name: self.polynom.tolist(), 'Dates': self.all_dates, 'Drift': self.avarage1, 'R2': self.R2,
                't_critical':self.t_critical, 'resid': self.resid, 'chi2_red': self.chi2_red, 'std_err': self.std_err,
            'confidence_interval':self.ci, 'prediction_interval': self.pi, 'c': self.c, 'm': self.m, 'n': self.n,
                       'model': self.model, 'y_model': self.y_model, 'skews': [self.skew1, self.skew2],
                       'juldays': [self.date_ini,self.date_end], 'components': self.components, 'order': self.order,
                       'skews_diff': [self.err_skew_ini_original, self.skew_estimated_end_original, self.skew_overall]}

            path = os.path.join(self.output_path, name)
            file_to_store = open(path, "wb")
            pickle.dump(polynom, file_to_store)

            file_name = os.path.join(self.output_path, self.obspair) + "_" + str(self.order) + ".pdf"
            plt.savefig(file_name, dpi=150)
            line = 'Skew ' + self.obspair + " " + str(self.err_skew_ini_original) + " " + str(self.err_skew_end) + " " + str(
                self.skew_overall) + " " + str(self.order) + " " + str(self.R2) + " " + str(self.std_err)
            output_file = os.path.join(self.output_path, "skews.txt")
            with open(output_file, 'a') as f:
                f.write(line)
                f.write('\n')
        else:
            print("Curve not saved")


if __name__ == "__main__":
    input_path = "/Users/robertocabieces/Documents/iMacROA/clock_dir_def/all_components_new"
    output_path = "/Users/robertocabieces/Documents/iMacROA/clock_dir_def/output_join"
    skews_path = "/Users/robertocabieces/Documents/iMacROA/clock_dir_def/skews/skews.txt"
    # example
    #list_of_files = list_all_dir(input_path)
    obs_pair = "X25H_UP15"
    components = ["ZZ", "HH", "RR", "TT"]
    order = 1
    # example
    jc = JoinClocks(input_path, skews_path, output_path, obs_pair, components, order)
    jc.list_dir()
    jc.find_obs_pair()
    jc.create_grid()
    jc.create_join_polynom()
