import math
import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from isp.ant.signal_processing_tools import noise_processing

class JoinClocks():
    def __init__(self, input_path, obs_pair):

        """
        polynom = {clocks_station_name: p.tolist(), 'Dates': dates, 'Dates_selected': x, 'Drift': y, 'Ref': ref,
                   'R2': R2, 'resid': resid, 'chi2_red': chi2_red, 'std_err': std_err, 'cross_correlation': cc,
                   'skew': skew}
        """

        self.join_curve = None
        self.input_path = input_path
        self.obspair = obs_pair


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
        date_jul = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').timetuple().tm_yday

        return date_jul

    def PolyCoefficients(self, x, coeffs):
        """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

        The coefficients must be in ascending order (``x**0`` to ``x**o``).
        """
        x = np.array(x)
        o = len(coeffs)
        # coeffs = np.flip(coeffs)
        # print(f'# This is a polynomial of order {o-1}.')
        y = 0
        for i in range(o):
            y += coeffs[i] * x ** i
        return y

    def get_straigt_line(self,gps_skew_1, gps_skew_2, date_ini_1 = '2021-07-16T00:00:00',
                         date_ini_2 = '2021-07-16T00:00:00',  date_end_1 = '2022-08-05T00:00:00',
                         date_end_2 = '2022-08-05T00:00:00'):
        jul_date_ini_1 = self.convert_date_to_julday(self, date_ini_1) + 365
        jul_date_end_1 = self.convert_date_to_julday(self, date_end_1)

        jul_date_ini_2 = self.convert_date_to_julday(self, date_ini_1) + 365
        jul_date_end_2 = self.convert_date_to_julday(self, date_end_1)

        jul_date_ini = max(jul_date_ini_1, jul_date_ini_2)
        jul_date_end = min(jul_date_end_1, jul_date_end_2)
        gps_skew = gps_skew_2-gps_skew_1
        x = np.array([jul_date_ini, jul_date_end])
        y = np.array([0, gps_skew])
        deg = 1
        p = np.polyfit(x, y, deg)


        m = p[0]
        c = p[1]
        p = np.flip(p)
        print(f'The fitted straight line has equation y = {m:.3f}x {c:=+6.3f}')
        # straight_line{"polynom":p,}
        return p

    def create_grid(self):
        min_dates = []
        max_dates = []
        join_curve = {}
        obs_pair = self.obspair.split("_")
        for file in self.file_end:
            name = os.path.basename(file)
            sta1 = name.split("_")[0]
            sta2 = name.split("_")[1]
            chn = name.split("_")[2]
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
                    join_curve[str(value)].append([drift[i],cc[i], chn])

        #join_curve["min_max_dates"] = [max(min_dates), min(max_dates)]
        join_curve = collections.OrderedDict(sorted(join_curve.items()))
        self.join_curve = join_curve
        self.correct_grid()

    def create_join_polynom(self, plot = True):
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
        m, n, R2, p, y_model, model, c, t_critical, resid, chi2_red, std_err, x, y = \
                                 noise_processing.statisics_fit(all_dates, avarage1, "Polynom", 3)


        self.plot_polynom_scatter(x, y_model)

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

    def plot_polynom_scatter(self, x, y):

        fig, ax = plt.subplots(figsize=(12, 8))
        title = "Skew drift "+self.obspair+"  // "+str(-1*self.skew[0])+" "+str(-1*self.skew[1])
        fig.suptitle(title, fontsize=16)


        self.ZZ, self.HH, self.RR, self.TT = self.extract_components()
        ZZ, HH, RR, TT = self.ZZ, self.HH, self.RR, self.TT
        self.correct_grid()

        if len(self.ZZ) > 0:
            cs = ax.scatter(ZZ[0][0], ZZ[0][1], c=ZZ[0][2], marker='o', edgecolors='k', s=18, vmin=0.0, vmax=1.0,
                            label='ZZ')
        #
        if len(self.HH) > 0:
            cs = ax.scatter(HH[0][0], HH[0][1], c=HH[0][2], marker='*', edgecolors='k', s=18, vmin=0.0, vmax=1.0,
                            label='HH')

        if len(self.RR) > 0:
            cs = ax.scatter(RR[0][0], RR[0][1], c=RR[0][2], marker="v", edgecolors='k', s=18, vmin=0.0, vmax=1.0,
                            label='RR & TT')

        if len(self.TT) > 0:
            cs = ax.scatter(TT[0][0], TT[0][1], c=TT[0][2], marker=">", edgecolors='k', s=18, vmin=0.0, vmax=1.0)


        ax.plot(x, y, linewidth = 1.5, color="red", alpha = 0.5)
        ax.legend(handler_map={cs: HandlerLine2D(numpoints=1)})
        fig.colorbar(cs, ax=ax, orientation='horizontal', fraction=0.05,
                                                       extend='both', pad=0.15, label='Normalized Cross Correlation')

        plt.ylabel('Skew [s]')
        plt.xlabel('Jul day')
        plt.show()


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


if __name__ == "__main__":
    input_path = "/Users/admin/Documents/Documentos - iMac de Admin/clock_dir_def/all_components"
    # example
    obs_pair = "UP12_UP13"
    date_ini_1 = '2021-07-16T00:00:00'
    date_ini_2 = '2021-07-16T00:00:00'
    date_end_1 = '2022-08-05T00:00:00'
    date_end_2 = '2022-08-05T00:00:00'
    # example
    jc = JoinClocks(input_path, obs_pair)
    jc.list_dir()
    jc.find_obs_pair()
    jc.create_grid()
    jc.create_join_polynom()
