import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from scipy.interpolate import interp1d


def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"

    a = np.array(a)
    idx = np.abs(a - a0).argmin()
    return idx


def convert_date_to_julday(date):

    date_jul = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').timetuple().tm_yday

    return date_jul+365

def get_straigt_line(date_jul, days_recorded, gps):

    date_ini = date_jul-days_recorded
    x = np.linspace(date_ini, date_jul, 200)
    y = np.linspace(0, gps, 200)
    deg = 1
    p = np.polyfit(x, y, deg)
    m = p[0]
    c = p[1]
    p = np.flip(p)
    print(f'The fitted straight line has equation y = {m:.3f}x {c:=+6.3f}')
    # straight_line{"polynom":p,}
    return p

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    x = np.array(x)
    o = len(coeffs)
    #coeffs = np.flip(coeffs)
    #print(f'# This is a polynomial of order {o-1}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

def summarize_polynoms(final_curve):

     for key, value in final_curve.items():
         key_split = key.split("_")
         if key_split[-1] == "ref":
            ref_polynom = final_curve[key][2]

     for key, value in final_curve.items():
         key_split = key.split("_")
         if key_split[-1] != "ref":
            final_curve[key][2] = final_curve[key][2] + ref_polynom

     return final_curve

path_file = "/Users/robertocabieces/Documents/UPFLOW/final_clock/clock_final"

obsfiles = []
for top_dir, sub_dir, files in os.walk(path_file):
    for file in files:
        if os.path.basename(file) != ".DS_Store":
            obsfiles.append(os.path.join(top_dir, file))

#obsfiles = [f for f in listdir(path_file) if isfile(join(path_file, f))]

def find_ref(obsfiles, obs, gps):
    land = ["CMLA", "ADHL", "BART", "CALA", "HORL", "PDAL", "PGRA", "PICO", "PMAR", "PMOZ", "PMPST",
            "PSMN", "ROSA", "SRBC"]

    final_curve = {}
    min_dates = []
    max_dates = []
    for idx, file in enumerate(obsfiles):
        x = re.search(obs, file)
        if x:
            #print(obsfiles[idx])
            name = os.path.basename(obsfiles[idx])
            df = pd.read_pickle(obsfiles[idx])
            sta = name.split("_")
            sta1 = sta[0]
            sta2 = sta[1]
            if sta1 in land or sta2 in land:
                print(df)
                y = PolyCoefficients(df["Dates_selected"], df[name])
                ref = PolyCoefficients(df["Dates_selected"][0], df[name])
                y = y - ref
                fig, ax1 = plt.subplots(figsize=(12, 6))
                fig.suptitle(name+" "+"Drift " + str(gps), fontsize=16)
                plt.plot(df["Dates_selected"], y)
                plt.scatter(df["Dates_selected"], df['Drift']-ref, c="red", marker='o', edgecolors='k', s=18)
                plt.ylabel('Skew [s]')
                plt.xlabel('Jul day')
                plt.show()
                print("Do you want to add this curve?, y or n? ")
                answer = input()
                if answer == "y":
                    min_dates.append(min(df["Dates_selected"]))
                    max_dates.append(max(df["Dates_selected"]))
                    ref_polynom = df[name]
                    name = name + "_" + "ref"
                    final_curve[name] = [df["Dates_selected"], y, ref_polynom]
                else:
                    pass

        if len(final_curve) > 0:

            final_curve["min_max_dates"] = [max(min_dates), min(max_dates)]

    return final_curve

def plot_ref_sum(final_curve, gps_skew, date_end, days_recorded):

    ref_curve = {}
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex = True, sharey = True, figsize=(12, 10))
    fig.suptitle("Sum References", fontsize=16)
    common_dates = np.arange(final_curve["min_max_dates"][0],final_curve["min_max_dates"][1],1)
    y_def = np.zeros(len(common_dates))
    j = 0
    for key, value in final_curve.items():
        if key != "min_max_dates":
            y1 = final_curve[key]
            y1 = PolyCoefficients(common_dates, y1[2])
            y_def = y_def+y1
            j = j+1
        axs[0].plot(common_dates, y1, alpha = 0.5)
    y_def = y_def/j
    date_end = convert_date_to_julday(date_end)
    date_ini = date_end - days_recorded
    straigt_line = get_straigt_line(date_end, days_recorded, gps)
    water_level = PolyCoefficients(common_dates[0], straigt_line)
    y_def = y_def + water_level

    #axs[0,0].plot(common_dates_new, y_model, color='red', label="including gps skew")
    axs[0].axhline(y=gps_skew, color='grey', linestyle='--', alpha = 0.5)
    axs[0].axvline(x=date_ini, color='grey', linestyle='--', alpha = 0.5)
    axs[0].plot(common_dates, y_def, color = 'black', label = "sum_ref")
    axs[0].axhline(y=water_level, color='grey', linestyle='--', alpha = 0.5)
    axs[0].axvline(x=date_end, color='grey', linestyle='--', alpha=0.5)
    axs[0].legend()
    axs[0].set_ylabel('Skew [s]')
    axs[0].set_xlabel('Jul day')

    # build the final ref model

    y_def_new = np.append(y_def, gps_skew)
    y_def_new = np.insert(y_def_new, 0, 0, axis=0)
    common_dates_new = np.append(common_dates, date_end)
    common_dates_new = np.insert(common_dates_new, 0, date_ini, axis=0)
    interp_func = interp1d(common_dates_new, y_def_new)
    common_dates_new_interp = np.linspace(date_ini, date_end, 500)
    y_def_new_interp = interp_func(common_dates_new_interp)
    p = np.polyfit(common_dates_new_interp, y_def_new_interp, 6)
    # Create the linear (1 degree polynomial) model
    model = np.poly1d(p)
    # Fit the model
    y_model = model(common_dates_new_interp)
    #fig.suptitle("All", fontsize=16)
    axs[1].plot(common_dates_new_interp, y_model, alpha=0.5, label = "sum_ref")
    plt.show()

    ref_curve["data"] = y_model
    ref_curve["dates"] = common_dates_new_interp

    return ref_curve



def find_obs_pair(obs_pair, obsfiles):
    file_ini = []
    file_end = []
    for idx, file in enumerate(obsfiles):

        x = re.search(obs_pair[0], os.path.basename(file))

        if x:
            file_ini.append(file)

    for idx, file in enumerate(file_ini):
        match = obs_pair[1]
        finding = os.path.basename(file)
        x = re.search(match, finding)
        #print(x)
        if x:
            file_end.append(file)

    return file_end


def build_rel_curve(file, gps_skew, date_end, days_recorded):
    single_curve = {}
    name = os.path.basename(file)
    df = pd.read_pickle(file)
    print(df)
    y = PolyCoefficients(df["Dates_selected"], df[name])
    ref = PolyCoefficients(df["Dates_selected"][0], df[name])
    y = y - ref
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle(name + " " + "Final Drift Curve" + str(gps_skew), fontsize=16)
    plt.plot(df["Dates_selected"], y)
    plt.scatter(df["Dates_selected"], df['Drift'] - ref, c="red", marker='o', edgecolors='k', s=18)
    plt.ylabel('Skew [s]')
    plt.xlabel('Jul day')

    # correct the curve

    date_end = convert_date_to_julday(date_end)
    date_ini = date_end - days_recorded
    straigt_line = get_straigt_line(date_end, days_recorded, gps_skew)
    water_level = PolyCoefficients(df["Dates_selected"][0], straigt_line)
    y_def = y + water_level
    y_def_new = np.append(y_def, gps_skew)
    y_def_new = np.insert(y_def_new, 0, 0, axis=0)
    common_dates_new = np.append(df["Dates_selected"], date_end)
    common_dates_new = np.insert(common_dates_new, 0, date_ini, axis=0)
    interp_func = interp1d(common_dates_new, y_def_new)
    common_dates_new_interp = np.linspace(date_ini, date_end, 500)
    y_def_new_interp = interp_func(common_dates_new_interp)
    p = np.polyfit(common_dates_new_interp, y_def_new_interp, 6)
    # Create the linear (1 degree polynomial) model
    model = np.poly1d(p)
    # Fit the model
    y_model = model(common_dates_new_interp)
    #fig, ax = plt.subplots(figsize=(12, 6))
    #fig.suptitle("All", fontsize=16)
    plt.plot(common_dates_new_interp, y_model, color="black")
    plt.show()

    single_curve["data"] = y_model
    single_curve["dates"] = common_dates_new_interp

    return single_curve

def join_curves(ref_curve, single_curve):

    # get date range

    min_date_ref = min(ref_curve["dates"])
    min_date_single = min(single_curve["dates"])

    max_date_ref = max(ref_curve["dates"])
    max_date_single = max(single_curve["dates"])

    lim_min_date = max([min_date_ref, min_date_single])
    lim_max_date = min([max_date_ref, max_date_single])

    # chop ref_curve
    idx_min = find_nearest(single_curve["dates"], lim_min_date)
    idx_max = find_nearest(single_curve["dates"], lim_max_date)

    single_dates = single_curve["dates"][idx_min:idx_max]
    single_data = single_curve["data"][idx_min:idx_max]


    interp_func = interp1d(single_curve["dates"], single_curve["data"])

    common_dates_new_interp = np.linspace(single_dates[0], single_dates[-1], 500)
    y_def_new_interp = interp_func(common_dates_new_interp)

    # join curves

    y_join = y_def_new_interp - ref_curve["data"]

    p = np.polyfit(common_dates_new_interp, y_join, 6)
    # Create the linear (1 degree polynomial) model
    model = np.poly1d(p)
    # Fit the model
    y_model = model(common_dates_new_interp)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle("UP11" + " " + "Drift ", fontsize=16)
    plt.plot(common_dates_new_interp, y_model)
    plt.ylabel('Skew [s]')
    plt.xlabel('Jul day')
    plt.show()

def join_polynoms(ref_obs, list_of_pairs, obsfiles, gps_skew, date_end, days_recorded):
    ref_polynom = find_ref(obsfiles, ref_obs, gps_skew)
    ref_curve = plot_ref_sum(ref_polynom, gps_skew, date_end, days_recorded)
    for obs_pair in list_of_pairs:
        file = find_obs_pair(obs_pair, obsfiles)
        print(file)
        single_curve = build_rel_curve(file[0], obs_pair[2], obs_pair[3], obs_pair[4])
        join_curves(ref_curve, single_curve)

ref_obs = "UP12"
gps = 0.62
days_recorded = 385
date_end = '2022-08-11T00:00:00'
#list_of_pairs = [["UP12", "X25", 4.52, '2022-08-13T00:00:00', 319], ["UP12", "UP11", 8.56, '2022-08-10T00:00:00', 386],
#                  ["UP12", "UP13", -0.79, '2022-08-13T00:00:00'], 387]
#list_of_pairs = [["UP12", "UP11", 8.56, '2022-08-10T00:00:00', 386], ["UP12", "UP13", -0.79, '2022-08-13T00:00:00', 386]]
list_of_pairs = [["UP12", "UP11", 8.56, '2022-08-10T00:00:00', 386]]
join_polynoms(ref_obs, list_of_pairs, obsfiles, gps, date_end, days_recorded)