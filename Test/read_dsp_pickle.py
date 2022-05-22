from isp import DISP_MAPS
import os
import pickle


def read_dispersion(file, wave_type='ZZ', dispersion_type="dsp"):
    disp_data = {}
    dict_ = pickle.load(open(file, "rb"))

    for pair in dict_.keys():
        stnm1, stnm2, wave, type_ = pair.split("_")
        if type_ == dispersion_type and wave == wave_type:
            disp_data.setdefault((stnm1, stnm2), {})
            disp_data[(stnm1, stnm2)]['c(f)'] = [float(x) for x in dict_[pair]['velocity']]
            disp_data[(stnm1, stnm2)]['f'] = [float(x) for x in dict_[pair]['period']]

    return disp_data


def check_items(dsp_map):
    items = ['period', 'paths', 'rejected_paths', 'ref_velocity', 'alpha0', 'alpha1', 'beta', 'sigma', 'm_opt_relative',
             'm_opt_absolute', 'grid', 'resolution_map', 'cov_map', 'residuals', 'rms']

    check_list = []
    for key in dsp_map[0].keys():
        print(key)

        check_list.append(key)

    return check_list == items

file = "/Users/robertocabieces/Documents/ISP/Test/test_data/disp_maps.pkl"
disp_data = read_dispersion(file, wave_type='TT', dispersion_type="phv")
print(disp_data)
#path = os.path.join(DISP_MAPS,"ZZ","dsp","pickles", "dispersion_ZZ_dsp_10.0s.pickle")
#dsp_map = pickle.load(open(path, "rb" ) )

#print(check_items(dsp_map))
#for key in dsp_map[0].items():
#    print(key)