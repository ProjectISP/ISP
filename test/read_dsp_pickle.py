from isp import DISP_MAPS
import os
import pickle





def check_items(dsp_map):
    items = ['period', 'paths', 'rejected_paths', 'ref_velocity', 'alpha0', 'alpha1', 'beta', 'sigma', 'm_opt_relative',
             'm_opt_absolute', 'grid', 'resolution_map', 'cov_map', 'residuals', 'rms']

    check_list = []
    for key in dsp_map[0].keys():
        print(key)

        check_list.append(key)

    return check_list == items



path = os.path.join(DISP_MAPS,"ZZ","dsp","pickles", "dispersion_ZZ_dsp_10.0s.pickle")
dsp_map = pickle.load(open(path, "rb" ) )

print(check_items(dsp_map))
#for key in dsp_map[0].items():
#    print(key)