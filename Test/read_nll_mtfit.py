import MTfit
import pandas as pd
from isp.Utils import read_nll_performance

from isp.Utils import ObspyUtil
path_hyp = "/isp/earthquakeAnalysis/location_output/loc/last.hyp"
#path_hyp = "/Users/admin/Documents/ISP/Test/test_data/last.hyp"
#path_hyp = "/Users/admin/Documents/test_meli/test_data/24_12_2022_1828.hyp"
#data = MTfit.MTfit(data_file = path_hyp, parallel = False)
#event_dict_list = MTfit.utilities.file_io.parse_hyp(path_hyp)
#df = pd.read_csv(path_hyp, delimiter=r"\s+", skiprows=16, false_values=[">"])
#picks = ObspyUtil.reads_pick_info(path_hyp)

cat= read_nll_performance.read_nlloc_hyp_ISP(path_hyp)
event = cat[0]
arrivals = event["origins"][0]["arrivals"]
for time_arrival in arrivals:
    print(time_arrival.azimuth)


import MTfit
from MTfit.plot import read, MTplot
# from MTfit.inversion import Inversion
#from MTfit.plot import MTplot
import numpy as np
# #from example_data import double_couple_data
# # Set parameters
# algorithm = 'iterate'  # uses an iterative random sampling approach
# parallel = False  # Runs on a dingle thread.
# phy_mem = 1  # uses a soft limit of 1Gb of RAM for estimating the sample sizes (This is only a soft limit, so no errors are thrown if the memory usage increases above this)
# dc = True  # runs the inversion in the double-couple space.
# max_samples = 1000  # runs the inversion for 100,000 samples.
# #path_hyp = "/Users/admin/Documents/test_meli/test_data/24_12_2022_1828.hyp"
# path_hyp = "/Users/admin/Documents/iMacROA/ISP/Test/test_data/last1.hyp"
# #event_dict_list = MTfit.utilities.file_io.parse_hyp(path_hyp)
# #data = MTfit.MTfit(data_file=path_hyp, parallel = False)
# inversion_object = Inversion(data_file=path_hyp,
#                                  algorithm=algorithm, parallel=parallel,
#                                  phy_mem=phy_mem, dc=dc,
#                                  max_samples=max_samples, convert=True)
#     # Run the forward model based inversion
# inversion_object.forward()
DCs,DCstations = read('/Users/admin/Documents/iMacROA/ISP/Test/test_data/2015091715114442408DC.mat')
print(DCs)
plot = MTplot([np.array([1,0,0,0,0,0]),DCs],
    stations=DCstations,
    station_distribution=False,
    plot_type='faultplane', fault_plane=False,
    show_mean=True, show_max=True,grid_lines=True,TNP=False,text=True, save_file = "test.png")

print(plot)
#plot.save_file("test.png")
#plot.show()
#plot=MTfit.plot.MTplot(DCs)
#MTplot(MTs,plot_type='beachball',stations={},plot=True,*args,**kwargs)