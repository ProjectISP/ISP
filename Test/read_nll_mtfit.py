#import MTfit
#import pandas as pd
from isp.Utils import read_nll_performance

#from isp.Utils import ObspyUtil

#path_hyp = "/Users/admin/Documents/ISP/Test/test_data/last.hyp"
path_hyp = "/Users/admin/Documents/test_meli/test_data/24_12_2022_1828.hyp"
#data = MTfit.MTfit(data_file = path_hyp, parallel = False)
#event_dict_list = MTfit.utilities.file_io.parse_hyp(path_hyp)
#df = pd.read_csv(path_hyp, delimiter=r"\s+", skiprows=16, false_values=[">"])
#picks = ObspyUtil.reads_pick_info(path_hyp)

cat= read_nll_performance.read_nlloc_hyp_ISP(path_hyp)
event = cat[0]
arrivals = event["origins"][0]["arrivals"]
for time_arrival in arrivals:
    print(time_arrival.azimuth)