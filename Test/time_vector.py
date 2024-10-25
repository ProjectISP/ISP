from obspy import read, UTCDateTime
import os
import numpy as np

path_test = ("/Users/robertocabieces/Documents/desarrollo/test_venezuela/data/2004/CAPC/BHZ.M/XT.CAPC..BHZ.M.2004.102")
st = read(path_test, format = 'mseed')
tr = st[0]
delta_t = 900
start = tr.stats.starttime
year = start.year
month = start.month
day = start.day
print(start,year,month,day)
check_starttime = UTCDateTime(year=year, month=month, day=day, hour = 00, minute=00, microsecond=00)
check_endtime = check_starttime+24*3600
tr2 = tr.copy()
tr2.trim(starttime=check_starttime, endtime=check_endtime, pad=True, nearest_sample=True, fill_value=0)
print(tr)
print(tr2)

#tr.plot()
#tr2.plot()
step = int((24*3600)/900)
values = np.linspace(0,24*3600, step, endpoint=False)

for increment in values:
    check_starttime = check_starttime+increment
    print(check_starttime)

