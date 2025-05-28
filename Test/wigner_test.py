#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wigner_test



:param : 
:type : 
:return: 
:rtype: 
"""
from obspy import read, UTCDateTime
import numpy as np
from isp.DataProcessing.wiegner import wigner_distribution, fast_wigner_distribution, \
    interference_reduced_wigner_distribution
import matplotlib.pyplot as plt

data_path = "/Users/admin/Documents/iMacROA/ISP/isp/examples/Earthquake_location_test/ES.EADA..HHZ.D.2015.260"
t1 = UTCDateTime('2015-09-17TT15:12:10')
t2 = t1 + 30
tr = read(data_path)[0]
tr.trim(starttime=t1, endtime=t2)
tr.plot()
wd, max_frequency= wigner_distribution(tr.data)
wd_clean = interference_reduced_wigner_distribution(wd)

fig, ax = plt.subplots()
ax.imshow(np.abs(wd_clean),
              extent=(np.min(tr.times()), np.max(tr.times()), 0, 50),
              aspect="auto")

# ax.set_xlabel(xlabel)
# ax.set_ylabel(ylabel)
# ax.set_title(title)
plt.show()