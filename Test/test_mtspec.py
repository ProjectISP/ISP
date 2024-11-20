#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_mtspec



:param : 
:type : 
:return: 
:rtype: 
"""

import os
from obspy import read
from isp import ROOT_DIR
#from multitaper import MTSpec as spec
import multitaper.mtspec as spec
path = os.path.join(ROOT_DIR,"examples/Earthquake_location_test","ES.EADA..HHZ.D.2015.260")
tr = read(path)[0]
#tr.plot()
t2, freq, QIspec, MTspec = spec.spectrogram(tr.data, tr.stats.delta, twin=20., olap=0.5,
                               nw=3.5, kspec=5,fmin=0.05, fmax=20.)
#psd = MTSpec(tr.data, dt=tr.stats.sampling_rate)

