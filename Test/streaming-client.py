#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:47:21 2021

@author: sysop
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from obspy.clients.seedlink.easyseedlink import create_client
from obspy import UTCDateTime


def handle_data(tr):
    print(tr.stats.network,tr.stats.station,tr.stats.channel)
    
def main():
    
    client = create_client("alertes.roa.es", on_data=handle_data)
    stations = ('WM.AVE', 'WM.CART', 'WM.EMAL')
    #Add the stations to the request
    for i in stations:
        net, sta = i.split('.')
        client.select_stream(net, sta, 'HH?')

    client.run()

if __name__ == '__main__':
    main()
