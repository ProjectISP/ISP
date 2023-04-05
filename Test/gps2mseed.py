from obspy import UTCDateTime
import numpy as np
from obspy import Trace, Stream
import os

def read_data(path_in):
    data = np.loadtxt(path_in)
    E = data[:,2]
    N = data[:, 3]
    Z = data[:, 4]
    data = [E,N,Z]
    return data

def convertGPS2mseed(data, path_out):
    chn = ["E", "N", "Z"]

    for i, item in enumerate(data):

        stats = {}
        stats['network'] = "WM"
        stats['station'] = "ALHU"
        stats['channel'] = "BB" + chn[i]
        stats['sampling_rate'] = 10
        stats['npts'] = len(item)
        stats['starttime'] = UTCDateTime("2023-03-20T15:30:00.0")
        # stats['info'] = {'geodetic': [dist, bazim, azim],'cross_channels':file_i[-1]+file_j[-1]}
        st = Stream([Trace(data=item, header=stats)])
        # Nombre del fichero = XT.STA1_STA2.BHZE
        filename = stats['network']+"."+stats['station']+"."+stats['channel']
        path_name = os.path.join(path_out, filename)
        print(path_name)
        st.write(path_name, format='MSEED')

if __name__ == "__main__":
    path_in = '/Users/admin/Desktop/sismo/enu_sismo.txt'
    path_out = '/Users/admin/Desktop/sismo/'
    data = read_data(path_in)
    convertGPS2mseed(data, path_out)

