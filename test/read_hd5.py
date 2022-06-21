from obspy import Trace, Stream
from obspy import read, UTCDateTime
import os
path = '/Users/robertocabieces/Documents/ISP/Test/test_data/COR_ULPC.XT._VIPC.XT._ZZ'
name = os.path.basename(path)
#'geodetic': [dist, bazim, azim]
st = read(path)
tr = st[0]

stats = {}
stats['network'] = 'XT'
stats['station'] = 'ULPC' + "_" + 'VIPC'
stats['channel'] = 'ZZ'
stats['sampling_rate'] = tr.stats.sampling_rate
stats['npts'] = tr.stats.npts
stats['mseed'] = {'dataquality': 'D', 'geodetic': [float(tr.stats.sac['dist']*1000), float(tr.stats.sac['baz']), float(tr.stats.sac['az'])], 'cross_channels': 'ZZ',
                  'coordinates': [float(tr.stats.sac['stla']), float(tr.stats.sac['stlo']), float(tr.stats.sac['evla']), float(tr.stats.sac['evlo'])]}
stats['starttime'] = UTCDateTime("2000-01-01T00:00:00.0")
# stats['info'] = {'geodetic': [dist, bazim, azim],'cross_channels':file_i[-1]+file_j[-1]}
st = Stream([Trace(data=tr.data, header=stats)])
tr = st[0]
name = tr.stats.network+"."+tr.stats.station+"."+tr.stats.channel
#tr.plot()
print(tr)
tr.write(name, format='H5')
# Nombre del fichero = XT.STA1_STA2.BHZE
#filename = file_i[:2] + "." + file_i[2:6] + "_" + file_j[2:6] + "." + file_i[-1] + file_j[-1]
#path_name = os.path.join(self.stack_files_path, filename)
#print(path_name)


#st = read("/Users/robertocabieces/Desktop/desarrollo/test_venezuela/output_test/stack/*.*")
#tr = st[0]

#print(tr.stats.station,tr.stats.mseed['geodetic'][0],tr.stats.mseed['cross_channels'])
