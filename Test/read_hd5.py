from obspy import read

st = read("/Users/robertocabieces/Desktop/desarrollo/test_venezuela/output_test/stack/*.*")
tr = st[0]

print(tr.stats.station,tr.stats.mseed['geodetic'][0],tr.stats.mseed['cross_channels'])
