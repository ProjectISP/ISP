from obspy import read, read_inventory
from obspy import UTCDateTime


inv = read_inventory("/Users/robertocabieces/Documents/ISPshare/isp/examples/boris/meta/4sta_mod.xml")
end = UTCDateTime("2021-01-31T00:00:00.000000Z")
net = inv[0]

for sta in net:

    sta.end_date = end

inv.write("/Users/robertocabieces/Documents/ISPshare/isp/examples/boris/meta/4sta_mod.xml",format="STATIONXML")

#inv = inv.select(network=stats.Network, station=stats.Station, starttime=stats.StartTime, endtime=stats.EndTime)
#inv = inv.select(network="IU", station="COLA", channel = "LHN")

#
#print(inv)


# st = read("/Users/robertocabieces/Documents/ISPshare/isp/examples/boris/IU.SFJD.00.LH*")
#
# for tr in st:
#     print(tr.stats.station,tr.stats.channel)
#     if tr.stats.channel[2] == "N":
#
#         tr.stats.channel = "BHN"
#         tr.write(tr.id, format = "mseed")
#     if tr.stats.channel[2] == "E":
#         tr.stats.channel = "BHE"
#         tr.write(tr.id, format="mseed")
#         tr.write(tr.id, format="mseed")
#     if tr.stats.channel[2] == "Z":
#         tr.stats.channel = "BHZ"
#         tr.write(tr.id, format="mseed")


