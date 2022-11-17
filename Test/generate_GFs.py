from obspy.clients.syngine import Client
from obspy import UTCDateTime
client = Client()

# bulk = [{"network": "IU", "station": "ANMO"}, {"latitude": 47.0, "longitude": 12.1}]

bulk = [{"network": "IU", "station": "ANMO", "latitude": 47.0, "longitude": 12.1},
        {"network": "IU", "station": "OTAV", "latitude": 0.24, "longitude": -78.45}]
origin = UTCDateTime("2022-10-20TT11:57:20")
start = origin
endtime = UTCDateTime("2022-10-20TT12:27:20")
print(start)
print(endtime)
print(endtime-start)

st = client.get_waveforms_bulk(model="prem_a_10s", bulk=bulk, sourcelatitude=7.671,sourcelongitude=-82.340,
                               sourcedepthinmeters=10000, origintime=origin, starttime=start,endtime=endtime,
                               units= "velocity", sourcemomenttensor=[0, 0, 0, 0, 0, 1E21], components="Z")


print(st)
st.plot()