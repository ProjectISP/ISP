import os
from obspy.clients.syngine import Client
from obspy import UTCDateTime, read
from obspy import read_inventory
import numpy as np



class generate_query:

    def __init__(self, data_path: str, inventory_file_path: str, output_root_path: str):

        self.bulk = []
        self.data_path = data_path
        self.inventory = read_inventory(inventory_file_path)
        self.output_root_path =  output_root_path
        print(self.inventory)
        # sources [m_rr, m_tt, m_pp, m_rt,m_rp, m_tp]
        source1 = [0.0, 0.0, 0.0, 0.0, 0.0, -1E21]
        source2 = [0.0, 0.0, 0.0, 1E21, 0.0, 0.0]
        source3 = [0.0, 0.0, 0.0, 0.0, 1E21, 0.0]
        source4 = [1E21, -1E21, 0, 0, 0, 0]
        source5 = [1E21, 0.0, -1E21, 0.0, 0.0, 0.0]
        source6 = [1E21, 1E21, 1E21, 0.0, 1E21, 0.0]
        self.sources = [source1, source2, source3, source4, source5, source6]
        self.dir_names = ["GFs1", "GFs2", "GFs3", "GFs4", "GFs5", "GFs6"]

    def get_tree_mseed_files(self):

        """
        Get a list of valid mseed files inside all folder tree from the the root_dir.
        If root_dir doesn't exists it returns a empty list.
        :param root_dir: The full path of the dir or a file.
        :return: A list of full path of mseed files.
        """

        data_files = []
        for top_dir, sub_dir, files in os.walk(self.data_path):
            for file in files:
                try:
                    st = read(os.path.join(top_dir, file), headonly=True)
                    print(st)
                    data_files.append(os.path.join(top_dir, file))
                except:
                    pass

        data_files.sort()
        self.data_files = data_files
        return data_files

    def extract_coordinates_from_trace(self):
        all_net_stations = []

        for file in self.data_files:
            st = read(file, headonly=True)
            tr = st[0]
            network = tr.stats.network
            station = tr.stats.station
            location = tr.stats.location
            channel = tr.stats.channel
            starttime = tr.stats.starttime
            endtime = tr.stats.endtime
            net_station = network+"."+station
            try:
                selected_inv = self.inventory.select(network=network, station=station, location=location,channel=channel,
                                                starttime=starttime, endtime=endtime)

                cont = selected_inv.get_contents()
                coords = selected_inv.get_coordinates(cont['channels'][0])

                if net_station not in all_net_stations:

                    self.bulk.append({"networkcode": network, "stationcode": station, "locationcode": location,
                                      "latitude": coords["latitude"], "longitude": coords["longitude"]})
                    all_net_stations.append(net_station)
            except:
                pass

    def do_query_simple(self, model, sourcelatitude: float, sourcelongitude: float, sourcedepthinmeters: float, origintime: UTCDateTime, starttime : UTCDateTime,
                 endtime: UTCDateTime):
        client = Client()
        # First loop over source --> loop around source point
        for sc in self.sources:
            st = client.get_waveforms_bulk(model=model, bulk=self.bulk, sourcelatitude=sourcelatitude,sourcelongitude=sourcelongitude,
                                    sourcedepthinmeters=sourcedepthinmeters, origintime=origintime, starttime=starttime,endtime=endtime,
                                    units= "velocity", sourcemomenttensor=sc)
            print(st)
            #st.plot()

    def do_query_full(self, model, sourcelatitude: float, sourcelongitude: float, sourcedepthinmeters: float,
                        origintime: UTCDateTime, starttime: UTCDateTime,
                        endtime: UTCDateTime, Lx, Ly, Lz, dx = 2, dy = 2, dz = 2):

        client = Client()
        depths = np.arange(int(sourcedepthinmeters-Lz*1000), int(sourcedepthinmeters+Lz*1000), dz*1000)
        longs = np.arange(sourcelongitude - (Lx/112), sourcelongitude + (Lx/112), (dx/112))
        lats = np.arange(sourcelatitude - (Ly / 112), sourcelatitude + (Ly / 112), (dy/112))
        # First loop over source --> loop around source point
        for count, sc in enumerate(self.sources):

            path_for_GFs = os.path.join(self.output_root_path, self.dir_names[count])
            if not os.path.exists(path_for_GFs):
                os.makedirs(path_for_GFs)

            for zz in depths: # fixed depth ready to fix lat
                for yy in lats: # fixed longs ready to fix lat
                    for xx in longs: # loop over lats
                        print(zz, yy, xx)
                        st = client.get_waveforms_bulk(model=model, bulk=self.bulk, sourcelatitude=yy,
                                                        sourcelongitude=xx,
                                                        sourcedepthinmeters=zz, origintime=origintime,
                                                        starttime=starttime, endtime=endtime,
                                                        units="velocity", sourcemomenttensor=sc)

                        # saving
                        for tr in st:
                            file_name = tr.id + "." + str("{:.1f}".format(zz))+"_"+str("{:.4f}".format(yy))+"_"+\
                                        str("{:.4f}".format(xx))
                            name = os.path.join(path_for_GFs, file_name)
                            print("saving ", name)
                            tr.write(name, format="MSEED")


data_path = "/Volumes/NO NAME/teleseismic_eartuqueake/data_example"
metadata_path = "/Volumes/NO NAME/teleseismic_eartuqueake/Pacific/metadata/iris_metadata"
output_root_path = "/Users/robertocabieces/Documents/desarrollo/test_GFs"
query = generate_query(data_path, metadata_path, output_root_path)
query.get_tree_mseed_files()
query.extract_coordinates_from_trace()
print(query.bulk)
#
origin = UTCDateTime("2022-10-20TT11:57:20")
start = origin
endtime = UTCDateTime("2022-10-20TT12:27:20")
print(start)
print(endtime)
print(endtime-start)

# query.do_query_simple(model="prem_a_10s", sourcelatitude=7.671,sourcelongitude=-82.340, sourcedepthinmeters=10000,
#                  origintime=origin, starttime=start, endtime=endtime)

query.do_query_full(model="prem_a_10s", sourcelatitude=7.671, sourcelongitude=-82.340, sourcedepthinmeters=10000,
                    origintime=origin, starttime=start, endtime=endtime, Lx= 5, Ly = 5, Lz =5)
# #
#
# bulk = [{"networkcode": "WM", "stationcode": "XRY", "latitude": 47.0, "longitude": 12.1}]
# client = Client()
# st = client.get_waveforms_bulk(model="prem_a_10s", bulk=bulk, sourcelatitude=7.671,sourcelongitude=-82.340,
#                                 sourcedepthinmeters=10000, origintime=origin, starttime=start,endtime=endtime,
#                                 units= "velocity", sourcemomenttensor=[0, 0, 0, 0, 0, 1E21], components="Z")
# #
# #
# print(st)
# st.plot()