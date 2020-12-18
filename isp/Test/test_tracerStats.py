import os
from unittest import TestCase

from obspy import read
from obspy.io.xseed import Parser

from isp import ROOT_DIR
from isp.Structures.structures import TracerStats, StationsStats
from isp.Utils import MseedUtil
from isp.seismogramInspector.readNLLevent import getNLLinfo


class TestTracerStats(TestCase):

    def setUp(self):
        dir_path = os.path.join(ROOT_DIR, "260", "Velocity")
        self.file = os.path.join(dir_path, "WM.OBS01..SHZ.D.2015.260")

    def test_from_dict(self):
        st = read(self.file)
        stats = st[0].stats
        print(stats.keys())
        trace_stat = TracerStats.from_dict(stats)
        fields_list_lower = [k.lower() for k in trace_stat._fields]
        print(stats.keys())
        for k in stats.keys():
            if k == "_format":
                self.assertEqual(stats.get(k), trace_stat.Format)
            else:
                index = fields_list_lower.index(k.lower())  # Compare all in lower case. Avoid Caps sensitive.
                safe_key = trace_stat._fields[index]
                self.assertEqual(stats.get(k), trace_stat.to_dict().get(safe_key))


    def test_getNLLinfo(self):
        path = os.path.join(ROOT_DIR, "260", "Locations", "2015-09-17.hyp")
        time, latitude, longitude, depth = getNLLinfo(path)
        print(time, latitude, longitude, depth)

    def test_dataless(self):
        dir_path = os.path.join(ROOT_DIR, "260", "dataless", "datalessOBS01.dlsv")
        parser = Parser()
        parser.read(dir_path)
        station_blk = parser.stations[0][0]
        station_dict = {"Name": station_blk.station_call_letters, "Lon": station_blk.longitude,
                        "Lat": station_blk.latitude, "Depth": station_blk.elevation}
        station_stats = StationsStats.from_dataless(dir_path)
        self.assertDictEqual(station_dict, station_stats.to_dict())
