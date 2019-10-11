import os
from unittest import TestCase

from obspy import read

from isp import ROOT_DIR
from isp.Structures.structures import TracerStats


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

