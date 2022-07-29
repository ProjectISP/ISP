import os
import unittest
from datetime import datetime

from obspy import UTCDateTime
from obspy.core.event import Origin

from isp.Structures.structures import Search
from isp.Utils import ObspyUtil
from isp.db import db
from isp.earthquakeAnalisysis import PickerManager

dir_path = os.path.dirname(os.path.abspath(__file__))
db.set_db_url("sqlite:///{}/isp_test.db".format(dir_path))
db.start()

from isp.db.models import FirstPolarityModel, MomentTensorModel, EventArrayModel, PhaseInfoModel,\
    EventLocationModel, ArrayAnalysisModel


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # create all tables into db.
        db.create_all()

    @classmethod
    def tearDownClass(cls) -> None:
        # remove db file when test is done.
        db.remove_sqlite_db()

    def test_event_location(self):
        print("run test")
        print(EventLocationModel.get_all())
        print(FirstPolarityModel.get_all())
        print(MomentTensorModel.get_all())
        print(ArrayAnalysisModel.get_all())
        print(EventArrayModel.get_all())
        print(PhaseInfoModel.get_all())

    def test_event_location_insert(self):
        hyp_file = os.path.join(dir_path, "test_data", "last.hyp")
        origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file)
        try:
            event_model = EventLocationModel.create_from_origin(origin)
            phases = PhaseInfoModel.create_phases_from_origin(origin, event_model.id)
            for phase in phases:
                phase.save()
                event_model.add_phase(phase)
            event_model.save()
            event_model: EventLocationModel = EventLocationModel.find_by_id(event_model.id)
            self.assertIsNotNone(event_model)
            print("Insert")
        except AttributeError:
            print("Already insert")

    def test_find_by(self):
        self.test_event_location_insert()
        date_time = datetime(2015, 9, 17, 15, 11, 44, 424088)
        event_model = EventLocationModel.find_by(origin_time=date_time, transformation="SIMPLE")
        self.assertIsNotNone(event_model)

    def test_search(self):
        self.test_event_location_insert()
        search = Search(SearchBy="latitude, longitude", SearchValue="35, -7",
                        Page=1, PerPage=10, OrderBy="latitude", MapColumnAndValue=True)
        self.assertIsNotNone(EventLocationModel.search(search).result[0])

        start_date = UTCDateTime("2015-05-08T14:01:58.112160Z").datetime
        end_date = UTCDateTime("2016-06-08T14:01:58.112160Z").datetime
        string_query = "(latitude BETWEEN 30 and 38) and " \
                       "(longitude BETWEEN -10 and -6) and " \
                       "origin_time BETWEEN '{}' and '{}'".format(start_date, end_date)
        search = Search(SearchBy="", SearchValue="", Page=1, PerPage=10, OrderBy="id",
                        TextualQuery=string_query, MapColumnAndValue=True)
        self.assertEqual(1, len(EventLocationModel.search(search).result))


if __name__ == '__main__':
    unittest.main()
