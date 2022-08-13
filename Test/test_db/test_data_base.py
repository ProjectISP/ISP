import unittest

from obspy import UTCDateTime
from obspy.core.event import Origin, OriginUncertainty, OriginQuality

from isp.db import db


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        db.set_db_url('sqlite://')  # create in memory db.
        db.start()
        db.create_all()

    def _create_origin(self, lat: float, long: float, depth: float) -> Origin:
        origin = Origin()
        origin.time = UTCDateTime(0)
        origin.latitude = lat
        origin.latitude_errors.uncertainty = 0.01
        origin.latitude_errors.confidence_level = 95.0
        origin.longitude = long
        origin.quality = OriginQuality()
        origin.quality.standard_error = 1.
        origin.depth = depth
        origin.depth_errors["uncertainty"] = 0.1
        origin.depth_type = 'from location'
        origin.origin_uncertainty = OriginUncertainty()
        origin.origin_uncertainty.max_horizontal_uncertainty = 2.
        origin.origin_uncertainty.min_horizontal_uncertainty = 2.
        origin.origin_uncertainty.azimuth_max_horizontal_uncertainty = 1
        origin.quality.used_phase_count = 1
        origin.quality.azimuthal_gap = 1
        origin.quality.maximum_distance = 2.
        origin.quality.minimum_distance = 8.

        return origin

    def test_event_location_creation(self):
        from isp.db.models import EventLocationModel

        entity = EventLocationModel.find_by_id('1')
        self.assertIsNone(entity)

        origin = self._create_origin(15., 7., 0.)
        event_location = EventLocationModel.create_from_origin(origin)
        event_location.save()  # saved to db.

        # simple query
        e = EventLocationModel.find_by(latitude=15., get_first=True)
        self.assertIsNotNone(e)
        e = EventLocationModel.pagination(per_page=1, page=1)
        self.assertIsNotNone(e)

        # query with filters. Filters by default will always use the AND logic. Read doc to use OR logic.
        # Filters always use the Column of a model + operator + value
        filters = [EventLocationModel.latitude > 15.]  # select all EventLocation with latitude bigger than 15.
        entities = EventLocationModel.find_by_filter(filters=filters)
        self.assertEqual(len(entities), 0)

        # select all EventLocation with latitude bigger or equal than 15.
        filters = [EventLocationModel.latitude >= 15.]
        entities = EventLocationModel.find_by_filter(filters=filters)
        self.assertEqual(len(entities), 1)

        # select all EventLocation with latitude bigger or equal than 15 AND longitude smaller than 10.
        filters = [EventLocationModel.latitude >= 15., EventLocationModel.longitude < 10]
        entities = EventLocationModel.find_by_filter(filters=filters)
        self.assertEqual(len(entities), 1)


if __name__ == '__main__':
    unittest.main()
