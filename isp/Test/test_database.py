import os
import unittest

from isp.db import db

dir_path = os.path.dirname(os.path.abspath(__file__))
db.url = "sqlite:///{}/isp_test.db".format(dir_path)
db.start()

from isp.db.models import EventLocationModel


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("setup")
        db.create_all()

    def test_event_location(self):
        print("run test")
        print(EventLocationModel.get_all())


if __name__ == '__main__':
    unittest.main()
