try:
    from isp.db.models.models import EventLocationModel, Dog

except TypeError as e:
    msg = "Before import models start the database with db.start().\n {}".format(e)
    raise TypeError(msg)


def create_tables():
    from isp.db import db

    db.create_all()
