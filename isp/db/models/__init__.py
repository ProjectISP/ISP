
# include models class name here.


class RelationShip:
    """
    Keep track of models class name for being used in relational tables.
    """
    FIRST_POLARITY = "FirstPolarityModel"
    MOMENT_TENSOR = "MomentTensorModel"
    EVENT_ARRAY = "EventArrayModel"
    PHASE_INFO = "PhaseInfoModel"

# Import models. Watch for circular dependencies.


try:
    from isp.db.models.models import FirstPolarityModel, MomentTensorModel, PhaseInfoModel, EventArrayModel,\
        ArrayAnalysisModel, EventLocationModel

except TypeError as e:
    msg = "Before import models start the database with db.start().\n {}".format(e)
    raise TypeError(msg)


def create_tables(force=False):
    """
     It will create tables if sqlite file don't exist. Set force to True if you want to run create anyway.

    :param force:If force is True it runs create tables anyway.  Default = False.

    :return:
    """
    from isp.db import db

    if not db.has_sqlite_db_file() or force:
        print("Creating all tables")
        db.create_all()


# will create tables if sqlite file don't exist. If force is True is created anyway.
create_tables()
