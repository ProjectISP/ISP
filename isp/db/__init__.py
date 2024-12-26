import random
import string

from obspy.core.event import Origin

from isp.db.data_base import DataBase

db: DataBase = DataBase()

"After importing db you must call db.start() and db.create_all() to initialize and create all tables into the database."

">>> db.start() "

"Before call db.create() you should import all your models."
">>> db.create_all()"


def generate_id(length: int) -> str:
    """
    Generate a random string with the combination of lowercase and uppercase letters.

    :param length: The size of the id key

    :return: An id of size length formed by lowe and uppercase letters.
    """
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(length))


def generate_id_from_origin(origin: Origin):

    """
    Generate an ID from Origin object

    :param origin: obspy object for origin time of earthquake

    :return: id based on origin time
    """
    return origin.time.strftime("%Y%m%d%H%M%S")
