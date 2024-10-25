from database.database import database as db
from mysql.connector import Error


class dbquery:
    def __init__(self):
        self.conn = None
        self.db = None

    def init(self, **kwargs):
        self.db = db(**kwargs)
        #self.db.init(**kwargs)
        self.conn = self.db.connect()

    ###def connect(self):
       ### self.conn = _db.connect()

    #def close(self):
     #   self.conn.close()

    def read(self, query):
        cursor = self.conn.cursor()

        try:
            cursor.execute(query)
            result = cursor.fetchall()

            return result
        except Error as err:
            print(f"Error: '{err}'")

    def getEvents(self, start, end):
        self.conn = self.db.connect()

        query = "SELECT _event.publicID, Origin.time_value, Event.* from Origin, PublicObject as _origin, " \
                "Event, PublicObject as _event "
        query += "WHERE _origin.publicID=Event.preferredOriginID AND Origin._oid=_origin._oid "
        query += "and Event._oid=_event._oid and Origin.time_value >= '"
        query += start
        query += "' and Origin.time_value <= '"
        query += end
        query += "'"

        result = self.read(query)
        self.conn.close()

        return result

    def getPicks(self, id):
        self.conn = self.db.connect()

        query = "SELECT DISTINCT(_pick.publicID), Pick.* FROM Event, PublicObject as _event, OriginReference, Origin, "
        query += "PublicObject as _origin, Arrival, Pick, PublicObject as _pick "
        query += "WHERE OriginReference.originID = _origin.publicID and Arrival.pickID = _pick.publicID "
        query += "and OriginReference._parent_oid = Event._oid and Arrival._parent_oid = Origin._oid "
        query += "and Event._oid = _event._oid and Origin._oid = _origin._oid and Pick._oid = _pick._oid "
        query += "and _event.publicID = '"
        query += id
        query += "'"

        result = self.read(query)
        self.conn.close()

        return result

    def getMagnitude(self, id):
        self.conn = self.db.connect()

        query = "SELECT _magnitude.publicID, Magnitude.* FROM Magnitude, PublicObject as _magnitude, Event, Origin, "
        query += "PublicObject as _origin WHERE _magnitude.publicID = Event.preferredMagnitudeID and "
        query += "_origin.publicID = Event.preferredOriginID and Magnitude._parent_oid = Origin._oid and "
        query += "Magnitude._oid = _magnitude._oid and Origin._oid = _origin._oid and "
        query += "_magnitude.publicID = '"
        query += id
        query += "'"

        result = self.read(query)

        self.conn.close()

        return result

    def getOrigin(self, id):
        self.conn = self.db.connect()

        query = "SELECT _origin.publicID, Origin.* FROM Origin, PublicObject as _origin, Event WHERE "
        query += "_origin.publicID = Event.preferredOriginID and Origin._oid = _origin._oid and "
        query += "_origin.publicID = '"
        query += id
        query += "'"

        result = self.read(query)

        self.conn.close()

        return result
