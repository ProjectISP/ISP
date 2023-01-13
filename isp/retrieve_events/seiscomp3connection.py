import datetime as date
import numpy as np
from isp.retrieve_events.query import dbquery
import pysftp as sftp
import time


class seiscompConnector:
    def __init__(self, **kwargs):

        self._hostname = None
        self._dbname = None
        self._user = None
        self._password = None

        self._sdshost = None
        self._sdsuser = None
        self._sdspass = None
        self._sdsdir = None
        self._sdsout = None
        self._port = None
        self._filename = None

        self._startTime = None
        self._endTime = None
        self._margin = None
        self.network = None
        self.station = None
        self.channel = None

        self._ids = []
        self._streams = []
        self.db = None
        self.pick = None
        self.event = None

        self.pick = Pick()
        self.event = Event()
        self.archive = Archive()

        if 'hostname' not in kwargs.keys():
            raise ValueError('Database hostname is mandatory')
        else:
            self._hostname = kwargs['hostname']

        if 'dbname' not in kwargs.keys():
            raise ValueError('Database name is mandatory')
        else:
            self._dbname = kwargs['dbname']

        if 'user' in kwargs.keys():
            self._user = kwargs['user']

        if 'password' in kwargs.keys():
            self._password = kwargs['password']

        if 'file' in kwargs.keys():
            self._filename = kwargs['file']

        if 'end' in kwargs.keys():
            self._endTime = date.datetime.strptime(kwargs['end'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        else:
            self._endTime = date.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        if 'start' in kwargs.keys():
            self._startTime = kwargs['start']
        else:
            _end = date.datetime.strptime(self._endTime, "%Y-%m-%d %H:%M:%S")
            self._startTime = (_end - date.timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")

        if 'margin' in kwargs.keys():
            if isinstance(kwargs['margin'], (list, tuple, np.ndarray)):
                self._margin = kwargs['margin']
            else:
                raise TypeError('margin have to be an array')
        else:
            self._margin = [300, 300]

        if 'network' in kwargs.keys():
            self.network = kwargs['network']

        if 'station' in kwargs.keys():
            self.station = kwargs['station']

        if 'channel' in kwargs.keys():
            if isinstance(kwargs['channel'], str):
                self.channel = kwargs['channel']
            else:
                raise TypeError('channel have to be a string')

        if 'sdshost' in kwargs.keys():
            self._sdshost = kwargs['sdshost']

        if 'sdsuser' in kwargs.keys():
            self._sdsuser = kwargs['sdsuser']

        if 'sdspass' in kwargs.keys():
            self._sdspass = kwargs['sdspass']

        if 'sdsdir' in kwargs.keys():
            self._sdsdir = kwargs['sdsdir']

        if 'sdsout' in kwargs.keys():
            self._sdsout = kwargs['sdsout']

        if 'sdsport' in kwargs.keys():
            self._port = int(kwargs['sdsport'])

        # Database init
        self.db = dbquery(host=self._hostname, name=self._dbname, user=self._user, password=self._password)
        # self.db(host=self._hostname, name=self._dbname, user=self._user, password=self._password)

        self.archive.init(host=self._sdshost, user=self._sdsuser, password=self._sdspass, sdsdir=self._sdsdir,
                          sdsout=self._sdsout, port=self._port)

    def find(self, start, end, **kwargs):
        start, end = self.parseTime(start, end)

        # Get Events
        result = self.db.getEvents(start, end)

        if result is None:
            raise ValueError('No events available')
        else:
            result = self.event.event(result)

        # Get Magnitude
        self.getMagnitude(self.event.events)

        # Get Origin and Depth
        self.getOriginDepth(self.event.events)

        # Get Picks
        self.pick.pick(self.getpicks(result))
        print(self.pick.picks)

        # Check filter
        if 'network' in kwargs.keys():
            result = self.pickFilter('network', kwargs['network'])

        if 'station' in kwargs.keys():
            result = self.pickFilter('station', kwargs['station'])

        if 'channel' in kwargs.keys():
            result = self.pickFilter('channel', kwargs['channel'])

        if 'magnitude' in kwargs.keys():
            result = self.eventFilter('magnitude', kwargs['magnitude'])

        if 'latitude' in kwargs.keys():
            result = self.eventFilter('longitude', kwargs['latitude'])

        if 'longitude' in kwargs.keys():
            result = self.eventFilter('longitude', kwargs['longitude'])

        if 'depth' in kwargs.keys():
            result = self.eventFilter('depth', kwargs['depth'])

        # Minimun and maximum pick time for every event
        self.margin(self.event.events)

        # Save streams
        if len(self.pick.picksFiltered) > 0:
            self.event.setStreams(self.pick.picksFiltered)
        else:
            self.event.setStreams(self.pick.picks)

        if len(self.event.eventsFiltered) == 0:
            return self.event.events
        else:
            return self.event.eventsFiltered

    def delete_filter(self, data, trash):
        for i in trash:
            data.remove(i)

        return data

    def filter_by_name(self, data, name, filter):
        streams_array = []
        data_array = []
        for d in data:
            if name in ['channel', 'station', 'network']:
                for i in range(len(d['streams'])):
                    if filter not in d['streams'][i].split('.'):
                        streams_array.append(d['streams'][i])

                if len(streams_array) > 0:
                    d['streams'] = self.delete_filter(d['streams'], streams_array)

                streams_array = []

                if len(d['streams']) == 0:
                    data_array.append(d)

            else:
                if d[name] < filter[0] or d[name] > filter[1]:
                    data_array.append(d)

        data = self.delete_filter(data, data_array)

        return data

    def filter_smart(self, data, **kwargs):

        network = kwargs.pop('network', [])
        station = kwargs.pop('station', [])
        channel = kwargs.pop('channel', [])

        if len(network) > 0:
            data = self.filter_by_name(data, 'network', network)

        if len(station) > 0:
            data = self.filter_by_name(data, 'station', station)

        if len(station) > 0:
            data = self.filter_by_name(data, 'channel', channel)

        return data

    def refilt(self, result, coords):

        # example
        # coords = [['2022-10-01 01:45:07', 39.696, -1.194, 10], ['2022-10-01 07:04:13', 39.64, -1.143, 12]]
        result_new = []
        for item in result:
            item_test = [item['time'], item['latitude'], item['longitude'], item['depth']]
            for coord_test in coords:
                if coord_test == item_test:
                    result_new.append(item)
            else:
                pass
        return result_new


    @staticmethod
    def parseTime(start, end):
        if end is None:
            end = date.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        else:
            end = end.strftime("%Y-%m-%d %H:%M:%S")

        if start is None:
            start = (date.datetime.strptime(end, "%Y-%m-%d %H:%M:%S") - date.timedelta(days=2)).strftime("%Y-%m-%d "
                                                                                                         "%H:%M:%S")
        else:
            start = start.strftime("%Y-%m-%d %H:%M:%S")

        return start, end

    def checkfile(self):
        return self._filename is not None

    def download(self, events):
        self.getStreams(self.event.events)
        self.archive.conn.close()

    """
        # Get mseed

        print(result)

    """

    """
    def getidfromfile(self):
        with open(self._name, newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                if (row[2] + ' ' + row[3] >= self._startTime and row[2] + ' ' + row[3] <= self._endTime):
                    self._ids.append((row[0]))
                print(row)
    """

    def eventFilter(self, name, item):
        if len(self.event.eventsFiltered) > 0:
            self.event.eventsFiltered = [e for e in self.event.eventsFiltered if (item[0] <= e[name] <= item[1])]

        else:
            self.event.eventsFiltered = [e for e in self.event.events if (item[0] <= e[name] <= item[1])]

        return self.event.eventsFiltered

    def pickFilter(self, name, item):
        if len(self.pick.picksFiltered) > 0:
            self.pick.picksFiltered = [p for p in self.pick.picksFiltered if p[name] == item]
        else:
            self.pick.picksFiltered = [p for p in self.pick.picks if p[name] == item]

        return self.pick.picksFiltered

    def getMagnitude(self, events):
        if events:
            for e in events:
                m = self.db.getMagnitude(e['magnitudeID'])
                e['magnitude'] = round(m[0][4], 1)

    def getOriginDepth(self, events):
        if events:
            for e in events:
                o = self.db.getOrigin(e['originID'])
                e['latitude'] = round(o[0][10], 3)
                e['longitude'] = round(o[0][15], 3)
                e['depth'] = round(o[0][20])

    def getidfromdb(self):
        return self.db.getEvents(self._startTime, self._endTime)

    def getpicks(self, events):
        result = []

        if events:
            for e in events:
                m = self.db.getPicks(e['id'])

                if m is not None:
                    result.append([e['id'], e['time'], m])

        return result

    def margin(self, events):
        minTime = 0
        maxTime = 0

        margin = []

        for e in events:
            if len(self.pick.picksFiltered) > 0:
                minTime = min(date.datetime.strptime(p['time'], "%Y-%m-%d %H:%M:%S") for p in self.pick.picksFiltered if
                              p['eventID'] == e['id'])

                maxTime = max(date.datetime.strptime(p['time'], "%Y-%m-%d %H:%M:%S") for p in self.pick.picksFiltered if
                              p['eventID'] == e['id'])
            else:
                minTime = min(date.datetime.strptime(p['time'], "%Y-%m-%d %H:%M:%S") for p in self.pick.picks if
                              p['eventID'] == e['id'])

                maxTime = max(date.datetime.strptime(p['time'], "%Y-%m-%d %H:%M:%S") for p in self.pick.picks if
                              p['eventID'] == e['id'])

            minTime = minTime - date.timedelta(seconds=self._margin[0])
            maxTime = maxTime + date.timedelta(seconds=self._margin[1])

            margin.append({
                'id': e['id'],
                'min': date.datetime.strftime(minTime, "%Y-%m-%d %H:%M:%S"),
                'max': date.datetime.strftime(maxTime, "%Y-%m-%d %H:%M:%S")
            })

        self.event.setMargin(margin)

    def getStreams(self, events):
        filepool = []
        for e in events:
            print("a")
            for s in e['streams']:
                filepool.append(self.archive.formatfile(e['min'], s))

        for f in filepool:
            x = f.split('/')
            print(x)
            print(self.archive.sdsdir + f)
            self.archive.conn.get(self.archive.sdsdir + f, self.archive.outdir + x[4])
        print(filepool)


class Pick:
    def __init__(self):
        self.picks = []
        self.picksFiltered = []

    def pick(self, pick):
        for p in pick:
            for a in p[2]:
                self.picks.append({
                    'id': a[0],
                    'eventID': p[0],
                    'eventTime': p[1],
                    'time': a[4].strftime("%Y-%m-%d %H:%M:%S"),
                    'network': a[10],
                    'station': a[11],
                    'location': a[12],
                    'channel': a[13],
                    'filter': a[15],
                    'method': a[16],
                    'slownessMethod': a[29]
                })

    def getAttribute(self, id, item):
        if len(self.picksFiltered) == 0:
            result = [p for p in self.picks if p['id'] == id]
        else:
            result = [p for p in self.picksFiltered if p['id'] == id]

        if len(result) == 1:
            result = result[0]

            if item == 'all':
                return result
            else:
                return result[item]

        elif len(result) == 0:
            return None

    def networkFilter(self, network):
        if len(self.picksFiltered) > 0:
            self.picksFiltered = [p for p in self.picksFiltered if p['network'] == network]
        else:
            self.picksFiltered = [p for p in self.picks if p['network'] == network]

    def stationFilter(self, station):
        if len(self.picksFiltered) > 0:
            self.picksFiltered = [p for p in self.picksFiltered if p['station'] == station]
        else:
            self.picksFiltered = [p for p in self.picks if p['station'] == station]

    def channelFilter(self, channel):
        if len(self.picksFiltered) > 0:
            self.picksFiltered = [p for p in self.picksFiltered if p['channel'] == channel]
        else:
            self.picksFiltered = [p for p in self.picks if p['channel'] == channel]


class Event:
    def __init__(self):
        self.events = []
        self.eventsFiltered = []

    def event(self, event):
        for e in event:
            self.events.append({
                'id': e[0],
                'time': e[1].strftime("%Y-%m-%d %H:%M:%S"),
                'originID': e[5],
                'magnitudeID': e[6]})

        return self.events

    def setMargin(self, margin):
        for m in margin:
            a = [i for i, e in enumerate(self.events) if e['id'] == m['id']][0]
            self.events[a]['min'] = m['min']
            self.events[a]['max'] = m['max']

    def setStreams(self, picks):
        _streams = []

        for e in self.events:
            for p in picks:
                if p['eventID'] == e['id']:
                    _streams.append(p['network'] + '.' + p['station'] + '.' + p['location'] + '.' + p['channel'])

            e['streams'] = _streams

            _streams = []


class Archive:
    def __init__(self):
        self.host = None
        self.user = None
        self.password = None
        self.sdsdir = None
        self.outdir = None
        self.conn = None
        self.cnopts = sftp.CnOpts()
        self.cnopts.hostkeys = None
        self.port = None

    def init(self, **kwargs):
        if 'host' in kwargs.keys():
            self.host = kwargs['host']

        if 'user' in kwargs.keys():
            self.user = kwargs['user']

        if 'password' in kwargs.keys():
            self.password = kwargs['password']

        if 'sdsdir' in kwargs.keys():
            self.sdsdir = kwargs['sdsdir']

        if 'sdsout' in kwargs.keys():
            self.outdir = kwargs['sdsout']

        if 'port' in kwargs.keys():
            self.port = kwargs['port']

        self.conn = self.connect()

    def connect(self):
        connection = None
        if self.host is not None:
            try:
                connection = sftp.Connection(host=self.host, username=self.user, password=self.password,
                                             cnopts=self.cnopts, port=self.port)
            except Exception as err:
                raise Exception(err)

            finally:

                print(f"Connected to {self.host} as {self.user}.")

        return connection

    def formatfile(self, start, stream):
        t = time.gmtime(int(date.datetime.strptime(start, "%Y-%m-%d %H:%M:%S").timestamp()))
        _s = stream.split('.')

        filedir = str(t[0]) + "/" + _s[0] + "/" + _s[1] + "/" + _s[3] + ".D/" + stream + ".D." + str(t[0]) + \
                  ".%03d" % t[7]

        return filedir
