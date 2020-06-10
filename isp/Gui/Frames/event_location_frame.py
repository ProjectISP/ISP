import math
import enum
import os

from isp.Gui import pw, pyc, qt
from isp.Gui.Frames import BaseFrame
from isp.Gui.Frames.uis_frames import UiEventLocationFrame
from isp.Gui.Models.sql_alchemy_model import SQLAlchemyModel

from isp.db.models import EventLocationModel

from sqlalchemy.sql.sqltypes import DateTime
from datetime import datetime, timedelta

from isp.Utils import ObspyUtil
from obspy.core.event import Origin

class EventColumn(enum.Enum):

    TIME = 0
    TRANSFORMATION = 1
    RMS = 2
    LATITUDE = 3
    LONGITUDE = 4
    DEPTH = 5
    UNCERTAINTY = 5

    def __str__(self):
        return str(self.value)

class MinMaxValidator(pyc.QObject):

    validChanged = pyc.pyqtSignal(bool)

    def __init__(self, min, max, signal='valueChanged', value='value', parent=None):
        super().__init__(parent)

        if not isinstance(min, pw.QWidget) or not isinstance(max, pw.QWidget):
            raise AttributeError("min and max are not QWidget or derived objects")

        if not hasattr(min, signal) or not hasattr(max, signal):
            raise AttributeError(f'min and max have no {signal} signal')

        def has_method(c, m): 
            return hasattr(c,m) and callable(getattr(c,m))

        if not has_method(min, value) or not has_method(max, value):
            raise AttributeError(f'min and max have no {value} method')

        self._min = min
        self._max = max
        self._min_tooltip = min.toolTip()
        self._max_tooltip = max.toolTip()
        self._min_style = min.styleSheet()
        self._max_style = max.styleSheet()
        self._valid = None
        self._value = value

        self._validate()

        getattr(self._min, signal).connect(self._validate)
        getattr(self._max, signal).connect(self._validate)

    @property
    def valid(self):
        return self._valid

    def _get_value(self, object):
        return getattr(object, self._value)()

    def _validate(self):
        if self.valid in (None, True) and self._get_value(self._min) > self._get_value(self._max) :
            self._min_tooltip = self._min.toolTip()
            self._max_tooltip = self._max.toolTip()
            self._min_style = self._min.styleSheet()
            self._max_style = self._max.styleSheet()
            self._min.setStyleSheet('background-color: red')
            self._max.setStyleSheet('background-color: red')
            self._min.setToolTip('Minimum and maximum are reversed')
            self._max.setToolTip('Minimum and maximum are reversed')
            self._valid = False
            self.validChanged.emit(False)

        elif self.valid in (None, False) and self._get_value(self._min) <= self._get_value(self._max) :
            self._min.setStyleSheet(self._min_style)
            self._max.setStyleSheet(self._max_style)
            self._min.setToolTip(self._min_tooltip)
            self._max.setToolTip(self._max_tooltip)
            self._valid = True
            self.validChanged.emit(True)

class EventLocationFrame(BaseFrame, UiEventLocationFrame):
    def __init__(self):
        super(EventLocationFrame, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Events Location')

        columns = ['origin_time', 'transformation', 'rms', 'latitude', 'longitude', 'depth', 'uncertainty',
                   'min_horizontal_error',"ellipse_azimuth"]
        col_names = ['Origin Time', 'Transformation', 'RMS', 'Latitude', 'Longitude', 'Depth', 'Uncertainty',
                     'smin',"Ellipse azimuth"]
        model = SQLAlchemyModel(EventLocationModel, columns, col_names, self)
        self.tableView.setModel(model)
        self.tableView.setSelectionBehavior(pw.QAbstractItemView.SelectRows)
        self.tableView.setContextMenuPolicy(qt.ActionsContextMenu)
        remove_action = pw.QAction("Remove selected location(s)", self)
        remove_action.triggered.connect(self._onRemoveRowsTriggered)
        self.tableView.addAction(remove_action)

        valLat = MinMaxValidator(self.minLat, self.maxLat)
        valLon = MinMaxValidator(self.minLon, self.maxLon)
        valDepth = MinMaxValidator(self.minDepth, self.maxDepth)
        valMag = MinMaxValidator(self.minMag, self.maxMag)
        valOrig = MinMaxValidator(self.minOrig, self.maxOrig, 'dateTimeChanged', 'dateTime')
        self._validators = [valLat, valLon, valDepth, valMag, valOrig]
        for validator in self._validators:
            validator.validChanged.connect(self._checkQueryParameters)
        
        self.actionRead_hyp_folder.triggered.connect(self._readHypFolder)
        self.btnRefreshQuery.clicked.connect(self._refreshQuery)
        self.btnShowAll.clicked.connect(self._showAll)
        self.PlotMapBtn.clicked.connect(self.plot_map)

    def refreshLimits(self):
        events = self.tableView.model().getRows()

        if events :
            max_lat = -math.inf
            min_lat = math.inf
            max_lon = -math.inf
            min_lon = math.inf
            max_dep = -math.inf
            min_dep = math.inf
            min_orig = datetime.max
            max_orig = datetime.min
            
            for event in events:
                if event.latitude > max_lat: max_lat = event.latitude
                if event.latitude < min_lat: min_lat = event.latitude
                if event.longitude > max_lon: max_lon = event.longitude
                if event.longitude < min_lon: min_lon = event.longitude
                if event.depth > max_dep: max_dep = event.depth
                if event.depth < min_dep: min_dep = event.depth
                if event.origin_time < min_orig : min_orig = event.origin_time
                if event.origin_time > max_orig : max_orig = event.origin_time

            self.maxLat.setValue(math.ceil(max_lat))
            self.minLat.setValue(math.floor(min_lat))
            self.maxLon.setValue(math.ceil(max_lon))
            self.minLon.setValue(math.floor(min_lon))
            self.maxDepth.setValue(math.ceil(max_dep))
            # TODO: depth can be negative?  fix ui limit
            self.minDepth.setValue(math.floor(min_dep))
            # TODO magnitude
            #self.maxMag.setValue(max_mag)
            #self.minMag.setValue(min_mag)
            self.maxOrig.setDateTime(max_orig + timedelta(seconds=1))
            self.minOrig.setDateTime(min_orig - timedelta(seconds=1))

    def _onRemoveRowsTriggered(self):
        selected_rowindexes = self.tableView.selectionModel().selectedRows()
        selected_rows = [r.row() for r in selected_rowindexes]
        for r in sorted(selected_rows, reverse=True):
            print(self.tableView.model().removeRow(r))

        self.tableView.model().submitAll()

    def _checkQueryParameters(self):
        self.btnRefreshQuery.setEnabled(all(v.valid for v in self._validators))

    def _readHypFolder(self):
        dir = pw.QFileDialog.getExistingDirectory(self, "Get directory to read .hyp files from")
        files = [f for f in os.listdir(dir) if f.endswith('.hyp')]
        for file in files:
            file_abs = os.path.abspath(os.path.join(dir, file))
            try:
                origin : Origin = ObspyUtil.reads_hyp_to_origin(file_abs)
                try:
                    event_model = EventLocationModel.create_from_origin(origin)
                    event_model.save()
                except AttributeError:
                    # TODO: what to do if it is already inserted?
                    pass
            except :
                print(f'File {file} could not be processed correctly')

        # TODO: show all after reading folder or let filters?
        self._showAll()
        self.refreshLimits()

    def _refreshQuery(self):
        lat = EventLocationModel.latitude.between(self.minLat.value(), self.maxLat.value())
        lon = EventLocationModel.longitude.between(self.minLon.value(), self.maxLon.value())
        depth = EventLocationModel.depth.between(self.minDepth.value(), self.maxDepth.value())
        # TODO: mag filter
        minOrig = self.minOrig.dateTime().toPyDateTime()
        maxOrig = self.maxOrig.dateTime().toPyDateTime()
        time = EventLocationModel.origin_time.between(minOrig, maxOrig)
        self.tableView.model().setFilter(lat, lon, depth, time)
        self.tableView.model().revertAll()

    def _showAll(self):
        self.tableView.model().setFilter()
        self.tableView.model().revertAll()

    def plot_map(self):
        events = self.tableView.model().getRows()
        lat = []
        lon = []
        for j in events:
            lat.append(j.latitude)
            lon.append(j.longitude)