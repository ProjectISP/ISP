import math
import os
from isp.Gui import pw, pyc, qt, pqg
from isp.Gui.Frames import BaseFrame, MessageDialog
from isp.Gui.Frames.uis_frames import UiEventLocationFrame
from isp.Gui.Models.sql_alchemy_model import SQLAlchemyModel
from isp.db.models import EventLocationModel, FirstPolarityModel, MomentTensorModel
from isp.db import generate_id
# from sqlalchemy.sql.sqltypes import DateTime
from datetime import datetime, timedelta
from isp.Utils import ObspyUtil
from obspy.core.event import Origin
# from sqlalchemy import Column
from obspy.imaging.beachball import beach
from isp import LOCATION_OUTPUT_PATH, MOMENT_TENSOR_OUTPUT, ROOT_DIR
from isp.mti.read_log import read_log
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from owslib.wms import WebMapService
import os


class DateTimeFormatDelegate(pw.QStyledItemDelegate):
    def __init__(self, date_format, parent=None):
        super().__init__(parent)
        self.date_format = date_format

    def displayText(self, value, locale):
        return value.toString(self.date_format)


class MinMaxValidator(pyc.QObject):
    validChanged = pyc.pyqtSignal(bool)

    def __init__(self, min, max, signal='valueChanged', value='value', parent=None):
        super().__init__(parent)

        if not isinstance(min, pw.QWidget) or not isinstance(max, pw.QWidget):
            raise AttributeError("min and max are not QWidget or derived objects")

        if not hasattr(min, signal) or not hasattr(max, signal):
            raise AttributeError(f'min and max have no {signal} signal')

        def has_method(c, m):
            return hasattr(c, m) and callable(getattr(c, m))

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
        if self.valid in (None, True) and self._get_value(self._min) > self._get_value(self._max):
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

        elif self.valid in (None, False) and self._get_value(self._min) <= self._get_value(self._max):
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
        self.setWindowIcon(pqg.QIcon(':\icons\compass-icon.png'))
        self.cb = None
        el_columns = [getattr(EventLocationModel, c)
                      for c in EventLocationModel.__table__.columns.keys()[1:]]

        fp_columns = [getattr(FirstPolarityModel, c)
                      for c in FirstPolarityModel.__table__.columns.keys()[2:]]

        mti_columns = [getattr(MomentTensorModel, c)
                       for c in MomentTensorModel.__table__.columns.keys()[2:]]

        columns = [*el_columns, *fp_columns, *mti_columns]

        col_names = ['Origin Time', 'Transformation', 'RMS',
                     'Latitude', 'Longitude', 'Depth', 'Uncertainty',
                     'Max. Hor. Error', 'Min. Hor. Error', 'Ellipse Az.',
                     'No. Phases', 'Az. Gap', 'Max. Dist.', 'Min. Dist.',
                     'Mb', 'Mb Error', 'Ms', 'Ms Error', 'Ml', 'Ml Error',
                     'Mw', 'Mw Error', 'Mc', 'Mc Error', 'Strike', 'Dip',
                     'Rake', 'Misfit', 'Az. Gap', 'Stat. Pol. Count', 'Latitude_mti', 'Longitude_mti', 'Depth_mti',
                     'VR', 'CN', 'dc', 'clvd', 'iso', 'Mw_mt', 'Mo', 'Strike_mt', 'dip_mt', 'rake_mt', 'mrr', 'mtt',
                     'mpp',
                     'mrt', 'mrp', 'mtp']

        entities = [EventLocationModel, FirstPolarityModel, MomentTensorModel]
        self.model = SQLAlchemyModel(entities, columns, col_names, self)
        self.model.addJoinArguments(EventLocationModel.first_polarity, isouter=True)
        self.model.addJoinArguments(EventLocationModel.moment_tensor, isouter=True)
        self.model.revertAll()
        sortmodel = pyc.QSortFilterProxyModel()
        sortmodel.setSourceModel(self.model)
        self.tableView.setModel(sortmodel)
        self.tableView.setSortingEnabled(True)
        self.tableView.sortByColumn(0, qt.AscendingOrder)
        self.tableView.setSelectionBehavior(pw.QAbstractItemView.SelectRows)
        self.tableView.setContextMenuPolicy(qt.ActionsContextMenu)
        remove_action = pw.QAction("Remove selected location(s)", self)
        remove_action.triggered.connect(self._onRemoveRowsTriggered)
        self.tableView.addAction(remove_action)
        self.tableView.setItemDelegateForColumn(0, DateTimeFormatDelegate('dd/MM/yyyy hh:mm:ss.zzz'))
        self.tableView.resizeColumnsToContents()
        self.tableView.doubleClicked.connect(self._plot_foc_mec)

        valLat = MinMaxValidator(self.minLat, self.maxLat)
        valLon = MinMaxValidator(self.minLon, self.maxLon)
        valDepth = MinMaxValidator(self.minDepth, self.maxDepth)
        valMag = MinMaxValidator(self.minMag, self.maxMag)
        valOrig = MinMaxValidator(self.minOrig, self.maxOrig, 'dateTimeChanged', 'dateTime')
        self._validators = [valLat, valLon, valDepth, valMag, valOrig]
        for validator in self._validators:
            validator.validChanged.connect(self._checkQueryParameters)
        self.minOrig.setDisplayFormat('dd/MM/yyyy hh:mm:ss.zzz')
        self.maxOrig.setDisplayFormat('dd/MM/yyyy hh:mm:ss.zzz')

        self.actionRead_hyp_folder.triggered.connect(self._readHypFolder)
        self.actionRead_last_location.triggered.connect(self._readLastLocation)
        self.btnRefreshQuery.clicked.connect(self._refreshQuery)
        self.btnShowAll.clicked.connect(self._showAll)
        self.PlotMapBtn.clicked.connect(self.__plot_map)

    # def _plot_row(self, index):
    #     # try:
    #     #     self.text.pop(0).remove()
    #     # except:
    #     #     pass
    #     lat_index = self.model.index(index.row(), 3)
    #     lon_index = self.model.index(index.row(), 4)
    #     depth_index = self.model.index(index.row(), 5)
    #     lat = self.model.data(lat_index)
    #     lon = self.model.data(lon_index)
    #     depth = self.model.data(depth_index)
    #     print(lat,lon,depth)
    #     self.text=self.map_widget.ax.annotate('Hola', xy=(-12, 35), xycoords='data',
    #                                 xytext=(-12, 36), textcoords='data', arrowprops=dict(arrowstyle="->",
    #                                                                                      connectionstyle="arc3,rad=.2"))
    #
    #     self.map_widget.fig.canvas.draw()

    def _plot_foc_mec(self, index):

        # Plot Focal Mechanism
        model = self.tableView.model()
        check = False
        if self.methodCB.currentText() == "First Polarity":

            strike = model.data(model.index(index.row(), 24))
            dip = model.data(model.index(index.row(), 25))
            rake = model.data(model.index(index.row(), 26))
            if None not in [strike, dip, rake]:
                self.plot_foc_mec(method=self.methodCB.currentText(), strike=strike, dip=dip, rake=rake)
                check = True

        if self.methodCB.currentText() == "MTI":
            mrr = model.data(model.index(index.row(), 43))
            mtt = model.data(model.index(index.row(), 44))
            mpp = model.data(model.index(index.row(), 45))
            mrt = model.data(model.index(index.row(), 46))
            mrp = model.data(model.index(index.row(), 47))
            mtp = model.data(model.index(index.row(), 48))

            if None not in [mrr, mtt, mpp, mrt, mrp, mtp]:
                self.plot_foc_mec(method=self.methodCB.currentText(), mrr=mrr, mtt=mtt, mpp=mpp, mrt=mrt, mrp=mrp,
                                  mtp=mtp)
                check = True

        if check:
            # plot in the map
            lat = model.data(model.index(index.row(), 3))
            lon = model.data(model.index(index.row(), 4))
            print(lat, lon)
            file = os.path.join(ROOT_DIR, 'db/map_class/foc_mec.png')

            img = Image.open(file)
            imagebox = OffsetImage(img, zoom=0.08)
            imagebox.image.axes = self.map_widget.ax
            ab = AnnotationBbox(imagebox, [lon + 0.3, lat + 0.3], frameon=False)
            self.map_widget.ax.add_artist(ab)

            self.map_widget.ax.annotate('', xy=(lon, lat), xycoords='data',
                                        xytext=(lon + 0.3, lat + 0.3), textcoords='data',
                                        arrowprops=dict(arrowstyle="->",
                                                        connectionstyle="arc3,rad=.2"))

            self.map_widget.fig.canvas.draw()

    def refreshLimits(self):
        entities = self.model.getEntities()
        events = []
        for t in entities:
            events.append(t[0])

        if events:
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
                if event.origin_time < min_orig: min_orig = event.origin_time
                if event.origin_time > max_orig: max_orig = event.origin_time

            self.maxLat.setValue(math.ceil(max_lat))
            self.minLat.setValue(math.floor(min_lat))
            self.maxLon.setValue(math.ceil(max_lon))
            self.minLon.setValue(math.floor(min_lon))
            self.maxDepth.setValue(math.ceil(max_dep))
            self.minDepth.setValue(math.floor(min_dep))
            # TODO magnitude
            # self.maxMag.setValue(max_mag)
            # self.minMag.setValue(min_mag)
            self.maxOrig.setDateTime(max_orig + timedelta(seconds=1))
            self.minOrig.setDateTime(min_orig - timedelta(seconds=1))

    def _onRemoveRowsTriggered(self):
        selected_rowindexes = self.tableView.selectionModel().selectedRows()
        # If table's model is a proxy model, map to source indexes
        if isinstance(self.tableView.model(), pyc.QAbstractProxyModel):
            selected_rowindexes = [self.tableView.model().mapToSource(i)
                                   for i in selected_rowindexes]

        selected_rows = [r.row() for r in selected_rowindexes]
        for r in sorted(selected_rows, reverse=True):
            self.model.removeRow(r)

        self.model.submitAll()

    def _checkQueryParameters(self):
        self.btnRefreshQuery.setEnabled(all(v.valid for v in self._validators))

    def _readHypFile(self, file_abs_path):
        origin: Origin = ObspyUtil.reads_hyp_to_origin(file_abs_path)
        try:
            event_model = EventLocationModel.create_from_origin(origin)
            event_model.save()
        except AttributeError:
            # TODO: what to do if it is already inserted?
            event_model = EventLocationModel.find_by(latitude=origin.latitude, longitude=origin.longitude,
                                                     depth=origin.depth, origin_time=origin.time.datetime)

        return event_model

    def _readHypFolder(self):
        dir = pw.QFileDialog.getExistingDirectory(self, "Get directory to read .hyp files from")

        # If user cancels selecting folder, return
        if not dir:
            return

        files = [f for f in os.listdir(dir) if f.endswith('.hyp')]
        errors = []
        for file in files:
            file_abs = os.path.abspath(os.path.join(dir, file))
            try:
                self._readHypFile(file_abs)
            except Exception as e:
                errors.append(str(e))

        if errors:
            m = pw.QMessageBox(pw.QMessageBox.Warning, self.windowTitle(),
                               'Some errors ocurred while processing files. See detailed.', parent=self)
            m.setDetailedText('\n'.join(errors))
            m.exec()

        # TODO: show all after reading folder or let filters?
        self._showAll()
        self.refreshLimits()

    def _readLastLocation(self):
        # Insert event location or get if it already exists
        hyp_path = os.path.join(LOCATION_OUTPUT_PATH, 'last.hyp')
        try:
            event = self._readHypFile(hyp_path)
        except Exception as e:
            pw.QMessageBox.warning(self, self.windowTitle(), f'An error ocurred reading hyp file: {e}')
            return

        # Update magnitude data
        mag_file = os.path.join(LOCATION_OUTPUT_PATH, 'magnitudes_output.mag')
        if os.path.isfile(mag_file):
            with open(mag_file) as f:
                next(f)
                mag_dict = {}
                for line in f:
                    key, value = line.split()
                    key = key.lower().replace('std', 'error')
                    try:
                        value = float(value)
                        mag_dict[key] = value
                    except ValueError:
                        pass
                event.set_magnitudes(mag_dict)
                event.save()

        # Update first polarity data
        # TODO: this could be improved by updating instead of removing the
        # existing one
        fp = FirstPolarityModel.find_by(event_info_id=event.id)
        if fp:
            fp.delete()

        fp_file = os.path.join(LOCATION_OUTPUT_PATH, 'first_polarity.fp')
        if os.path.isfile(fp_file):
            with open(fp_file) as f:
                next(f)
                fp_dict = {}
                fp_fields = {'Strike': 'strike_fp', 'Dip': 'dip_fp',
                             'Rake': 'rake_fp', 'misfit_first_polarity': 'misfit_fp',
                             'azimuthal_gap': 'azimuthal_fp_Gap',
                             'number_of_polarities': 'station_fp_polarities_count'}
                for line in f:
                    key, value = line.split()
                    try:
                        key = fp_fields[key]
                        value = float(value)
                        fp_dict[key] = value
                    except (ValueError, KeyError):
                        pass
                fp_dict['event_info_id'] = event.id
                fp_dict['id'] = generate_id(16)
                fp = FirstPolarityModel.from_dict(fp_dict)
                fp.save()

        mti = MomentTensorModel.find_by(event_info_id=event.id)
        if mti:
            mti.delete()

        mti_file = os.path.join(MOMENT_TENSOR_OUTPUT, 'log.txt')
        if os.path.isfile(mti_file):
            mti_dict = read_log(mti_file)
            mti_dict['event_info_id'] = event.id
            mti_dict['id'] = generate_id(16)
            mti = MomentTensorModel.from_dict(mti_dict)
            mti.save()

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
        self.model.setFilter(lat, lon, depth, time)
        self.model.revertAll()

    def _showAll(self):
        self.model.setFilter()
        self.model.revertAll()

    def __plot_map(self):

        URL = self.wmsLE.text()
        layer = self.layerLE.text()
        if URL is not "" and layer is not "":
            self.plot_map(map_service=URL, layer=layer)
        else:
            self.plot_map()

    def plot_map(self, map_service='https://www.gebco.net/data_and_products/gebco_web_services/2020/mapserv?',
                 layer='GEBCO_2020_Grid'):

        try:
            if self.cb:
                self.cb.remove()

                # self.map_widget.fig.delaxes(self.map_widget.fig.axes[1])

            MAP_SERVICE_URL = map_service
            try:
                wms = WebMapService(MAP_SERVICE_URL)
            except:
                pass
            layer = layer

            entities = self.model.getEntities()
            lat = []
            lon = []
            depth = []
            mag = []
            for j in entities:
                lat.append(j[0].latitude)
                lon.append(j[0].longitude)
                depth.append(j[0].depth)

                if j[0].mw == None:
                    j[0].mw = 3.0

                mag.append(j[0].mw)

            # print(entities)

            mag = np.array(mag)
            mag = 0.5 * np.exp(mag)
            min_lon = min(lon) - 0.5
            max_lon = max(lon) + 0.5
            min_lat = min(lat) - 0.5
            max_lat = max(lat) + 0.5
            extent = [min_lon, max_lon, min_lat, max_lat]

            self.map_widget.ax.set_extent(extent, crs=ccrs.PlateCarree())

            try:
                self.map_widget.ax.add_wms(wms, layer)
            except:
                os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join(ROOT_DIR, "maps")
                self.map_widget.ax.background_img(name='ne_shaded', resolution="high")


            lon = np.array(lon)
            lat = np.array(lat)
            depth = np.array(depth) / 1000
            color_map = plt.cm.get_cmap('rainbow')
            reversed_color_map = color_map.reversed()
            cs = self.map_widget.ax.scatter(lon, lat, s=mag, c=depth, edgecolors="black", cmap=reversed_color_map,
                                            transform=ccrs.PlateCarree())
            self.cb = self.map_widget.fig.colorbar(cs, ax=self.map_widget.ax, orientation='horizontal', fraction=0.05,
                                                   extend='both', pad=0.15, label='Depth (km)')
            self.map_widget.lat.scatter(depth, lat, s=mag, c=depth, edgecolors="black", cmap=reversed_color_map)
            self.map_widget.lat.set_ylim((min_lat, max_lat))
            self.map_widget.lon.scatter(lon, depth, s=mag, c=depth, edgecolors="black", cmap=reversed_color_map)
            self.map_widget.lon.xaxis.tick_top()
            self.map_widget.lon.yaxis.tick_right()
            self.map_widget.lon.invert_yaxis()
            # self.map_widget.lon.set(xlabel='Longitude', ylabel='Depth (km)')
            self.map_widget.lon.set_xlim((min_lon, max_lon))

            # magnitude legend
            kw = dict(prop="sizes", num=5, fmt="{x:.0f}", color="red", func=lambda s: np.log(s / 0.5))
            self.map_widget.ax.legend(*cs.legend_elements(**kw), loc="lower right", title="Magnitudes")

            gl = self.map_widget.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                              linewidth=0.2, color='gray', alpha=0.2, linestyle='-')

            gl.top_labels = False
            gl.left_labels = False
            gl.xlines = False
            gl.ylines = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            self.map_widget.fig.canvas.draw()
        except:
            md = MessageDialog(self)
            md.set_error_message("Couldn't extract info from the DB, please check that your database "
                             "is loaded and is not empty")

    def plot_foc_mec(self, method, **kwargs):

        # Plot and save beachball from First Polarity or MTI
        if method == "First Polarity":

            strike = kwargs.pop('strike')
            dip = kwargs.pop('dip')
            rake = kwargs.pop('rake')
            fm = [strike, dip, rake]

        elif method == "MTI":

            mrr = kwargs.pop('mrr')
            mtt = kwargs.pop('mtt')
            mpp = kwargs.pop('mpp')
            mrt = kwargs.pop('mrt')
            mrp = kwargs.pop('mrp')
            mtp = kwargs.pop('mtp')
            fm = [mrr, mtt, mpp, mrt, mrp, mtp]

        ax = plt.axes()
        plt.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        lw = 2
        plt.xlim(-100 - lw / 2, 100 + lw / 2)
        plt.ylim(-100 - lw / 2, 100 + lw / 2)
        if method == "First Polarity":
            beach2 = beach(fm, facecolor='r', linewidth=1., alpha=0.8, width=80)
        elif method == "MTI":
            beach2 = beach(fm, facecolor='b', linewidth=1., alpha=0.8, width=80)
        ax.add_collection(beach2)
        outfile = os.path.join(ROOT_DIR, 'db/map_class/foc_mec.png')
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, transparent=True, edgecolor='none')
        plt.clf()
        plt.close()
