import math
import matplotlib.dates as mdates
import cartopy
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.transforms import offset_copy
from obspy import read_inventory, Inventory
from obspy.geodetics import gps2dist_azimuth
from isp.Gui.Frames import BaseFrame
from isp.Gui.Frames.qt_components import MessageDialog
from isp.Gui.Frames.uis_frames import UiEventLocationFrame
from isp.Gui.Models.sql_alchemy_model import SQLAlchemyModel
from isp.Utils.explote_meta import find_coords
from isp.Utils.statistics_utils import GutenbergRichterLawFitter
from isp.db import generate_id
from isp.db.models import EventLocationModel, FirstPolarityModel, MomentTensorModel, PhaseInfoModel
from datetime import datetime, timedelta
from isp.Utils import ObspyUtil
from obspy.core.event import Origin
from obspy.imaging.beachball import beach
from isp import ROOT_DIR, MAP_SERVICE_URL, MAP_LAYER
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from owslib.wms import WebMapService
import os
from sys import platform
import pandas as pd
import random


pqg = QtGui
pw = QtWidgets
pyc = QtCore
qt = pyc.Qt


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


class PhaseInfoDialog(pw.QDialog):
    tableView = None
    model = None

    def __init__(self, event_id, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Phase Info Inspector')

        columns = [getattr(PhaseInfoModel, c) for c in PhaseInfoModel.__table__.columns.keys()[2:]]

        col_names = ['Stat. Code', 'Instrument', 'Component', 'Phase', 'Polarity', 'Time', 'Amplitude', 'Travel Time',
                     'Time Residual', 'Time Weight', 'Distance (km)', 'Distance (deg)', 'Azimuth', 'Take Off angle']

        self.model = SQLAlchemyModel(PhaseInfoModel, columns, col_names, self)
        self.model.setFilter(PhaseInfoModel.event_info_id == event_id)
        self.model.revertAll()

        self.tableView = pw.QTableView()
        self.tableView.setModel(self.model)
        self.tableView.setSelectionBehavior(pw.QAbstractItemView.SelectRows)

        self.tableView.setItemDelegateForColumn(5, DateTimeFormatDelegate('dd/MM/yyyy hh:mm:ss.zzz'))
        self.tableView.resizeColumnsToContents()

        layout = pw.QHBoxLayout()
        layout.addWidget(self.tableView)
        self.setLayout(layout)

        self.setFixedHeight(500)
        self.setFixedWidth(1024)


class EventLocationFrame(BaseFrame, UiEventLocationFrame):
    phase_info_inspector = None
    def __init__(self):
        super(EventLocationFrame, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Events Location')
        self.setWindowIcon(pqg.QIcon(':\icons\compass-icon.png'))
        self.inv = None
        self.cb = None
        self.mag_amplification = 0.5
        self.topoCB.toggled.connect(self.set_topo_param_enable)
        self.set_topo_param_enable(False)
        self.topoCB.setChecked(False)
        # Connect the custom signal to update the latitude and longitude in MapApp
        self.map_widget.clickEventSignal.connect(self.double_click)
        self.magSlider.valueChanged.connect(self.updateSlider)


        el_columns = [getattr(EventLocationModel, c)
                      for c in EventLocationModel.__table__.columns.keys()[0:]]

        fp_columns = [getattr(FirstPolarityModel, c)
                      for c in FirstPolarityModel.__table__.columns.keys()[2:]]

        mti_columns = [getattr(MomentTensorModel, c)
                       for c in MomentTensorModel.__table__.columns.keys()[2:]]

        columns = [*el_columns, *fp_columns, *mti_columns]

        col_names = ['Id', 'Origin Time', 'Transformation', 'RMS [s]',
                     'Latitude', 'Longitude', 'Depth [m]', 'Depth Unc [km]',
                     'Semi Axis Maj [km]', 'Semi Axis Min [km]', 'Ellipse Az.',
                     'No. Phases', 'Az. Gap', 'Max. Dist.', 'Min. Dist.',
                     'Mb', 'Mb Error', 'Ms', 'Ms Error', 'Ml', 'Ml Error',
                     'Mw', 'Mw Error', 'Mc', 'Mc Error', 'Strike', 'Dip',
                     'Rake', 'Misfit', 'Az. Gap', 'Stat. Pol. Count', 'Latitude CMT', 'Longitude CMT', 'Depth CMT',
                     'VR', 'CN', 'dc', 'clvd', 'iso', 'Mw_mt', 'Mo', 'Strike_mt', 'dip_mt', 'rake_mt', 'mrr', 'mtt',
                     'mpp', 'mrt', 'mrp', 'mtp']

        entities = [EventLocationModel, FirstPolarityModel, MomentTensorModel]
        self.model = SQLAlchemyModel(entities, columns, col_names, self)
        self.model.addJoinArguments(EventLocationModel.first_polarity, isouter=True)
        self.model.addJoinArguments(EventLocationModel.moment_tensor, isouter=True)
        self.model.revertAll()
        sortmodel = pyc.QSortFilterProxyModel()
        sortmodel.setSourceModel(self.model)
        self.tableView.setModel(sortmodel)
        self.tableView.setColumnHidden(0, True)
        self.tableView.setSortingEnabled(True)
        self.tableView.sortByColumn(0, qt.AscendingOrder)
        self.tableView.setSelectionBehavior(pw.QAbstractItemView.SelectRows)
        self.tableView.setContextMenuPolicy(qt.ActionsContextMenu)

        remove_action = pw.QAction("Remove selected location(s)", self)
        remove_action.triggered.connect(self._onRemoveRowsTriggered)
        self.tableView.addAction(remove_action)

        watch_phase_action = pw.QAction("See current column phase info", self)
        watch_phase_action.triggered.connect(self._onShowPhaseInfo)
        self.tableView.addAction(watch_phase_action)

        copy_table = pw.QAction("Copy table to clipboard", self)
        copy_table.triggered.connect(self._copy_table)
        self.tableView.addAction(copy_table)

        highlight_Event = pw.QAction("Highlight Event on Map", self)
        highlight_Event.triggered.connect(self._highlight_Event)
        self.tableView.addAction(highlight_Event)

        self.tableView.setItemDelegateForColumn(1, DateTimeFormatDelegate('dd/MM/yyyy hh:mm:ss.zzz'))
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
        self.actionUpdate_Magnitudes.triggered.connect(self.update_magnitudes)
        self.actionUpdate_MTI.triggered.connect(self.update_mti)
        #self.actionRead_last_location.triggered.connect(self._readLastLocation)
        self.btnRefreshQuery.clicked.connect(self._refreshQuery)
        self.btnShowAll.clicked.connect(self._showAll)
        self.PlotMapBtn.clicked.connect(self.__plot_map)

        self.loadMetaBtn.clicked.connect(self.load_stations)
        # stations #


        ### statistics ###
        self.run_statisticsBtn.clicked.connect(self.plot_statistics)
        self.clearStatisticsBtn.clicked.connect(self.clear_statistics)

        self.showMTIBtn.clicked.connect(self.show_mtis)


    ## load metadata ##

    def load_stations(self):
        metadatata_file = ""
        selected = pw.QFileDialog.getOpenFileName(self, "Select file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            metadatata_file = selected[0]

        md = MessageDialog(self)
        try:
            self.inv = read_inventory(metadatata_file)
            md.set_info_message("Loaded Metadata")
        except:
            md.set_error_message("The file selected is not a valid metadata", )

    def updateSlider(self, value):

        self.mag_amplification = value/2

    ## statistics ##

    def plot_statistics(self):

        entities = self.model.getEntities()
        avarage = {}
        latitudes = []
        longitudes = []
        dates = []
        rmss = []
        cum_events = []
        magnitudes_mw = []
        magnitudes_ml = []
        depths = []
        depth_errors = []
        max_horizontal_errors = []
        min_horizontal_errors = []
        number_of_phasess = []
        ellipse_azimuths = []
        azimuthal_gaps = []
        max_distances = []
        min_distances = []
        for i, j in enumerate(entities):
            dates.append(j[0].origin_time)
            rmss.append(j[0].rms)
            latitudes.append(j[0].latitude)
            longitudes.append(j[0].longitude)
            max_horizontal_errors.append(j[0].max_horizontal_error)
            min_horizontal_errors.append(j[0].min_horizontal_error)
            depths.append(j[0].depth * 1E-3)
            depth_errors.append(j[0].uncertainty)
            number_of_phasess.append(j[0].number_of_phases)
            ellipse_azimuths.append(j[0].ellipse_azimuth)
            azimuthal_gaps.append(j[0].azimuthal_gap)
            max_distances.append(j[0].max_distance)
            min_distances.append(j[0].min_distance)
            #cum_events.append(i+1)



            if j[0].mw is None:
                random_float = random.uniform(-0.25, 1.5)
                magnitudes_mw.append(random_float)
            elif self.minMagCB.value() <= j[0].mw <= self.maxMagCB.value():
                magnitudes_mw.append(j[0].mw)

            if j[0].ml is None:
                random_float = random.uniform(-0.25, 1.5)
                magnitudes_ml.append(random_float)
            elif self.minMagCB.value() <= j[0].ml <= self.maxMagCB.value():
                magnitudes_ml.append(j[0].ml)

        if self.magPrefCB.currentText() == "Mw":
            magnitudes = magnitudes_mw
        elif self.magPrefCB.currentText() == "ML":
            magnitudes = magnitudes_ml

        dates = sorted(dates)
        for i in range(len(dates)):
            cum_events.append(i + 1)


        avarage["rmss"] = np.mean(rmss)
        avarage["max_horizontal_errors"] = np.mean(max_horizontal_errors)
        avarage["min_horizontal_errors"] = np.mean(min_horizontal_errors)
        avarage["latitudes"] = np.mean(latitudes)
        avarage["longitudes"] = np.mean(longitudes)
        avarage["depths"] = np.mean(depths)
        avarage["depths"] = np.mean(depths)
        avarage["depth_errors"] = np.mean(depth_errors)
        avarage["number_of_phasess"] = np.mean(number_of_phasess)
        avarage["ellipse_azimuths"] = np.mean(ellipse_azimuths)
        avarage["azimuthal_gaps"] = np.mean(azimuthal_gaps)
        avarage["max_distances"] = np.mean(max_distances)
        avarage["min_distances"] = np.mean(min_distances)
        avarage["magnitudes_mw"] = np.mean(magnitudes_mw)
        avarage["magnitudes_ml"] = np.mean(magnitudes_mw)

        self.print_statistics(avarage)
        if self.binsSelectionCB.currentText() == "Auto":
            min_date = min(dates)
            max_date = max(dates)
            one_month = timedelta(days=30)
            date_difference = max_date - min_date
            if date_difference <= one_month:
                self.statistics_widget.ax1.hist(dates, bins='auto', edgecolor='black', alpha=0.7)
                self.statistics_widget.ax1.xaxis.set_major_locator(mdates.DayLocator())
            elif date_difference >= one_month:
                self.statistics_widget.ax1.hist(dates, bins='auto', edgecolor='black', alpha=0.7)
                self.statistics_widget.ax1.xaxis.set_major_locator(mdates.MonthLocator())

        elif self.binsSelectionCB.currentText() == "Manual":
            self.statistics_widget.ax1.hist(dates, bins=self.numBinsSB.value(), edgecolor='black', alpha=0.7)

        self.statistics_widget.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.statistics_widget.ax1.set(ylabel='Number of Events')
        self.statistics_widget.ax1.set(xlabel='Date')
        self.statistics_widget.ax1.set_title('Number of Events vs Date')
        self.statistics_widget.ax1.tick_params(axis='x', rotation=30)
        self.statistics_widget.ax2.plot(dates, cum_events, color='orange')
        self.statistics_widget.ax2.set(ylabel='Cumulative Number of  Events')
        self.statistics_widget.ax2.xaxis.set_major_locator(mdates.MonthLocator())
        self.statistics_widget.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Fit Gutenberg-Richter Law using the dedicated class
        gr_fitter = GutenbergRichterLawFitter(magnitudes)
        if self.magBinsSB.value() !=0:
            params, bin_centers, counts = gr_fitter.fit_gutenberg_richter_law(bins=self.magBinsSB.value())
        else:
            params, bin_centers, counts = gr_fitter.fit_gutenberg_richter_law(bins=self.magBinsSB.value())

        self.statistics_widget.ax3.plot(bin_centers, np.log10(counts), 'bo', label='Event magnitudes', markersize=3,
                                        markeredgecolor='black')
        self.statistics_widget.ax3.set(ylabel='Log(Number of Events')
        #self.statistics_widget.ax3.set_yscale('log')  # Set y-axis to logarithmic scale

        # Plot the fitted line
        magnitude_range = np.linspace(min(magnitudes), max(magnitudes), 100)
        fitted_values = gr_fitter.gutenberg_richter_law(magnitude_range, *params)
        a = f'{params[0]:.2f}'
        b = f'{params[1]:.2f}'
        self.statistics_widget.ax3.plot(magnitude_range, fitted_values, color='red', label=f'Gutenberg–Richter '
                                                                                           f'(a = {a}, b={b})')
        self.statistics_widget.ax3.legend()
        self.statistics_widget.fig.canvas.draw()

    def clear_statistics(self):
        self.statistics_widget.ax1.clear()
        self.statistics_widget.ax2.clear()
        self.statistics_widget.ax3.clear()
        self.statistics_widget.fig.canvas.draw()
        self.statistics_widget.ax2 = self.statistics_widget.ax1.twinx()
        #self.statistics_widget.ax2.set_label_position("right")
        #self.statistics_widget.ax2.tick_right()

    def print_statistics(self, avarage):
        self.statistics_Text.clear()
        self.statistics_Text.appendPlainText("Focal Parameters Avarage")

        latitudes = str("{: .2f}".format(avarage["latitudes"]))
        longitudes = str("{: .2f}".format(avarage["longitudes"]))

        rms = str("{: .2f}".format(avarage["rmss"]))
        depth = str("{: .2f}".format(avarage["depths"]))
        depth_uncertainty = str("{: .2f}".format(avarage["depth_errors"]))
        max_horizontal_errors = str("{: .2f}".format(avarage["max_horizontal_errors"]))
        min_horizontal_errors = str("{: .2f}".format(avarage["min_horizontal_errors"]))
        number_of_phases = str(int(avarage["number_of_phasess"]))
        ellipse_azimuths = str("{: .1f}".format(avarage["ellipse_azimuths"]))
        azimuthal_gaps = str("{: .1f}".format(avarage["azimuthal_gaps"]))
        max_distances = str("{: .3f}".format(avarage["max_distances"]))
        min_distances = str("{: .3f}".format(avarage["min_distances"]))

        self.statistics_Text.appendPlainText("Latitude: {latitudes}º".format(latitudes=latitudes))
        self.statistics_Text.appendPlainText("Longitude: {longitudes}º".format(longitudes=longitudes))
        self.statistics_Text.appendPlainText("RMS: {RMS} s".format(RMS=rms))
        self.statistics_Text.appendPlainText("Depth: {Depth} km ".format(Depth=depth))
        self.statistics_Text.appendPlainText("Depth Uncertainty: {Depth_Uncertainty} km ".
                                             format(Depth_Uncertainty = depth_uncertainty))

        self.statistics_Text.appendPlainText("Max Horizontal Unc: {max_horizontal_errors} km ".
                                             format(max_horizontal_errors = max_horizontal_errors))
        self.statistics_Text.appendPlainText("Min Horizontal Unc: {min_horizontal_errors} km ".
                                             format(min_horizontal_errors = min_horizontal_errors))
        self.statistics_Text.appendPlainText("Number of Phases: {number_of_phases} ".
                                             format(number_of_phases=number_of_phases))
        self.statistics_Text.appendPlainText("Ellipse Azimuth: {ellipse_azimuths}º ".
                                             format(ellipse_azimuths=ellipse_azimuths))
        self.statistics_Text.appendPlainText("Azimuthal GAP: {azimuthal_gaps}º ".
                                             format(azimuthal_gaps=azimuthal_gaps))
        self.statistics_Text.appendPlainText("Max Distance: {max_distances}º ".
                                             format(max_distances=max_distances))
        self.statistics_Text.appendPlainText("Min Distance: {min_distances}º ".
                                             format(min_distances=min_distances))
        if "magnitudes_mw" in avarage.keys():
            magnitudes_mw = str("{: .2f}".format(avarage["magnitudes_mw"]))
            self.statistics_Text.appendPlainText("Magnitude: Mw {magnitudes_mw} ".
                                                 format(magnitudes_mw=magnitudes_mw))

        if "magnitudes_ml" in avarage.keys():
            magnitudes_ml = str("{: .2f}".format(avarage["magnitudes_ml"]))
            self.statistics_Text.appendPlainText("Magnitude: ML {magnitudes_ml} ".
                                                 format(magnitudes_ml=magnitudes_ml))


    def set_topo_param_enable(self, enabled):
        self.wmsLE.setEnabled(enabled)
        self.layerLE.setEnabled(enabled)

    def on_click_select_file(self):
        selected = pw.QFileDialog.getOpenFileName(self, "Select file")
        if isinstance(selected[0], str) and os.path.isfile(selected[0]):
            file_path = selected[0]
        return file_path

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
            min_mag = -math.inf
            max_mag = math.inf

            for event in events:
                if event.latitude > max_lat: max_lat = event.latitude
                if event.latitude < min_lat: min_lat = event.latitude
                if event.longitude > max_lon: max_lon = event.longitude
                if event.longitude < min_lon: min_lon = event.longitude
                if event.depth > max_dep: max_dep = event.depth
                if event.depth < min_dep: min_dep = event.depth
                if event.ml is not None:
                    if event.ml > max_mag: max_mag = event.ml
                    if event.ml < min_mag: min_mag = event.ml
                if event.origin_time < min_orig: min_orig = event.origin_time
                if event.origin_time > max_orig: max_orig = event.origin_time

            self.maxLat.setValue(math.ceil(max_lat))
            self.minLat.setValue(math.floor(min_lat))
            self.maxLon.setValue(math.ceil(max_lon))
            self.minLon.setValue(math.floor(min_lon))
            self.maxDepth.setValue(math.ceil(max_dep))
            self.minDepth.setValue(math.floor(min_dep))
            self.maxMag.setValue(max_mag)
            self.minMag.setValue(min_mag)
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

    def _onShowPhaseInfo(self):
        current_idx = self.tableView.currentIndex()
        if not current_idx or not current_idx.isValid():
            return
        self.phase_info_inspector = PhaseInfoDialog(
            self.tableView.model().data(self.tableView.model().index(current_idx.row(), 0)), self)

        self.phase_info_inspector.show()

    def _copy_table(self):
        date_format = "MM.dd.yyyy hh:mm:ss.zzz"
        index_list = self.tableView.selectionModel().selectedRows()
        model = self.tableView.model()
        col_num = model.columnCount()
        results = ""
        for i in index_list:
            for j in range(1, col_num):
                var = model.data(model.index(i.row(), j))
                if isinstance(var, pyc.QDateTime):
                    results += var.toString(date_format)
                else:
                    results += str(var)
                results += " "

            results += "\n"

        pqg.QGuiApplication.instance().clipboard().setText(results)

    def _checkQueryParameters(self):
        self.btnRefreshQuery.setEnabled(all(v.valid for v in self._validators))

    def _update_magnitudes(self, magnitudes_dict):

        event_model = EventLocationModel.find_by(latitude=magnitudes_dict["lats"], longitude=magnitudes_dict["longs"],
                                                 origin_time=magnitudes_dict["date_id"])
        if event_model:
            event_model.mw = magnitudes_dict["mw"]
            event_model.mw_error = magnitudes_dict["mw_error"]
            event_model.ml = magnitudes_dict["ml"]
            event_model.ml_error = magnitudes_dict["ml_error"]
            event_model.save()

    def update_magnitudes(self):
        magnitude_file = self.on_click_select_file()
        df = pd.read_csv(magnitude_file, sep=";", na_values='missing')
        date_format = "%m/%d/%Y, %H:%M:%S.%f"
        for i in range(len(df)):
            data = {}
            info = df.loc[i]
            data["date_id"] = datetime.strptime(info.date_id, date_format)
            data["lats"] = info.lats
            data["longs"] = info.longs
            data["depths"] = info.depths * 1E3
            data["mw"] = info.Mw
            data["mw_error"] = info.Mw_error
            data["ml"] = info.ML
            data["ml_error"] = info.ML_error
            self._update_magnitudes(data)

    def _update_mti(self, info_mti):
        mti_dict = {}
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        date = datetime.strptime(info_mti.date_id, date_format)
        event_model = EventLocationModel.find_by(origin_time=date)
        mti = MomentTensorModel.find_by(event_info_id=event_model.id)
        if mti:
            mti.delete()
        mti_dict['latitude'] = info_mti.lat
        mti_dict['longitude'] = info_mti.long
        mti_dict['depth'] = info_mti.depth
        mti_dict['VR'] = info_mti.vr
        mti_dict['CN'] = info_mti.cn
        mti_dict['dc'] = info_mti.dc
        mti_dict['clvd'] = info_mti.clvd
        mti_dict['iso'] = info_mti.isotropic_component
        mti_dict['mw_mt'] = info_mti.mw
        mti_dict['mo'] = info_mti.mo
        mti_dict['strike_mt'] = info_mti.plane_1_strike
        mti_dict['dip_mt'] = info_mti.plane_1_dip
        mti_dict['rake_mt'] = info_mti.plane_1_slip_rake
        mti_dict['mtt'] = info_mti.mtt
        mti_dict['mpp'] = info_mti.mpp
        mti_dict['mrr'] = info_mti.mrr
        mti_dict['mrt'] = info_mti.mrt
        mti_dict['mrp'] = info_mti.mrp
        mti_dict['mtp'] = info_mti.mtp
        mti_dict['event_info_id'] = event_model.id
        mti_dict['id'] = generate_id(16)
        mti = MomentTensorModel.from_dict(mti_dict)
        mti.save()
        self._showAll()
        self.refreshLimits()

    def update_mti(self):
        magnitude_file = self.on_click_select_file()
        df = pd.read_csv(magnitude_file, sep=";", na_values='missing')
        for i in range(len(df)):
            info = df.loc[i]
            self._update_mti(info)

    def _readHypFile(self, file_abs_path):
        origin: Origin = ObspyUtil.reads_hyp_to_origin(file_abs_path)
        event_model = EventLocationModel.find_by(latitude=origin.latitude, longitude=origin.longitude,
                                                 depth=origin.depth, origin_time=origin.time.datetime)

        if event_model:
            print("Even location already exist")
            return

        event_model = EventLocationModel.create_from_origin(origin)
        phases = PhaseInfoModel.create_phases_from_origin(
            event_model.id,
            picks=ObspyUtil.reads_pick_info(file_abs_path)
        )
        for phase in phases:
            event_model.add_phase(phase)
        event_model.save()

        return event_model

    def _readHypFolder(self):

        if "darwin" == platform:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Get directory to read .hyp files from')
        else:
            dir_path = pw.QFileDialog.getExistingDirectory(self, 'Select Directory', "",
                                                           pw.QFileDialog.DontUseNativeDialog)

        # If user cancels selecting folder, return
        if not dir_path:
            return

        files = [f for f in os.listdir(dir_path) if f.endswith('.hyp')]
        errors = []
        for file in files:
            file_abs = os.path.abspath(os.path.join(dir_path, file))
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


    def _refreshQuery(self):
        lat = EventLocationModel.latitude.between(self.minLat.value(), self.maxLat.value())
        lon = EventLocationModel.longitude.between(self.minLon.value(), self.maxLon.value())
        depth = EventLocationModel.depth.between(self.minDepth.value(), self.maxDepth.value())
        mag = EventLocationModel.ml.between(self.minMag.value(), self.maxMag.value())
        minOrig = self.minOrig.dateTime().toPyDateTime()
        maxOrig = self.maxOrig.dateTime().toPyDateTime()
        time = EventLocationModel.origin_time.between(minOrig, maxOrig)
        self.model.setFilter(lat, lon, depth, mag, time)
        self.model.revertAll()

    def get_entities(self):
        return self.model.getEntities()

    def get_model(self):
        return EventLocationModel

    def _showAll(self):
        self.model.setFilter()
        self.model.revertAll()

    def __plot_map(self):

        URL = self.wmsLE.text()
        layer = self.layerLE.text()
        if not URL and not layer:
            URL = MAP_SERVICE_URL
            layer = MAP_LAYER
        self.plot_map(topography=self.topoCB.isChecked(), map_service=URL, layer=layer)

    def update_inset_axes(self, event):

        self.map_widget.lat.set_ylim(self.map_widget.ax.get_ylim())
        self.map_widget.lon.set_xlim(self.map_widget.ax.get_xlim())

    def update_resize_axes(self, event):

        self.map_widget.lat.set_ylim(self.map_widget.ax.get_ylim())
        self.map_widget.lon.set_xlim(self.map_widget.ax.get_xlim())

    def plot_map(self, topography=False, map_service=MAP_SERVICE_URL,
                 layer=MAP_LAYER):

        self.map_widget.fig.canvas.mpl_connect('draw_event', self.update_inset_axes)
        self.map_widget.fig.canvas.mpl_connect('resize_event', self.update_resize_axes)

        try:
            wms = ""

            self.map_widget.ax.clear()
            self.map_widget.lon.clear()
            self.map_widget.lat.clear()

            #Clear existing colorbar
            if hasattr(self, 'cb') and self.cb:
                try:
                    self.cb.remove()
                except Exception as e:
                    #print(f"Error removing colorbar: {e}")
                    pass

            MAP_SERVICE_URL = map_service
            try:
                if topography:
                    wms = WebMapService(MAP_SERVICE_URL)
                    list(wms.contents)
            except:
                print("No topography detected")
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

                if j[0].mw is None:
                    j[0].mw = 2.5

                mag.append(j[0].mw)

            # print(entities)

            mag = np.array(mag)
            #mag_original = 0.5 * np.exp(1.2*mag)
            mag = self.mag_amplification * np.exp(1.2*mag)
            min_lon = min(lon) - 0.5
            max_lon = max(lon) + 0.5
            min_lat = min(lat) - 0.5
            max_lat = max(lat) + 0.5
            extent = [min_lon, max_lon, min_lat, max_lat]

            self.map_widget.ax.set_extent(extent, crs=ccrs.PlateCarree())

            try:
                if topography:
                    self.map_widget.ax.add_wms(wms, layer)
                else:
                    coastline_10m = cartopy.feature.NaturalEarthFeature('physical', 'coastline', '10m',
                                                                        edgecolor='k', linewidth=0.5,
                                                                        facecolor=cartopy.feature.COLORS['land'])
                    self.map_widget.ax.add_feature(coastline_10m, zorder=1)

            except:
                os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join(ROOT_DIR, "maps")
                self.map_widget.ax.background_img(name='ne_shaded', resolution="high")

            lon = np.array(lon)
            lat = np.array(lat)
            depth = np.array(depth) / 1000
            color_map = plt.cm.get_cmap('rainbow')
            reversed_color_map = color_map.reversed()
            cs = self.map_widget.ax.scatter(lon, lat, s=mag, c=depth, edgecolors="black", cmap=reversed_color_map,
                                            transform=ccrs.PlateCarree(), zorder=2)

            cax = self.map_widget.fig.add_axes([0.87, 0.1, 0.02, 0.8])  # Adjust these values as needed
            self.cb = self.map_widget.fig.colorbar(cs, cax=cax, orientation='vertical', extend='both', label='Depth (km)')
            self.map_widget.lat.scatter(depth, lat, s=mag, c=depth, edgecolors="black", cmap=reversed_color_map)
            self.map_widget.lat.set_ylim((min_lat, max_lat))
            self.map_widget.lon.scatter(lon, depth, s=mag, c=depth, edgecolors="black", cmap=reversed_color_map)
            self.map_widget.lon.xaxis.tick_top()
            self.map_widget.lon.yaxis.tick_right()
            self.map_widget.lat.set(ylabel='Latitude')
            self.map_widget.lon.invert_yaxis()
            self.map_widget.lon.yaxis.set_label_position('left')
            self.map_widget.lon.set(xlabel='Longitude')
            self.map_widget.lon.set_xlim((min_lon, max_lon))
            self.map_widget.lon.set_xbound((min_lon, max_lon))
            self.map_widget.lon.xaxis.set_label_position('top')
            self.map_widget.lon.yaxis.set_label_position('right')
            self.map_widget.lon.xaxis.set_ticks_position('top')
            self.map_widget.lon.yaxis.set_ticks_position('right')

            # plot stations
            if isinstance(self.inv, Inventory):
                names = []
                all_lon = []
                all_lat = []
                networks = find_coords(self.inv)
                for key in networks.keys():
                    names += networks[key][0]
                    all_lon += networks[key][1]
                    all_lat += networks[key][2]
                self.map_widget.ax.scatter(all_lon, all_lat, s=75, marker="^", color='black', edgecolors="white",
                                           alpha=0.7, transform=ccrs.PlateCarree(), zorder=3)

                if self.stationNameCB.isChecked():
                    N = len(names)
                    geodetic_transform = ccrs.PlateCarree()._as_mpl_transform(self.map_widget.ax)
                    text_transform = offset_copy(geodetic_transform, units='dots', x=-25)
                    for n in range(N):
                        lon1 = all_lon[n]
                        lat1 = all_lat[n]
                        name = names[n]

                        self.map_widget.ax.text(lon1, lat1, name, verticalalignment='center', horizontalalignment='right',
                                transform=text_transform,
                                bbox=dict(facecolor='sandybrown', alpha=0.5, boxstyle='round'))


            # magnitude legend
            #kw = dict(prop="sizes", num=5, fmt="{x:.1f}", color="red", func=lambda mag_original: np.log(mag_original / 0.5))
            kw = dict(prop="sizes", num=5, fmt="{x:.1f}", color="red", func=lambda mag_original: np.log(mag_original /
                                                                                                        self.mag_amplification))

            self.map_widget.ax.legend(*cs.legend_elements(**kw), loc="lower right", title="Magnitudes")
            self.map_widget.fig.subplots_adjust(right=0.986, bottom=0.062, top=0.828, left=0.014)
            self.map_widget.fig.canvas.draw()
            self.inverted = False

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


    def _highlight_Event(self):
        # Plot Selected Event
        model = self.tableView.model()
        current_idx = self.tableView.currentIndex()
        lat = model.data(model.index(current_idx.row(), 4))
        lon = model.data(model.index(current_idx.row(), 5))
        lon_lat_transform = ccrs.PlateCarree()._as_mpl_transform(self.map_widget.ax)
        self.map_widget.ax.annotate("Event Selected", xy=(lon, lat), xycoords=lon_lat_transform,
                                    xytext=(lon + 0.3, lat + 0.3), textcoords=lon_lat_transform,
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        self.map_widget.fig.canvas.draw()

    def _plot_foc_mec(self, index):

        # Plot Focal Mechanism
        model = self.tableView.model()
        check = False
        if self.methodCB.currentText() == "First Polarity":

            strike = model.data(model.index(index.row(), 25))
            dip = model.data(model.index(index.row(), 26))
            rake = model.data(model.index(index.row(), 27))
            if None not in [strike, dip, rake]:
                self.plot_foc_mec(method=self.methodCB.currentText(), strike=strike, dip=dip, rake=rake)
                check = True

        if self.methodCB.currentText() == "MTI":
            mrr = model.data(model.index(index.row(), 44))
            mtt = model.data(model.index(index.row(), 45))
            mpp = model.data(model.index(index.row(), 46))
            mrt = model.data(model.index(index.row(), 47))
            mrp = model.data(model.index(index.row(), 48))
            mtp = model.data(model.index(index.row(), 49))

            if None not in [mrr, mtt, mpp, mrt, mrp, mtp]:
                self.plot_foc_mec(method=self.methodCB.currentText(), mrr=mrr, mtt=mtt, mpp=mpp, mrt=mrt, mrp=mrp,
                                  mtp=mtp)
                check = True

        if check:
            # plot in the map
            lat = model.data(model.index(index.row(), 4))
            lon = model.data(model.index(index.row(), 5))
            print(lat, lon)
            file = os.path.join(ROOT_DIR, 'db/map_class/foc_mec.png')
            random_number = random.uniform(-0.3, 0.3)
            img = Image.open(file)
            imagebox = OffsetImage(img, zoom=0.08)
            imagebox.image.axes = self.map_widget.ax
            ab = AnnotationBbox(imagebox, [lon + random_number, lat - random_number], frameon=False)
            self.map_widget.ax.add_artist(ab)
            lon_lat_transform = ccrs.PlateCarree()._as_mpl_transform(self.map_widget.ax)
            self.map_widget.ax.annotate('', xy=(lon, lat), xycoords=lon_lat_transform,
                                        xytext=(lon + random_number, lat - random_number), textcoords=lon_lat_transform,
                                        arrowprops=dict(arrowstyle="->",
                                                        connectionstyle="arc3,rad=.2"))

            self.map_widget.fig.canvas.draw()

    def double_click(self, latitude, longitude):

        self.dataSelect(longitude, latitude)

    def dataSelect(self, lon1, lat1):
        dist = []
        selection_model = self.tableView.model()
        for row in range(selection_model.rowCount()):
            lat_index = selection_model.index(row, 4)
            lon_index = selection_model.index(row, 5)
            # Retrieve data from the model using the index
            lat = lat_index.data()
            lon = lon_index.data()
            great_arc, az0, az2 = gps2dist_azimuth(lat1, lon1, lat, lon, a=6378137.0, f=0.0033528106647474805)
            dist.append(great_arc)
        idx, val = self.find_nearest(dist, min(dist))
        self.tableView.selectRow(idx)


    def find_nearest(self, array, value):
        idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
        return idx, val

    def show_mtis(self):
        entities = self.model.getEntities()
        moments = []
        for entity in entities:
            if entity[0].moment_tensor:
                mt = entity[0].moment_tensor[0]
                moments.append([mt.mrr, mt.mtt, mt.mpp, mt.mrt, mt.mrp, mt.mtp, mt.latitude, mt.longitude])

        for moment in moments:
            self.plot_foc_mec(method=self.methodCB.currentText(), mrr=moment[0], mtt=moment[1], mpp=moment[2],
                              mrt=moment[3], mrp=moment[4], mtp=moment[5])
            # plot in the map
            lat = moment[6]
            lon = moment[7]
            file = os.path.join(ROOT_DIR, 'db/map_class/foc_mec.png')
            random_number = random.uniform(-0.3, 0.3)
            img = Image.open(file)
            imagebox = OffsetImage(img, zoom=0.08)
            imagebox.image.axes = self.map_widget.ax
            ab = AnnotationBbox(imagebox, [lon + random_number, lat - random_number], frameon=False)
            self.map_widget.ax.add_artist(ab)
            lon_lat_transform = ccrs.PlateCarree()._as_mpl_transform(self.map_widget.ax)
            self.map_widget.ax.annotate('', xy=(lon, lat), xycoords=lon_lat_transform,
                                        xytext=(lon + random_number, lat - random_number), textcoords=lon_lat_transform,
                                        arrowprops=dict(arrowstyle="->",
                                                        connectionstyle="arc3,rad=.2"))

            self.map_widget.fig.canvas.draw()

