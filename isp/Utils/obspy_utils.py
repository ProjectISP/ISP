import math
import pickle
from datetime import datetime
from enum import unique, Enum
from multiprocessing import Pool
from typing import List
import pandas as pd
import numpy as np
from obspy import Stream, read, Trace, UTCDateTime, read_events, Inventory
from obspy.core.event import Origin
from obspy.core.inventory import Station, Network
from obspy.geodetics import gps2dist_azimuth, degrees2kilometers
from obspy.io.mseed.core import _is_mseed
from obspy.io.stationxml.core import _is_stationxml
from obspy.io.xseed.parser import Parser
from obspy.taup import TauPyModel
from surfquakecore.project.surf_project import SurfProject
from isp import PICKING_DIR
from isp.Exceptions import InvalidFile
from isp.Structures.structures import TracerStats
from isp.Utils.nllOrgErrors import computeOriginErrors
import os
import re
from pathlib import Path
from isp.Utils import read_nll_performance
from scipy.signal import cheby1, cheby2, ellip, bessel, sosfilt, sosfiltfilt

@unique
class Filters(Enum):

    Default = "Filter"
    BandPass = "bandpass"
    BandStop = "bandstop"
    LowPass = "lowpass"
    HighPass = "highpass"

    def __eq__(self, other):
        if type(other) is str:
            return self.value == other
        else:
            return self.value == other.value

    def __ne__(self, other):
        if type(other) is str:
            return self.value != other
        else:
            return self.value != other.value

    @classmethod
    def get_filters(cls):
        return [item.value for item in cls.__members__.values()]


class ObspyUtil:


    @staticmethod
    def stationsCoodsFromMeta(dataXml):

        sta_names = []
        latitudes = []
        longitudes = []
        networks = {}

        for network in dataXml:
            for station in network.stations:
                if station.code not in sta_names:
                    sta_names.append(network.code + "." + station.code)
                    latitudes.append(station.latitude)
                    longitudes.append(station.longitude)
                else:
                    pass
            networks[network.code] = [sta_names, longitudes, latitudes]

        return networks

    @staticmethod
    def get_figure_from_stream(st: Stream, **kwargs):
        if st:
            return st.plot(show=False, **kwargs)
        return None

    @staticmethod
    def get_tracer_from_file(file_path) -> Trace:
        st = read(file_path)
        return st[0]

    @staticmethod
    def get_stats(file_path):
        """
        Reads only the header for the metadata and return a :class:`TracerStats`.
        :param file_path: The full file's path for the mseed.
        :return: A TracerStats contain the metadata.
        """
        st = read(file_path, headonly=True)
        tr = st[0]
        stats = TracerStats.from_dict(tr.stats)
        return stats

    @staticmethod
    def get_stats_from_trace(tr: Trace):

        """
        Reads only the header for the metadata and return a :class:`TracerStats`.
        :param ftrace: obspy trace.
        :return: A Dictionary with TracerStats contain the metadata.
        """
        net = tr.stats.network
        station = tr.stats.station
        location = tr.stats.location
        channel = tr.stats.channel
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime
        npts= tr.stats.npts
        sampling_rate=tr.stats.sampling_rate
        stats =  {'net': net, 'station': station, 'location':location, 'channel':channel, 'starttime':starttime,
                  'endtime':endtime,'npts': npts, 'sampling_rate':sampling_rate}
        return stats

    @staticmethod
    def get_stations_from_stream(st: Stream):

        stations = []

        for tr in st:
            station = tr.stats.station
            if stations.count(station):
                pass
            else:
                stations.append(station)

        return stations

    @staticmethod
    def get_trip_times(source_depth, min_dist, max_dist, phases):

        model = TauPyModel(model="iasp91")
        distances = np.linspace(min_dist, max_dist, 25)
        arrivals_list = []



        for value in distances:

            if phases == ["ALL"]:
                arrival = model.get_travel_times(source_depth_in_km=source_depth, distance_in_degree=float(value))

            else:
                arrival = model.get_travel_times(source_depth_in_km=source_depth, distance_in_degree=float(value),
                                             phase_list = phases)


            arrivals_list.append(arrival)

        return arrivals_list

    @staticmethod
    def convert_travel_times(arrivals, otime, dist_km=True):

        all_arrival = {}

        # Loop over arrivals objects in list
        for arrival_set in arrivals:
            # Loop over phases in list
            phase_id_check = []
            for arrival in arrival_set:
                phase_id = arrival.purist_name

                if phase_id not in phase_id_check:
                    phase_id_check.append(phase_id)

                    time = otime + arrival.time

                    dist = arrival.purist_distance % 360.0
                    distance = arrival.distance
                    if distance < 0:
                        distance = (distance % 360)
                    if abs(dist - distance) / dist > 1E-5:
                        continue

                    if dist_km:
                        distance = degrees2kilometers(distance)

                    if phase_id in all_arrival.keys():
                        all_arrival[phase_id]["times"].append(time.matplotlib_date)
                        all_arrival[phase_id]["distances"].append(distance)

                    else:
                        all_arrival[phase_id] = {}
                        all_arrival[phase_id]["times"] = [time.matplotlib_date]
                        all_arrival[phase_id]["distances"] = [distance]

        return all_arrival

    @staticmethod
    def coords2azbazinc(station_latitude, station_longitude,station_elevation, origin_latitude,
                        origin_longitude, origin_depth):

        """
        Returns azimuth, backazimuth and incidence angle from station coordinates
        given in first trace of stream and from event location specified in origin
        dictionary.
        """

        dist, bazim, azim = gps2dist_azimuth(station_latitude, station_longitude, float(origin_latitude),
                                             float(origin_longitude))
        elev_diff = station_elevation - float(origin_depth)
        inci = math.atan2(dist, elev_diff) * 180.0 / math.pi

        return azim, bazim, inci

    @staticmethod
    def rename_traces(st: Stream):
        for tr in st:
            old_channel = tr.stats.channel
            if old_channel.endswith('1') or old_channel.endswith('Y'):
                tr.stats.channel = old_channel[:-1] + "N"
            elif old_channel.endswith('2') or old_channel.endswith('X'):
                tr.stats.channel = old_channel[:-1] + "E"
        return st



    @staticmethod
    def filter_trace(trace, trace_filter, f_min, f_max, **kwargs):
        """
            Filter an ObsPy Trace using standard and advanced filters.
            Supports Butterworth, Chebyshev I & II, Elliptic, Bessel.

            Parameters:
                trace: ObsPy Trace object.
                trace_filter: Filter type as string or enum.
                f_min, f_max: Bandpass frequency limits (Hz).
                corners: Number of poles (default: 4).
                zerophase: Apply filter forward and backward (default: True).
                ripple: Optional ripple for Chebyshev/Elliptic filters.

            Returns:
                True if success, False if bad parameters.
            """
        if trace_filter != Filters.Default:
            tf = trace_filter.lower()

            if tf in ["bandpass", "bandstop"]:
                if not (f_max > f_min):
                    print("Bad frequency range: f_max must be > f_min for band filters.")
                    return False

            elif tf in ["highpass"]:
                if f_min <= 0:
                    print("Invalid highpass cutoff: f_min must be > 0.")
                    return False

            elif tf in ["lowpass"]:
                if f_max <= 0:
                    print("Invalid lowpass cutoff: f_max must be > 0.")
                    return False

            corners = kwargs.pop("corners", 4)
            zerophase = kwargs.pop("zerophase", True)
            ripple = kwargs.pop("ripple", 1)  # in dB
            rp = kwargs.pop("rp", 0.5)  # Passband ripple
            rs = kwargs.pop("rs", 60)  # Stopband attenuation
            sr = trace.stats.sampling_rate
            nyq = 0.5 * sr
            wp = [f_min / nyq, f_max / nyq]  # normalized passband
            trace.detrend(type="simple")
            trace.taper(max_percentage=0.05, type="cosine")

            if trace_filter.lower() in ["bandpass", "bandstop"]:
                trace.filter(trace_filter, freqmin=f_min, freqmax=f_max, corners=corners, zerophase=zerophase)

            elif trace_filter.lower() in ["highpass"]:
                trace.filter(trace_filter, freq=f_min, corners=corners, zerophase=zerophase)

            elif trace_filter.lower() in ["lowpass"]:
                trace.filter(trace_filter, freq=f_max, corners=corners, zerophase=zerophase)

            elif trace_filter.lower() == "cheby1":

                sos = cheby1(N=corners, rp=ripple, Wn=wp, btype='band', output='sos')

                trace.data = sosfiltfilt(sos, trace.data) if zerophase else sosfilt(sos, trace.data)

            elif trace_filter.lower() == "cheby2":
                sos = cheby2(N=corners, rs=ripple, Wn=wp, btype='band', output='sos')
                trace.data = sosfiltfilt(sos, trace.data) if zerophase else sosfilt(sos, trace.data)

            elif trace_filter.lower() == "elliptic":
                sos = ellip(N=corners, rp=rp, rs=rs, Wn=wp, btype='band', output='sos')
                trace.data = sosfiltfilt(sos, trace.data) if zerophase else sosfilt(sos, trace.data)

            elif trace_filter.lower() == "bessel":
                # Warning: Bessel filters are not typically available in 'sos' format for all scipy versions
                sos = bessel(N=corners, Wn=wp, btype='band', norm='phase', output='sos')
                trace.data = sosfiltfilt(sos, trace.data) if zerophase else sosfilt(sos, trace.data)

            else:
                print(f"Unsupported filter type: {trace_filter}")
                return False

        return True

    @staticmethod
    def merge_files_to_stream(files_path: List[str], *args, **kwargs) \
            -> Stream:
        """
        Reads all files in the list and concatenate in a Stream.
        :param files_path: A list of valid mseed files.
        :arg args: Valid arguments of obspy.read().
        :keyword kwargs: Valid kwargs for obspy.read().
        :return: The concatenate stream.
        """
        st = Stream()
        for file in files_path:
            if MseedUtil.is_valid_mseed(file):
                st += read(file, *args, **kwargs)
            else:
                raise InvalidFile("The file {} either doesn't exist or is not a valid mseed.".format(file))
        return st

    @staticmethod
    def trim_stream(st: Stream, start_time: UTCDateTime, end_time: UTCDateTime):
        """
        This method is a safe wrapper to Stream.trim(). If start_time and end_time don't overlap the
        stream, it will be trimmed by the maximum start time and minimum end time within its tracers .
        :param st: The Stream to be trimmed.
        :param start_time: The UTCDatetime for start the trim.
        :param end_time: The UTCDatetime for end the trim.
        :return:
        """
        max_start_time = np.max([tr.stats.starttime for tr in st])
        min_end_time = np.min([tr.stats.endtime for tr in st])
        st.trim(max_start_time, min_end_time)

        overlap = start_time < min_end_time and max_start_time < end_time  # check if dates overlap.
        if overlap:
            if max_start_time - start_time < 0 < min_end_time - end_time:  # trim start and end time
                st.trim(start_time, end_time)
            elif max_start_time - start_time < 0:  # trim only start time.
                st.trim(starttime=start_time)
            elif min_end_time - end_time > 0:  # trim only end time.
                st.trim(endtime=end_time)

    @staticmethod
    def reads_hyp_to_origin(hyp_file_path: str, modified=False) -> Origin:

        import warnings
        warnings.filterwarnings("ignore")

        """
        Reads an hyp file and returns the Obspy Origin.
        :param hyp_file_path: The file path to the .hyp file
        :param modified: To use the modified version of read_events, including more info into Origin
        :return: An Obspy Origin
        """

        if os.path.isfile(hyp_file_path):
            if modified==False:
                cat = read_events(hyp_file_path)
            else:
                cat = read_nll_performance.read_nlloc_hyp_ISP(hyp_file_path)
            event = cat[0]
            origin = event.origins[0]
            modified_origin_90 = computeOriginErrors(origin)
            origin.depth_errors["uncertainty"] = modified_origin_90['depth_errors'].uncertainty
            origin.origin_uncertainty.max_horizontal_uncertainty = modified_origin_90['origin_uncertainty'].max_horizontal_uncertainty
            origin.origin_uncertainty.min_horizontal_uncertainty = modified_origin_90[
                'origin_uncertainty'].min_horizontal_uncertainty
            origin.origin_uncertainty.azimuth_max_horizontal_uncertainty = modified_origin_90['origin_uncertainty'].azimuth_max_horizontal_uncertainty

        if modified:
            return origin, event
        else:
            return origin


    @staticmethod
    def get_ellipse(hyp_file_path: str):

        ellipse = {}
        if isinstance(hyp_file_path, str) and os.path.isfile(hyp_file_path):

            origin: Origin = ObspyUtil.reads_hyp_to_origin(hyp_file_path)
            latitude = origin.latitude
            longitude = origin.longitude
            smin = origin.origin_uncertainty.min_horizontal_uncertainty
            smax = origin.origin_uncertainty.max_horizontal_uncertainty
            azimuth = origin.origin_uncertainty.azimuth_max_horizontal_uncertainty
            ellipse['latitude'] = latitude
            ellipse['longitude'] = longitude
            ellipse['smin'] = smin
            ellipse['smax'] = smax
            ellipse['azimuth'] = azimuth

        azimuth = (90 - ellipse['azimuth']) % 360

        azimuth_minor = azimuth + 90

        # Generate angles from 0 to 2*pi
        angles = np.linspace(0, 2 * np.pi, 100)

        # Calculate the major and minor axes vectors
        major_axis = np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))]) * ellipse['smax']
        minor_axis = np.array([np.cos(np.radians(azimuth_minor)), np.sin(np.radians(azimuth_minor))]) * ellipse['smin']

        # Calculate ellipse points
        den = 111.2 * np.cos(np.radians(ellipse['latitude']))
        x_points = ellipse['longitude'] + (major_axis[0] * np.cos(angles) + minor_axis[0] * np.sin(angles)) / den
        y_points = ellipse['latitude'] + (major_axis[1] * np.cos(angles) + minor_axis[1] * np.sin(angles)) / 111.2

        return x_points, y_points

    @staticmethod
    def reads_pick_info(hyp_file_path: str):
        """
        Reads an hyp file and returns the Obspy Origin.
        :param hyp_file_path: The file path to the .hyp file
        :return: list Pick info
        """
        if os.path.isfile(hyp_file_path):
            cat = read_nll_performance.read_nlloc_hyp_ISP(hyp_file_path)
            return cat[0]["origins"][0]["arrivals"]


    @staticmethod
    def has_same_sample_rate(st: Stream, value):
        for tr in st:
            print(tr.stats.sampling_rate)
            if tr.stats.sampling_rate != value:
                return False
        return True



class MseedUtil:

    def __init__(self, robust=True, **kwargs):

        self.start = kwargs.pop('starttime', [])
        self.end = kwargs.pop('endtime', [])
        self.obsfiles = []
        self.pos_file = []
        self.robust = robust
        self.use_ind_files = False

    @classmethod
    def get_mseed_files(cls, root_dir: str):

         """
         Get a list of valid mseed files inside the root_dir. If root_dir doesn't exists it returns a empty list.
         :param root_dir: The full path of the dir or a file.
         :return: A list of full path of mseed files.
         """

         if cls.is_valid_mseed(root_dir):
             return [root_dir]
         elif os.path.isdir(root_dir):
             files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if
                      cls.is_valid_mseed(os.path.join(root_dir, file))]
             files.sort()
             return files

         return []


    def get_tree_mseed_files(self, root_dir: str):

        """
        Get a list of valid mseed files inside all folder tree from the the root_dir.
        If root_dir doesn't exists it returns a empty list.
        :param root_dir: The full path of the dir or a file.
        :return: A list of full path of mseed files.
        """

        for top_dir, sub_dir, files in os.walk(root_dir):
            for file in files:
                self.pos_file.append(os.path.join(top_dir, file))

        with Pool(processes=os.cpu_count()) as pool:
            r =  pool.map(self.loop_tree, range(len(self.pos_file)))

        r = list(filter(None, r))
        r.sort()

        return r

    # @staticmethod
    # def get_project_basic_info(project):
    #
    #     try:
    #         total_components = sum(len(value_list) for value_list in project.values())
    #         stations_channel = len(project)
    #     except:
    #         total_components = None
    #         stations_channel = None
    #
    #     return stations_channel, total_components
    @staticmethod
    def get_project_basic_info(project: dict) -> dict:

        """
        Counts the number of unique stations in a dictionary where keys are in the format 'NET.STATION.CHANNEL'.

        Args:
            data_dict (dict): The input dictionary with keys in the format 'NET.STATION.CHANNEL'.

        Returns:
            int: The number of unique stations and channels
        """

        ## Take the stations names and its number

        info = {}

        networks = set()
        stations = set()  # Use a set to store unique stations
        channels = set()
        num_stations = 0
        num_channels = 0
        num_networks = 0
        start_time = []
        end_time = []
        total_components = 0

        for key in project.keys():
            parts = key.split('.')  # Split the key by '.'
            network = f"{parts[0]}"
            networks.add(network)  # Add to the set
            station = f"{parts[1]}"
            stations.add(station)
            channel = f"{parts[2]}"
            channels.add(channel)
            for item in project[key]:
                start_time.append(item[1].starttime)
                end_time.append(item[1].endtime)

        if len(stations) > 0:
            num_stations = len(stations)
            num_channels = len(channels)
            num_networks = len(networks)


        ## Take the number of files

        if len(stations) > 0:
            total_components = sum(len(value_list) for value_list in project.values())

        if len(stations) > 0:
            info["Networks"] = [networks, num_networks]
            info["Stations"] = [stations, num_stations]
            info["Channels"] = [channels, num_channels]
            info["num_files"] = total_components
            info["Start"] = min(start_time).strftime(format="%Y-%m-%d %H:%M:%S")
            info["End"] = max(end_time).strftime(format="%Y-%m-%d %H:%M:%S")

        return info

    def loop_tree(self, i):
        result = None
        if isinstance(self.start, UTCDateTime):
            try:
                header = read(self.pos_file[i], headlonly=True)
                #check times as a filter
                st0 = header[0].stats.starttime
                st1 = self.start
                et0 = header[0].stats.endtime
                et1 = self.end
                if st1>=st0 and et1>et0 and (st1-st0) <= 86400:
                    result = self.pos_file[i]
                elif st1<=st0 and et1>=et0:
                    result = self.pos_file[i]
                elif st1<=st0 and et1<=et0 and (et0-et1) <= 86400:
                    result = self.pos_file[i]
                elif st1 >= st0 and et1 <= et0:
                    result = self.pos_file[i]
                else:
                    pass
            except:
                pass

        else:

            if self.robust and self.is_valid_mseed(self.pos_file[i]):

                result = self.pos_file[i]

            elif not self.robust:

                result = self.pos_file[i]


        return result

    ####### New Project ###################

    @classmethod
    def load_project(cls, file: str):
        project = {}
        try:
            project = pickle.load(open(file, "rb"))
            if isinstance(project, SurfProject):
                project = project.project
        except:
            pass
        return project

    def search_indiv_files(self, list_files: list):

        self.use_ind_files = True
        self.list_files = list_files
        with Pool(processes=os.cpu_count()) as pool:
            returned_list = pool.map(self.create_dict, range(len(self.list_files)))

        project = self.convert2dict(returned_list)
        self.use_ind_files = False

        return project

    def search_files(self, rooth_path: str):

        self.search_file = []
        for top_dir, sub_dir, files in os.walk(rooth_path):
            for file in files:
                self.search_file.append(os.path.join(top_dir, file))

        with Pool(processes=os.cpu_count()) as pool:
            returned_list = pool.map(self.create_dict, range(len(self.search_file)))

        project = self.convert2dict(returned_list)

        return project

    def create_dict(self, i):
        key = None
        data_map = None

        try:
            if self.use_ind_files:
                header = read(self.list_files[i], headeronly=True)
                print(self.list_files[i])
                net = header[0].stats.network
                sta = header[0].stats.station
                chn = header[0].stats.channel
                key = net + "." + sta + "." + chn
                data_map = [self.list_files[i], header[0].stats]
            else:
                header = read(self.search_file[i], headeronly=True)
                print(self.search_file[i])
                net = header[0].stats.network
                sta = header[0].stats.station
                chn = header[0].stats.channel
                key = net + "." + sta + "." + chn
                data_map = [self.search_file[i], header[0].stats]

        except:
            pass


        return [key, data_map]


    def estimate_size(self, rooth_path):

        nbytes = sum(file.stat().st_size for file in Path(rooth_path).rglob('*')) * 1E-6

        return nbytes

    def convert2dict(self, project):
        project_converted = {}
        for name in project:
            if name[0] in project_converted.keys() and name[0] is not None:
                project_converted[name[0]].append([name[1][0],name[1][1]])

            elif name[0] not in project_converted.keys() and name[0] is not None:
                project_converted[name[0]] = [[name[1][0],name[1][1]]]

        return project_converted

    @staticmethod
    def search(project, event):
        res = {}
        for key in project.keys():
            name_list = key.split('.')
            net = name_list[0]
            sta = name_list[1]
            channel = name_list[2]
            if re.search(event[0], net) and re.search(event[1], sta) and re.search(event[2], channel):
                res[key] = project[key]

        return res

    @classmethod
    def filter_project_keys(cls, project, **kwargs):

        # filter dict by python wilcards remind

        # * --> .+
        # ? --> .

        net = kwargs.pop('net', '.+')
        station = kwargs.pop('station', '.+')
        channel = kwargs.pop('channel', '.+')
        if net == '':
            net = '.+'
        if station == '':
            station = '.+'
        if channel == '':
            channel = '.+'

        # filter for regular expresions
        event = [net, station, channel]
        project = cls.search(project, event)

        data_files = []
        project = {key: value for key, value in project.items() if value}
        for item in project.items():
            list_channel = item[1]
            for file_path in list_channel:
                data_files.append(file_path[0])


        return project, data_files

    @classmethod
    def filter_time(cls, project, starttime, endtime, tol=86400):

        """
        Filters project data based on time range.

        - starttime (str or UTCDateTime): Start time of the range.
          If str, it should follow the format "%Y-%m-%d %H:%M:%S".
          Example: "2023-12-10 00:00:00"
        - endtime (str or UTCDateTime): End time of the range.
          If str, it should follow the format "%Y-%m-%d %H:%M:%S".
          Example: "2023-12-23 00:00:00"
        """

        # Convert starttime and endtime to datetime objects if they are strings
        date_format = "%Y-%m-%d %H:%M:%S"
        exact = False

        if isinstance(starttime, str):
            start = datetime.strptime(starttime, date_format)
            start = UTCDateTime(start)
        elif isinstance(starttime, UTCDateTime):
            start = starttime
        else:
            raise TypeError("starttime must be a string or UTCDateTime object.")

        if isinstance(endtime, str):
            end = datetime.strptime(endtime, date_format)
            end = UTCDateTime(end)
        elif isinstance(endtime, UTCDateTime):
            end = endtime
        else:
            raise TypeError("endtime must be a string or UTCDateTime object.")


        # to check we are chopping the same day
        start_date_exact = str(starttime.year) + str(starttime.julday)
        end_date_exact = str(endtime.year) + str(endtime.julday)

        if (end-start) <= 86400 and start_date_exact == end_date_exact:
            exact = True

        # Process project data
        if len(project) > 0:
            for key in project:
                item = project[key]
                indices_to_remove = []

                if exact:
                    for index, value in enumerate(item):
                        start_data = value[1].starttime
                        end_data = value[1].endtime

                        if start_data <= start and end <= end_data:
                            pass
                        elif start_data >= start and end_data <= end:
                            pass

                        else:
                            indices_to_remove.append(index)

                else:
                    for index, value in enumerate(item):
                        start_data = value[1].starttime
                        end_data = value[1].endtime
                        if start_data >= start and end_data > end and (start_data - start) <= tol:
                            pass

                        elif start_data <= start and end_data >= end:
                            pass

                        elif start_data <= start and end_data <= end and (end - end_data) <= tol:
                            pass

                        elif start_data >= start and end_data <= end:
                            pass

                        else:
                            indices_to_remove.append(index)

                for index in reversed(indices_to_remove):
                    item.pop(index)
                    project[key] = item

        # Fill the data_files list
        data_files = []
        project = {key: value for key, value in project.items() if value}
        for item in project.items():
            list_channel = item[1]
            for file_path in list_channel:
                data_files.append(file_path[0])

        return project, data_files

    @staticmethod
    def generate_data_files(project):
        data_files = []
        project = {key: value for key, value in project.items() if value}
        for item in project.items():
            list_channel = item[1]
            for file_path in list_channel:
                data_files.append(file_path[0])

        return data_files



    # old style filter time
    # @classmethod
    # def filter_time(cls, list_files, **kwargs):
    #
    #     #filter the list output of filter_project_keys by trimed times
    #
    #     result = []
    #     st1 = kwargs.pop('starttime', None)
    #     et1 = kwargs.pop('endtime', None)
    #
    #     if st1 is None and et1 is None:
    #         for file in list_files:
    #             result.append(file[0])
    #
    #     else:
    #
    #         for file in list_files:
    #             pos_file = file[0]
    #             st0 = file[1]
    #             et0 = file[2]
    #             # check times as a filter
    #
    #             if st1 >= st0 and et1 > et0 and (st1 - st0) <= 86400:
    #                 result.append(pos_file)
    #             elif st1 <= st0 and et1 >= et0:
    #                 result.append(pos_file)
    #             elif st1 <= st0 and et1 <= et0 and (et0 - et1) <= 86400:
    #                 result.append(pos_file)
    #             elif st1 >= st0 and et1 <= et0:
    #                 result.append(pos_file)
    #             else:
    #                 pass
    #
    #     result.sort()
    #
    #     return result


    ###### New Project ###########

    def get_tree_hd5_files(self, root_dir: str, robust=True, **kwargs):

        self.start = kwargs.pop('starttime', [])
        self.end = kwargs.pop('endtime', [])
        self.robust = robust
        pos_file = []

        for top_dir, sub_dir, files in os.walk(root_dir):
            for file in files:
                try:
                    header = read(os.path.join(top_dir, file), headlonly=True)
                    pos_file.append(os.path.join(top_dir, file))
                except:
                    pass

        return pos_file


    # @classmethod
    # def get_tree_hd5_files(cls, root_dir: str, robust=True, **kwargs):
    #     """
    #     Get a list of valid mseed files inside all folder tree from the the root_dir.
    #     If root_dir doesn't exists it returns a empty list.
    #     :param root_dir: The full path of the dir or a file.
    #     :return: A list of full path of mseed files.
    #     """
    #     cls.start = kwargs.pop('starttime', [])
    #     cls.end = kwargs.pop('endtime', [])
    #     cls.obsfiles = []
    #     cls.pos_file = []
    #     cls.robust = robust
    #
    #     for top_dir, sub_dir, files in os.walk(root_dir):
    #         for file in files:
    #             cls.pos_file.append(os.path.join(top_dir, file))
    #
    #     with Pool(processes=6) as pool:
    #         r = pool.map(cls.loop_tree_h5, range(len(cls.pos_file)))
    #
    #     r = list(filter(None, r))
    #     r.sort()
    #
    #     return r

    # @classmethod
    # def loop_tree_h5(cls, i):
    #     result = None
    #     try:
    #         header = read(cls.pos_file[i], headlonly=True)
    #         result = cls.pos_file[i]
    #     except:
    #         pass
    #
    #     return result

    @classmethod
    def get_geodetic(cls,file):
        dist  = None
        bazim = None
        azim = None
        geodetic = [dist, bazim, azim]

        try:
            st = read(file)
            geodetic = st[0].stats.mseed['geodetic']
        except:
            pass

        return geodetic

    @classmethod
    def get_selected_files(cls, files, selection):
        new_list = []
        for file in files:
            st = read(file, headonly=True)
            if len(selection[0]) >0 and len(selection[1]) > 0 and len(selection[2]):
                if st.select(network = selection[0], station = selection[1], channel = selection[2]):
                    new_list.append(file)
            if len(selection[0]) > 0 and len(selection[1]) == 0 and len(selection[2]) == 0:
                if st.select(network=selection[0]):
                    new_list.append(file)
            if len(selection[0]) > 0 and len(selection[1]) > 0 and len(selection[2]) == 0:
                if st.select(network=selection[0], station = selection[1]):
                    new_list.append(file)
        return new_list


    @staticmethod
    def is_valid_mseed(file_path):
        """
        Return True if path is an existing regular file and a valid mseed. False otherwise.
        :param file_path: The full file's path.
        :return: True if path is an existing regular file and a valid mseed. False otherwise.
        """
        #if os.path.isfile(file_path) and _is_mseed(file_path):
        #   return True
        #elif os.path.isfile(file_path) and _is_sac(file_path):
        #   return True
        return os.path.isfile(file_path) and _is_mseed(file_path)


    @staticmethod
    def is_valid_dataless(file_path):
        """
        Check if is a valid dataless file.
        :param file_path: The full file's path.
        :return: True if path is a valid dataless. False otherwise.
        """
        parser = Parser()
        try:
            parser.read(file_path)
            return True
        except IOError:
            return False

    @classmethod
    def get_metadata_files(cls, file):
        from obspy import read_inventory
        try:

            inv = read_inventory(file)

            return inv

        except IOError:

           return []

    @classmethod
    def get_dataless_files(cls, root_dir):
        """
        Get a list of valid dataless files inside the root_dir. If root_dir doesn't exists it returns a empty list.
        :param root_dir: The full path of the dir or a file.
        :return: A list of full path of dataless files.
        """

        #if os.path.isfile(root_dir) and cls.is_valid_dataless(root_dir):

        if os.path.isfile(root_dir) and _is_stationxml(root_dir) and cls.is_valid_dataless(root_dir):
        #if os.path.isfile(root_dir):
            return [root_dir]
        elif os.path.isdir(root_dir):
            files = []
            for file in os.listdir(root_dir):
                if _is_stationxml(os.path.join(root_dir, file)):
                    files.append(file)
                if cls.is_valid_dataless(root_dir):
                    files.append(file)

            #files = [os.path.join(root_dir, file) for file in os.listdir(root_dir)
            #         if os.path.isfile(os.path.join(root_dir, file)) and os.path.isfile(os.path.join(root_dir, file))
            #         != ".DS_Store" and cls.is_valid_dataless(os.path.join(root_dir, file)) and _is_stationxml(os.path.join(root_dir, file))]
            files.sort()
            return files
        return []

    @classmethod
    def get_xml_files(cls, root_dir: str):
        """
        Get a list of valid dataless files inside the root_dir. If root_dir doesn't exists it returns a empty list.
        :param root_dir: The full path of the dir or a file.
        :return: A list of full path of dataless files.
        """

        if os.path.isfile(root_dir):
            return [root_dir]
        elif os.path.isdir(root_dir):
            files = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
            files.sort()
            return files
        return []

    @classmethod
    def data_availability_new(cls, list_files: list):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdt
        from isp.Gui.Frames import MatplotlibFrame

        fig, hax = plt.subplots(1, 1, figsize=(12, 6))
        cls.mpf = MatplotlibFrame(fig)
        starttimes = []
        endtimes = []


        data_map = {}
        data_map['nets'] = {}

        for i in list_files:

            print("Processing Waveform at ", i)
            header = read(i, headlonly=True)
            gap = header.get_gaps()
            net = header[0].stats.network
            sta = header[0].stats.station
            chn = header[0].stats.channel
            # times
            starttimes.append(header[0].stats.starttime)
            start = header[0].stats.starttime.matplotlib_date
            endtimes.append(header[0].stats.endtime)
            end = header[0].stats.endtime.matplotlib_date
            name = net + "." + sta + "." + chn
            hax.hlines(name, start, end, colors='k', linestyles='solid', label=name, lw=2)
            if len(gap) > 0:
                for i in range(len(gap)):
                    starttime_gap = gap[i][4].matplotlib_date
                    endtime_gap = gap[i][5].matplotlib_date
                    hax.hlines(name, starttime_gap, endtime_gap, colors='r', linestyles='solid', label=name, lw=2)

        start_time = min(starttimes)
        end_time = max(endtimes)
        formatter = mdt.DateFormatter('%y/%m/%d/%H:%M:%S.%f')
        hax.xaxis.set_major_formatter(formatter)
        hax.set_xlabel("Date")
        hax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)
        cls.mpf.show()


    @classmethod
    def cluster_events(cls, times, eps=20.0):
        from obspy import UTCDateTime
        points = []
        for j in range(len(times)):
            points.append(times[j].timestamp)

        clusters = []
        points_sorted = sorted(points)
        curr_point = points_sorted[0]
        curr_cluster = [curr_point]
        for point in points_sorted[1:]:
            if point <= curr_point + eps:
                curr_cluster.append(point)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [point]
            curr_point = point
        clusters.append(curr_cluster)
        new_times = []
        string_times = []
        for k  in  range(len(clusters)):
            new_times.append(UTCDateTime(clusters[k][0]))
            string_times.append(UTCDateTime(clusters[k][0]).strftime(format="%Y-%m-%dT%H:%M:%S.%f"))
        return new_times, string_times

    @classmethod
    def get_NLL_phase_picks_phase(cls, phase=None, **kwargs ):

        pick_times = {}
        pick_file = os.path.join(PICKING_DIR, "output.txt")
        pick_file = kwargs.pop("input_file", pick_file)

        if os.path.isfile(pick_file):
            df = pd.read_csv(pick_file, delimiter='\s+')
            for index, row in df.iterrows():
                tt = str(row['Date']) + "TT" + str(row['Hour_min']) + '{:0>2}'.format(row['Seconds'])
                if phase == row["P_phase_descriptor"]:
                    pick_times[row['Station_name'] + "." + row["Component"]] = [row["P_phase_descriptor"], UTCDateTime(tt)]
                elif phase is None:
                    pick_times[row['Station_name'] + "." + row["Component"]] = [row["P_phase_descriptor"],
                                                                                UTCDateTime(tt)]
            return pick_times

    @classmethod
    def get_NLL_phase_picks(cls, input_file=None, delimiter='\s+'):
        """
        Reads a NonLinLoc output file and returns a dictionary of phase picks.

        Parameters:
            input_file (str, optional): Path to the NonLinLoc output file.
                                        If not provided, raises an error.
            delimiter (str, optional): Delimiter used in the file (default is a space).
            **kwargs: Additional arguments for customization.

        Returns:
            dict: Dictionary of picks with the structure:
              {
                  "Station.Component": [
                      ["Station_name", "Instrument", "Component", "P_phase_onset", "P_phase_descriptor",
                            "First_Motion", "Date", "Hour_min", "Seconds", "Err", "ErrMag", "Coda_duration",
                            "Amplitude", "Period"
                  ],
                  ...
              }
        """
        if input_file is None:
            raise ValueError("An input file must be provided.")

        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"The file {input_file} does not exist.")

        pick_times = {}

        try:
            # Load the file into a DataFrame
            df = pd.read_csv(input_file, delimiter=delimiter)

            # Validate necessary columns

            required_columns = ["Station_name", "Instrument", "Component", "P_phase_onset", "P_phase_descriptor",
                                "First_Motion", "Date", "Hour_min", "Seconds", "Err", "ErrMag", "Coda_duration",
                                "Amplitude", "Period"]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"The input file is missing required columns: {missing_columns}")

            # Process each row
            for _, row in df.iterrows():
                try:
                    # Construct timestamp
                    tt = f"{row['Date']}T{str(row['Hour_min']).zfill(4)}:{float(row['Seconds']):06.3f}"
                    timestamp = UTCDateTime(tt)

                    # Construct ID
                    id = f"{row['Station_name']}.{row['Component']}"

                    # Collect pick details
                    pick_details = [
                        row["P_phase_descriptor"], timestamp, row["Component"],
                        row["First_Motion"], row["Err"], row["ErrMag"],
                        row["Coda_duration"], row["Amplitude"], row["Period"]
                    ]

                    # Add to dictionary
                    if id not in pick_times:
                        pick_times[id] = []
                    pick_times[id].append(pick_details)

                except Exception as e:
                    print(f"Error processing row {row.to_dict()}: {e}")

        except Exception as e:
            raise RuntimeError(f"Error reading or processing the file {input_file}: {e}")

        return pick_times

    @staticmethod
    def get_stream(files_path: str, selection: dict):

        traces = []

        # List all files in the folder with full paths
        files_in_directory = [os.path.join(files_path, f) for f in os.listdir(files_path) if
                 os.path.isfile(os.path.join(files_path, f))]

        for file in files_in_directory:
            try:
                tr = read(file)[0]
                traces.append(tr)
            except:
                print(file, " not accepted as valid seismogram file")

        # selection
        if len(traces) > 0:
            stream = Stream(traces)
            stream_selected = stream.select(network=selection["network"], station=selection["station"], location=None,
                          channel=selection["channel"], sampling_rate=None, npts=None,
                          component=None, id=None, inventory=None)
        else:
            stream_selected = []

        return stream_selected

    @staticmethod
    def filter_inventory_by_stream(stream: Stream, inventory: Inventory) -> Inventory:

        # Create an empty list to hold filtered networks
        filtered_networks = []

        # Loop through networks in the inventory
        for network in inventory:
            # Create a list to hold filtered stations for each network
            filtered_stations = []

            # Loop through stations in the network
            for station in network:
                # Find channels in this station that match the stream traces
                filtered_channels = []

                # Check if any trace in the stream matches the station and network
                for trace in stream:
                    # Extract network, station, location, and channel codes from trace
                    trace_net, trace_sta, trace_loc, trace_chan = trace.id.split(".")

                    # Check if the current station and network match the trace
                    if station.code == trace_sta and network.code == trace_net:
                        # Look for a channel in the station that matches the trace's channel code
                        for channel in station.channels:
                            if channel.code == trace_chan and (not trace_loc or channel.location_code == trace_loc):
                                filtered_channels.append(channel)

                # If there are any matching channels, create a filtered station
                if filtered_channels:
                    filtered_station = Station(
                        code=station.code,
                        latitude=station.latitude,
                        longitude=station.longitude,
                        elevation=station.elevation,
                        creation_date=station.creation_date,
                        site=station.site,
                        channels=filtered_channels
                    )
                    filtered_stations.append(filtered_station)

            # If there are any matching stations, create a filtered network
            if filtered_stations:
                filtered_network = Network(
                    code=network.code,
                    stations=filtered_stations
                )
                filtered_networks.append(filtered_network)

        # Create a new inventory with the filtered networks
        filtered_inventory = Inventory(networks=filtered_networks, source=inventory.source)
        return filtered_inventory

    @staticmethod
    def check_stream_in_inventory(stream: Stream, inventory: Inventory) -> list:
        """
        Checks if any trace in the stream is not present in the inventory.

        Parameters:
            stream (Stream): The input ObsPy Stream to check.
            inventory (Inventory): The input ObsPy Inventory to match against.

        Returns:
            list: A list of trace IDs (net.sta.loc.chan) that are not found in the inventory.
        """
        unmatched_traces = []

        # Loop through each trace in the stream
        for trace in stream:
            trace_net, trace_sta, trace_loc, trace_chan = trace.id.split(".")
            found = False

            # Loop through networks, stations, and channels in the inventory
            for network in inventory:
                if network.code == trace_net:
                    for station in network.stations:
                        if station.code == trace_sta:
                            for channel in station.channels:
                                # Match channel code and location
                                if channel.code == trace_chan and (not trace_loc or channel.location_code == trace_loc):
                                    found = True
                                    break
                        if found:
                            break
                if found:
                    break

            # If the trace was not found, add it to the unmatched list
            if not found:
                unmatched_traces.append(trace.id)

        if len(unmatched_traces) == 0:
            return True
        else:
            print("Not matches with inventory")
            print(unmatched_traces)
            return False
