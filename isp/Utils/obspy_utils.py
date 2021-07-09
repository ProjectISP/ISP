import math
import os
from enum import unique, Enum
from os import listdir
from os.path import isfile, join
from typing import List

import numpy as np
from obspy import Stream, read, Trace, UTCDateTime, read_events
# noinspection PyProtectedMember
from obspy.core.event import Origin
from obspy.geodetics import gps2dist_azimuth
from obspy.io.mseed.core import _is_mseed
#from obspy.io.sac.core import _is_sac
from obspy.io.xseed.parser import Parser

from isp.Exceptions import InvalidFile
from isp.Structures.structures import TracerStats
from isp.Utils.nllOrgErrors import computeOriginErrors


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
    def filter_trace(trace, trace_filter, f_min, f_max, **kwargs):
        """
        Filter a obspy Trace or Stream.

        :param trace: The trace or stream to be filter.
        :param trace_filter: The filter name or Filter enum, ie. Filter.BandPass or "bandpass".
        :param f_min: The lower frequency.
        :param f_max: The higher frequency.

        :keyword kwargs:
        :keyword corners: The number of poles, default = 4.
        :keyword zerophase: True for keep the phase without shift, false otherwise, Default = True.

        :return: False if bad frequency filter, True otherwise.
        """
        if trace_filter != Filters.Default:
            if not (f_max - f_min) > 0:
                print("Bad filter frequencies")
                return False

            corners = kwargs.pop("corners", 4)
            zerophase = kwargs.pop("zerophase", True)

            trace.taper(max_percentage=0.05, type="blackman")

            if trace_filter == Filters.BandPass or trace_filter == Filters.BandStop:
                trace.filter(trace_filter, freqmin=f_min, freqmax=f_max, corners=corners, zerophase=zerophase)

            elif trace_filter == Filters.HighPass:
                trace.filter(trace_filter, freq=f_min, corners=corners, zerophase=zerophase)

            elif trace_filter == Filters.LowPass:
                trace.filter(trace_filter, freq=f_max, corners=corners, zerophase=zerophase)

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
    def reads_hyp_to_origin(hyp_file_path: str) -> Origin:
        """
        Reads an hyp file and returns the Obspy Origin.

        :param hyp_file_path: The file path to the .hyp file

        :return: An Obspy Origin
        """

        if os.path.isfile(hyp_file_path):
            cat = read_events(hyp_file_path)
            event = cat[0]
            origin = event.origins[0]
            modified_origin_90 = computeOriginErrors(origin)
            origin.depth_errors["uncertainty"]=modified_origin_90['depth_errors'].uncertainty
            origin.origin_uncertainty.max_horizontal_uncertainty = modified_origin_90['origin_uncertainty'].max_horizontal_uncertainty
            origin.origin_uncertainty.min_horizontal_uncertainty = modified_origin_90[
                'origin_uncertainty'].min_horizontal_uncertainty
            origin.origin_uncertainty.azimuth_max_horizontal_uncertainty = modified_origin_90['origin_uncertainty'].azimuth_max_horizontal_uncertainty



            return origin
        else:
            raise FileNotFoundError("The file {} doesn't exist. Please, run location".format(hyp_file_path))

    @staticmethod
    def has_same_sample_rate(st: Stream, value):
        for tr in st:
            print(tr.stats.sampling_rate)
            if tr.stats.sampling_rate != value:
                return False
        return True


class MseedUtil:

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

    @classmethod
    def get_tree_mseed_files(cls, root_dir: str, **kwargs):
        """
        Get a list of valid mseed files inside all folder tree from the the root_dir.
        If root_dir doesn't exists it returns a empty list.
        :param root_dir: The full path of the dir or a file.
        :return: A list of full path of mseed files.
        """
        start = kwargs.pop('starttime', [])
        end = kwargs.pop('endtime', [])
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(root_dir):
            for file in files:
                pos_file = os.path.join(top_dir, file)
                if cls.is_valid_mseed(pos_file):

                    if isinstance(start, UTCDateTime):
                        header = read(pos_file, headlonly=True)
                        #check times as a filter
                        check1 = start-header[0].stats.starttime >= 0
                        check2 = end-header[0].stats.endtime <= 0
                        if check1 and check2:
                            obsfiles.append(os.path.join(top_dir, pos_file))
                        else:
                            pass

                    else:

                        obsfiles.append(os.path.join(top_dir, pos_file))


        obsfiles.sort()

        return obsfiles



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
    def get_dataless_files(cls, root_dir: str):
        """
        Get a list of valid dataless files inside the root_dir. If root_dir doesn't exists it returns a empty list.

        :param root_dir: The full path of the dir or a file.

        :return: A list of full path of dataless files.
        """

        #if os.path.isfile(root_dir) and cls.is_valid_dataless(root_dir):
        if os.path.isfile(root_dir):
            return [root_dir]
        elif os.path.isdir(root_dir):
            files = [os.path.join(root_dir, file) for file in os.listdir(root_dir)
                     if os.path.isfile(os.path.join(root_dir, file))]
                     #and cls.is_valid_dataless(os.path.join(root_dir, file))]
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
    def data_availability(cls, files_path: str,only_this = True):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdt
        from isp.Gui.Frames import MatplotlibFrame

        fig, hax = plt.subplots(1, 1, figsize=(12, 6))
        cls.mpf = MatplotlibFrame(fig)
        starttimes = []
        endtimes = []
        if only_this:

            obsfiles = [f for f in listdir(files_path) if isfile(join(files_path, f))]
            obsfiles.sort()
        else:
            obsfiles = []
            for top_dir, sub_dir, files in os.walk(files_path):
                for file in files:
                    obsfiles.append(os.path.join(top_dir, file))
            obsfiles.sort()

        data_map = {}
        data_map['nets'] = {}

        for i in obsfiles:
            paths = os.path.join(files_path, i)
            if _is_mseed(paths):

                header = read(paths, headlonly=True)
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
        return new_times,string_times
