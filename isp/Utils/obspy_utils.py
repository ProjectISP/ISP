import os
from enum import unique, Enum

from obspy import Stream, read, Trace
# noinspection PyProtectedMember
from obspy.io.mseed.core import _is_mseed
from obspy.io.xseed.parser import Parser

from isp.Structures.structures import TracerStats


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
    def filter_trace(trace, trace_filter, f_min, f_max):
        """
        Filter a obspy Trace or Stream.

        :param trace: The trace or stream to be filter.
        :param trace_filter: The filter name or Filter enum, ie. Filter.BandPass or "bandpass".
        :param f_min: The lower frequency.
        :param f_max: The higher frequency.
        :return: False if bad frequency filter, True otherwise.
        """
        if trace_filter != Filters.Default:
            if not (f_max - f_min) > 0:
                print("Bad filter frequencies")
                return False

            trace.taper(max_percentage=0.05, type="blackman")

            if trace_filter == Filters.BandPass or trace_filter == Filters.BandStop:
                trace.filter(trace_filter, freqmin=f_min, freqmax=f_max, corners=4, zerophase=True)

            elif trace_filter == Filters.HighPass:
                trace.filter(trace_filter, freq=f_min, corners=4, zerophase=True)

            elif trace_filter == Filters.LowPass:
                trace.filter(trace_filter, freq=f_max, corners=4, zerophase=True)

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

    @staticmethod
    def is_valid_mseed(file_path):
        """
        Return True if path is an existing regular file and a valid mseed. False otherwise.

        :param file_path: The full file's path.
        :return: True if path is an existing regular file and a valid mseed. False otherwise.
        """
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

        if os.path.isfile(root_dir) and cls.is_valid_dataless(root_dir):
            return [root_dir]
        elif os.path.isdir(root_dir):
            files = [os.path.join(root_dir, file) for file in os.listdir(root_dir)
                     if os.path.isfile(os.path.join(root_dir, file)) and
                     cls.is_valid_dataless(os.path.join(root_dir, file))]
            files.sort()
            return files
        return []
