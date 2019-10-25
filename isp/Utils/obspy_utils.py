import os

from obspy import Stream, read
# noinspection PyProtectedMember
from obspy.io.mseed.core import _is_mseed


class ObspyUtil:

    @staticmethod
    def get_figure_from_stream(st: Stream, **kwargs):
        if st:
            return st.plot(show=False, **kwargs)
        return None

    @staticmethod
    def get_tracer_from_file(file_path):
        st = read(file_path)
        return st[0]


class MseedUtil:

    @staticmethod
    def get_mseed_files(root_dir: str):
        """
        Get a list of valid mseed files inside the root_dir. If root_dir doesn't exists it returns a empty list.

        :param root_dir: The full path of the dir or a file.

        :return: A list of full path of mseed files.
        """

        if not os.path.exists(root_dir):
            return []

        if os.path.isfile(root_dir) and _is_mseed(root_dir):
            return [root_dir]
        elif os.path.isdir(root_dir):
            files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if
                     os.path.isfile(os.path.join(root_dir, file)) and _is_mseed(os.path.join(root_dir, file))]
            files.sort()
            return files

    @staticmethod
    def is_valid_mseed(file_path):
        return os.path.isfile(file_path) and _is_mseed(file_path)
