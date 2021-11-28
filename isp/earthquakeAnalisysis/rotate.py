import numpy as np
from obspy import Stream
from obspy import UTCDateTime
from obspy.signal.polarization import polarization_analysis

from isp.DataProcessing import SeismogramDataAdvanced
from isp.Gui.Frames import MessageDialog
from isp.Utils import ObspyUtil, Filters


class PolarizationAnalyis:

    def __init__(self, path_z, path_n, path_e):
        """
        Manage nll files for run nll program.

        Important: The  obs_file_path is provide by the class :class:`PickerManager`.

        :param obs_file_path: The file path of pick observations.
        """
        self.start_time = None
        self.end_time = None
        self.path_z = path_z
        self.path_n = path_n
        self.path_e = path_e
        self.list_path = [path_z, path_n, path_e]

    def __get_stream(self, start_time: UTCDateTime, end_time: UTCDateTime) -> Stream:

        st = ObspyUtil.merge_files_to_stream([self.path_z, self.path_n, self.path_e])
        ObspyUtil.trim_stream(st, start_time, end_time)
        self.start_time = st[0].stats.starttime
        self.end_time = st[0].stats.endtime
        return st

    def filter_error_message(self, msg):
        md = MessageDialog(self)
        md.set_info_message(msg)

    def rotate(self, inventory, t1: UTCDateTime, t2: UTCDateTime, angle, incidence_angle, method="NE->RT", **kwargs):
        all_traces = []
        self.__get_stream(t1, t2)
        self.t1 = t1
        self.t2 = t2
        # Automatic trim to the starttime and endtime
        if t1 < self.start_time:
            self.t1 = self.start_time
        if t2 > self.end_time:
            self.t2 = self.end_time
        # Process advance
        parameters = kwargs.get("parameters")
        trim = kwargs.get("trim")

        for index, file_path in enumerate(self.list_path):

            sd = SeismogramDataAdvanced(file_path)

            if trim:
                tr = sd.get_waveform_advanced(parameters, inventory, filter_error_callback=self.filter_error_message,
                                              start_time=self.t1, end_time=self.t2)
            else:
                tr = sd.get_waveform_advanced(parameters, inventory, filter_error_callback=self.filter_error_message)

            all_traces.append(tr)

            st = Stream(traces=all_traces)

        #
        #sampling_rate = st[0].stats.sampling_rate
        # time = np.arange(0, len(st[0].data) / sampling_rate, 1. / sampling_rate)
        time = st[0].times("matplotlib")
        # rotate
        if method == "NE->RT":
            st.rotate(method=method, back_azimuth=angle)
        elif method == 'ZNE->LQT':
            st.rotate(method=method, back_azimuth=angle, inclination=incidence_angle)

        n = len(st)
        data = []
        for i in range(n):
            tr = st[i]
            data.append(tr.data)

        return time, data[0], data[1], data[2], st

    def polarize(self, t1: UTCDateTime, t2: UTCDateTime, win_len, frqlow, frqhigh, method='flinn'):

       # win_frac=int(win_len*win_frac/100)
        st = self.__get_stream(t1, t2)
        fs=st[0].stats.sampling_rate
        win_frac=1/(fs*win_len)

        out = polarization_analysis(st, win_len, win_frac, frqlow, frqhigh, st[0].stats.starttime,
                                    st[0].stats.endtime, verbose=False, method=method, var_noise=0.0)

        time = out["timestamp"]
        azimuth = out["azimuth"] + 180
        incident_angle = out["incidence"]
        planarity = out["planarity"]
        rectilinearity = out["rectilinearity"]
        #time=np.arange(0,len(azimuth))
        variables = {'time': time, 'azimuth': azimuth, 'incident_angle': incident_angle, 'planarity': planarity,
                     'rectilinearity': rectilinearity}

        return variables

