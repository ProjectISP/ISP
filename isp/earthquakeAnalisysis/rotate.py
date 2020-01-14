from obspy import read
from obspy import UTCDateTime
import numpy as np
from isp.Utils import ObspyUtil, Filters
from obspy.signal.polarization import polarization_analysis

class rotate:

    def __init__(self, path_z, path_n, path_e):
        """
        Manage nll files for run nll program.

        Important: The  obs_file_path is provide by the class :class:`PickerManager`.

        :param obs_file_path: The file path of pick observations.
        """
        self.path_z = path_z
        self.path_n = path_n
        self.path_e = path_e

    @staticmethod
    def filter_stream(trace, trace_filter, f_min, f_max):
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

        return trace

    def rot(self, t1, t2, method="NE->RT", angle=0, **kwargs):
         t1 = UTCDateTime(t1)
         t2 = UTCDateTime(t2)
         #read seismograms
         st = read(self.path_z)
         st += read(self.path_n)
         st += read(self.path_e)
         # trim
         maxstart = np.max([tr.stats.starttime for tr in st])
         minend = np.min([tr.stats.endtime for tr in st])

         print(maxstart)
         print(minend)
         st.trim(maxstart, minend)

         if maxstart - t1 < 0 and minend - t2 > 0:
            st.clear()
            st = read(self.path_z, starttime=t1, endtime=t2)
            st += read(self.path_n, starttime=t1, endtime=t2)
            st += read(self.path_e, starttime=t1, endtime=t2)
         sampling_rate=st[0].stats.sampling_rate
         time = np.arange(0, len(st[0].data) / sampling_rate, 1. / sampling_rate)

         # rotate
         st.rotate(method=method, back_azimuth=angle)

         #filter
         filter_value = kwargs.get("filter_value", Filters.Default)
         f_min = kwargs.get("f_min", 0.)
         f_max = kwargs.get("f_max", 0.)
         n=len(st)
         data=[]
         for i in range(n):

             tr=st[i]
             trace=self.filter_stream(tr, filter_value, f_min, f_max)
             data.append(trace.data)

         return time, data[0], data[1], data[2], st

    def polarization(self,t1,t2, win_len, win_frac, frqlow, frqhigh, method='flinn'):

        win_frac=int(win_len*win_frac/100)
        t1 = UTCDateTime(t1)
        t2 = UTCDateTime(t2)
        # read seismograms
        st = read(self.path_z)
        st += read(self.path_n)
        st += read(self.path_e)
        # trim
        maxstart = np.max([tr.stats.starttime for tr in st])
        minend = np.min([tr.stats.endtime for tr in st])

        print(maxstart)
        print(minend)
        st.trim(maxstart, minend)

        if maxstart - t1 < 0 and minend - t2 > 0:
            st.clear()
            st = read(self.path_z, starttime=t1, endtime=t2)
            st += read(self.path_n, starttime=t1, endtime=t2)
            st += read(self.path_e, starttime=t1, endtime=t2)

        out = polarization_analysis(st, win_len, win_frac, frqlow, frqhigh, st[0].stats.starttime, st[0].stats.endtime, verbose=False, method=method, var_noise=0.0)

        time = out["timestamp"]
        azimuth = out["azimuth"] + 180
        incident_angle = out["incidence"]
        Planarity = out["planarity"]
        rectilinearity = out["rectilinearity"]

        return time,azimuth,incident_angle,Planarity,rectilinearity

