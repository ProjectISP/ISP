from obspy import read
from obspy import UTCDateTime
import numpy as np
from isp.Utils import ObspyUtil, Filters
from obspy.signal.polarization import polarization_analysis

class PolarizationAnalyis:

    def __init__(self, path_z, path_n, path_e):
        """
        Manage nll files for run nll program.

        Important: The  obs_file_path is provide by the class :class:`PickerManager`.

        :param obs_file_path: The file path of pick observations.
        """
        self.path_z = path_z
        self.path_n = path_n
        self.path_e = path_e


    def rotate(self, t1, t2, method="NE->RT", angle=0, **kwargs):

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

         if maxstart - t1 < 0 < minend - t2:
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
             ObspyUtil.filter_trace(tr, filter_value, f_min, f_max)
             data.append(tr.data)

         return time, data[0], data[1], data[2], st

    def polarize(self,t1,t2, win_len, win_frac, frqlow, frqhigh, method='flinn'):

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
        planarity = out["planarity"]
        rectilinearity = out["rectilinearity"]
        #time=np.arange(0,len(azimuth))
        variables = {'time': time, 'azimuth': azimuth, 'incident_angle': incident_angle, 'planarity': planarity,
                     'rectilinearity': rectilinearity}

        return variables

