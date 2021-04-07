from enum import Enum, unique
from obspy import read, Stream
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from isp.Structures.structures import TracerStats
from isp.Utils import ObspyUtil, Filters
import numpy as np
from isp.seismogramInspector.signal_processing_advanced import add_white_noise, whiten, normalize, wavelet_denoise, \
    smoothing, wiener_filter


@unique
class ArrivalsModels(Enum):
    Iasp91 = "iasp91"
    Ak135 = "ak135"


class SeismogramAnalysis:

    def __init__(self, station_latitude, station_longitude):

        self.station_latitude = station_latitude
        self.station_longitude = station_longitude

    def get_phases_and_arrivals(self, event_lat, event_lon, depth):
        dist = locations2degrees(event_lat, event_lon, self.station_latitude, self.station_longitude)
        model = TauPyModel(model=ArrivalsModels.Iasp91.value)

        travel_times = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist)

        arrival = [(tt.time, tt.name) for tt in travel_times]
        arrivals, phases = zip(*arrival)

        return phases, arrivals


class SeismogramData:

    def __init__(self, file_path):
        if file_path:
            st = read(file_path)
            self.__tracer = st[0]
            self.stats = TracerStats.from_dict(self.tracer.stats)

    @classmethod
    def from_tracer(cls, tracer):
        sd = cls(None)
        sd.set_tracer(tracer)
        return sd

    @property
    def tracer(self):
        return self.__tracer

    def set_tracer(self, tracer):
        self.__tracer = tracer
        self.stats = TracerStats.from_dict(self.__tracer.stats)

    def __send_filter_error_callback(self, func, msg):
        if func:
            func(msg)

    def get_waveform(self, filter_error_callback=None, **kwargs):
        filter_value = kwargs.get("filter_value", Filters.Default)
        f_min = kwargs.get("f_min", 0.)
        f_max = kwargs.get("f_max", 0.)
        start_time = kwargs.get("start_time", self.stats.StartTime)
        end_time = kwargs.get("end_time", self.stats.EndTime)

        tr = self.tracer

        tr.detrend(type="demean")

        try:
            if not ObspyUtil.filter_trace(tr, filter_value, f_min, f_max):
                self.__send_filter_error_callback(filter_error_callback,
                                                  "Lower frequency {} must be "
                                                  "smaller than Upper frequency {}".format(f_min, f_max))
        except ValueError as e:
            print(e)
            self.__send_filter_error_callback(filter_error_callback, str(e))

        try:
            tr.trim(starttime=start_time, endtime=end_time)
        except:
            print("Please Check Starttime and Endtime")

        return tr




class SeismogramDataAdvanced:

    def __init__(self, file_path):
        if file_path:
            self.st = read(file_path)

            gaps = self.st.get_gaps()

            if len(gaps) > 0:
                self.st.print_gaps()
                self.st.merge(fill_value = "interpolate")

            self.__tracer = self.st[0]
            self.stats = TracerStats.from_dict(self.tracer.stats)


    @classmethod
    def from_tracer(cls, tracer):
        sd = cls(None)
        sd.set_tracer(tracer)
        return sd

    @property
    def tracer(self):
        return self.__tracer

    def set_tracer(self, tracer):
        self.__tracer = tracer
        self.stats = TracerStats.from_dict(self.__tracer.stats)

    def __send_filter_error_callback(self, func, msg):
        if func:
            func(msg)

    def get_waveform_advanced(self, parameters, inventory, filter_error_callback=None, **kwargs):

        start_time = kwargs.get("start_time", self.stats.StartTime)
        end_time = kwargs.get("end_time", self.stats.EndTime)
        trace_number = kwargs.get("trace_number", 0)
        tr = self.tracer

        tr.trim(starttime = start_time, endtime = end_time)
        N = len(parameters)

        for j in range(N):

            if parameters[j][0] == 'rmean':

                tr.detrend(type=parameters[j][1])

            if parameters[j][0] == 'taper':

                tr.taper(max_percentage = parameters[j][2],type=parameters[j][1])

            if parameters[j][0] == 'normalize':

                if parameters[j][1] == 0:
                    tr.normalize(norm = None)
                else:
                    tr.normalize(norm = parameters[j][1])

            if parameters[j][0] == "differentiate":
                tr.differentiate(method = parameters[j][1])

            if parameters[j][0] == "integrate":
                tr.integrate(method = parameters[j][1] )


            if parameters[j][0] == 'filter':
                filter_value = parameters[j][1]
                f_min = parameters[j][2]
                f_max = parameters[j][3]
                zero_phase = parameters[j][4]
                poles = parameters[j][5]
                try:
                    if not ObspyUtil.filter_trace(tr, filter_value, f_min, f_max, corners = poles,
                                                  zerophase = zero_phase):
                        self.__send_filter_error_callback(filter_error_callback,
                                                          "Lower frequency {} must be "
                                                          "smaller than Upper frequency {}".format(f_min, f_max))
                except ValueError as e:
                    self.__send_filter_error_callback(filter_error_callback, str(e))

            if parameters[j][0] == "wiener filter":
                print("applying wiener filter")
                time_window = parameters[j][1]
                noise_power = parameters[j][2]
                print(time_window,noise_power)
                tr = wiener_filter(tr, time_window=time_window,noise_power=noise_power)


            if parameters[j][0] == 'shift':
                shifts = parameters[j][1]
                for c, value in enumerate(shifts, 1):
                    if value[0] == trace_number:
                        tr.stats.starttime = tr.stats.starttime+value[1]

            if parameters[j][0] == 'remove response':

                f1 = parameters[j][1]
                f2 = parameters[j][2]
                f3 = parameters[j][3]
                f4 = parameters[j][4]
                water_level = parameters[j][5]
                units = parameters[j][6]
                pre_filt = (f1, f2, f3, f4)

                if inventory and units != "Wood Anderson":
                    #print("Deconvolving")
                    try:
                        tr.remove_response(inventory=inventory, pre_filt=pre_filt, output=units, water_level=water_level)
                    except:
                        print("Coudn't deconvolve", tr.stats)
                        tr.data = np.array([])

                elif inventory and units == "Wood Anderson":
                    #print("Simulating Wood Anderson Seismograph")
                    resp = inventory.get_response(tr.id, tr.stats.starttime)
                    resp = resp.response_stages[0]
                    paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
                              'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

                    paz_mine = {'sensitivity': resp.stage_gain * resp.normalization_factor, 'zeros': resp.zeros,
                                'gain': resp.stage_gain, 'poles': resp.poles}

                    try:
                        tr.simulate(paz_remove=paz_mine, paz_simulate=paz_wa, water_level=water_level)
                    except:
                        print("Coudn't deconvolve", tr.stats)
                        tr.data = np.array([])

            if parameters[j][0] == 'add white noise':
                tr = add_white_noise(tr,parameters[j][1])

            if parameters[j][0] == 'whitening':
                tr = whiten(tr,parameters[j][1], parameters[j][2])

            if parameters[j][0] == 'time normalization':
                tr = normalize(tr, norm_win=parameters[j][1],norm_method=parameters[j][2])

            if parameters[j][0] == 'wavelet denoise':
                tr = wavelet_denoise(tr, dwt = parameters[j][1], threshold=parameters[j][2])

            if parameters[j][0] == 'resample':
                tr.resample(sampling_rate=parameters[j][1],window='hanning',no_filter=parameters[j][2])

            if parameters[j][0] == 'fill gaps':
                st = Stream(tr)
                st.merge(fill_value=parameters[j][1])
                tr = st[0]

            if parameters[j][0] == 'smoothing':

                tr = smoothing(tr, type=parameters[j][1], k=parameters[j][2], fwhm=parameters[j][3])

        return tr
