import gc
import os
import pickle
from obspy import read, Trace, Stream, UTCDateTime
import numpy as np
import math
from multiprocessing import Pool
import obspy
from isp.ant.signal_processing_tools import noise_processing, noise_processing_horizontals
from isp import DISP_MAPS
from isp.arrayanalysis.array_analysis import array
from datetime import datetime
from scipy.signal import hilbert


class process_ant:

    def __init__(self, output_files_path, param_dict, metadata):

        """
                Process ANT,



                :param No params required to initialize the class
        """

        self.inventory = metadata
        self.dict_matrix = None
        self.list_item = None
        self.inc_time = None
        self.num_rows = None
        self.data_domain = "frequency"
        self.save_files = "True"
        self.taper_max_percent = 0.05
        self.timeWindowCB = param_dict["processing_window"]
        self.num_minutes_dict_matrix = int(self.timeWindowCB/60)
        self.gaps_tol = 120
        self.cpuCount = os.cpu_count()-1
        self.output_files_path = output_files_path
        self.parameters = param_dict
        self.f1 = param_dict["f1"]
        self.f2 = param_dict["f2"]
        self.f3 = param_dict["f3"]
        self.f4 = param_dict["f4"]
        self.waterlevelSB = param_dict["waterlevel"]
        self.unitsCB = param_dict["units"]
        self.factor = param_dict["factor"]
        self.timenorm = param_dict["method"]
        self.timewindow = param_dict["timewindow"]
        self.freqbandwidth = param_dict["freqbandwidth"]
        self.remove_responseCB = param_dict["remove_responseCB"]
        self.decimationCB = param_dict["decimationCB"]
        self.time_normalizationCB = param_dict["time_normalizationCB"]
        self.whitheningCB = param_dict["whitheningCB"]
        self.water_level = param_dict["waterlevel"]
        self.preFilter = param_dict["prefilter"]
        self.freqminFilter = param_dict["filter_freqmin"]
        self.freqmaxFilter = param_dict["filter_freqmax"]
        self.cornersFilter = param_dict["filter_corners"]


    def create_all_dict_matrix(self, list_raw, info):
        """
        :Description: Method that sort data Vertical / horizontals and send to create_dict_matrix to create the matrix
        with the spectrums ready to make later cross correlations

        :param list_raw: List with inside a header with the reference and the list with the files corresponding to

        :param info: Dictionary {'NETSTATIONCHANNEL'}:[[ST,ET],INVENTORY]

        :return:
        """
        all_dict_matrix = []
        check_N = False
        check_E = False
        list_item_horizontals = {"North": [], "East": []}

        for list_item in list_raw:

            channels = []
            station = list_item[0][1]
            station_check = station
            channel = list_item[0][2]

            if channel[2] == "Z" or channel[2] == "H":
                channels.append(list_item[0][2])
                key_info = list_item[0][0] + list_item[0][1] + list_item[0][2]  # Data, Metadata, dates 'XTCAPCBHE', ...
                list_item, info[key_info] = noise_processing.sort_verticals(list_item, info[key_info])
                self.create_dict_matrix(list_item, info[key_info])

            else:

                # check both channels belongs to the same station
                if channel[2] in ["N", "E", "1", "2", "Y", "X"] and station_check == station:

                    #channels.append(list_item[0][2])

                    if channel[2] in ["N", "1", "Y"]:
                        key_info = list_item[0][0] + list_item[0][1] + list_item[0][2]
                        info_N = info[key_info]
                        list_item_horizontals["North"] = list_item
                        check_N = True

                    elif channel[2] in ["E", "2", "X"]:
                        key_info = list_item[0][0] + list_item[0][1] + list_item[0][2]
                        info_E = info[key_info]
                        list_item_horizontals["East"] = list_item
                        check_E = True

                    if check_N and check_E:
                        # TODO VERY IMPORTANT TO CLEAN DAYS WITH NO SIMILAR STARTTIME IN BOTH COMPONENTS
                        list_item_horizontals, info_N, info_E = noise_processing.clean_horizontals_unique(list_item_horizontals,
                                                                                               info_N, info_E)
                        self.create_dict_matrix_horizontals(list_item_horizontals, info_N, info_E)
                        check_N = False
                        check_E = False


        return all_dict_matrix

    def create_dict_matrix(self, list_item, info_item):
        # create an object to compress

        self.dict_matrix = {'data_matrix': [], 'metadata_list': [], 'date_list': []}
        print(" -- Matrix: " + list_item[0][0] + list_item[0][1] + list_item[0][2])

        # 1.- dict_matrix['date_list']
        # taking account changes of years
        year_ini = info_item[0][0].year
        year_end = info_item[0][1].year
        date_ini = info_item[0][0].julday
        date_end = info_item[0][1].julday
        #self.dict_matrix['date_list'] = self.__list_days(year_ini, year_end, date_ini, date_end)

        # 2.- dict_matrix['metadata_list']
        self.dict_matrix['metadata_list'] = info_item[1]
        # update the sampling_rate
        sampling_rate = info_item[1][0][0][0].sample_rate

        if self.decimationCB:
            self.sampling_rate_new = self.factor
        else:
            self.sampling_rate_new = sampling_rate

        # self.sampling_rate_new = 5 #hacking
        self.dict_matrix['metadata_list'][0][0][0].sample_rate = self.sampling_rate_new

        # 3.- dict_matrix['data_matrix']

        num_minutes = self.num_minutes_dict_matrix  # 15min 96 incrementos
        self.num_rows = int((24 * 60) / num_minutes)
        num_columns = len(list_item) - 1
        N = num_minutes * 60 * self.sampling_rate_new  # seconds*fs
        self.dict_matrix['data_length'] = N
        # correction 16/01/2025
        # DD = 2 ** math.ceil(math.log2(N)) #Even Number of points
        DD = self.next_power_of_2(2 * N)
        self.list_item = list_item
        # ······
        # f = [0, 1, ..., n / 2 - 1, n / 2] / (d * n) if n is even
        # f = [0, 1, ..., (n - 1) / 2 - 1, (n - 1) / 2] / (d * n) if n is odd
        # ······
        self.DD_half_point = int(((DD) / 2) + 1)

        if self.timenorm == "PCC":
            self.dict_matrix['data_matrix'] = np.zeros((self.num_rows, num_columns, DD),
                                                       dtype=np.complex64)
            self.dict_matrix['CC'] = "PCC"
            self.DD_half_point = DD

        else:
            self.dict_matrix['data_matrix'] = np.zeros((self.num_rows, num_columns, self.DD_half_point),
                                                       dtype=np.complex64)
            self.dict_matrix['CC'] = "CC"

        self.inc_time = [i * 60 * num_minutes for i in range(self.num_rows + 1)]
        with Pool(processes=self.cpuCount) as pool:
            r = pool.map(self.process_col_matrix, range(num_columns))

        j = 0
        for col in r:
            i = 0
            for row in col[0]:
                if row is not None and len(row) == self.DD_half_point:
                    self.dict_matrix['data_matrix'][i, j, :] = row

                i += 1
            j += 1

            self.dict_matrix['date_list'].append(col[1])


        print("Finished", info_item)

        # compressing matrix
        # dict_matrix['data_matrix']=xr.DataArray(dictmatrix)
        if self.save_files:
            print(" -- File: " + self.output_files_path + '/' + list_item[0][0] + list_item[0][1] + list_item[0][2])
            path = self.output_files_path + '/' + list_item[0][0] + list_item[0][1] + list_item[0][2]
            print("Saving to ", path)
            #print("Saving Days", self.dict_matrix['date_list'])
            file_to_store = open(path, "wb")
            pickle.dump(self.dict_matrix, file_to_store)
            # try:
            #     del self.dict_matrix
            #     gc.collect()
            # except:
            #     pass
        #return self.dict_matrix

    def create_dict_matrix_horizontals(self, list_item_horizonrals, info_N, info_E):
        # create an object to compress

        self.dict_matrix_N = {'data_matrix_N': [], 'metadata_list_N': [], 'date_list_N': []}
        self.dict_matrix_E = {'data_matrix_E': [], 'metadata_list_E': [], 'date_list_E': []}
        print(" -- Matrix: " + list_item_horizonrals["North"][0][0] + list_item_horizonrals["North"][0][1] +
              list_item_horizonrals["North"][0][2])
        print(" -- Matrix: " + list_item_horizonrals["East"][0][0] + list_item_horizonrals["East"][0][1] +
              list_item_horizonrals["East"][0][2])

        # 1.- dict_matrix['date_list']

        # date_ini_N = info_N[0][0].julday
        # year_ini_N = info_N[0][0].year
        #
        # date_ini_E = info_E[0][0].julday
        # year_ini_E = info_E[0][0].year
        #
        # date_end_N = info_N[0][1].julday
        # year_end_N = info_N[0][1].year
        #
        # date_end_E = info_E[0][1].julday
        # year_end_E = info_E[0][1].year

        #self.dict_matrix_N['date_list_N'] = self.__list_days(year_ini_N, year_end_N, date_ini_N, date_end_N)
        #self.dict_matrix_E['date_list_E'] = self.__list_days(year_ini_E, year_end_E, date_ini_E, date_end_E)

        # 2.- dict_matrix['metadata_list']
        self.dict_matrix_N['metadata_list_N'] = info_N[1]
        self.dict_matrix_E['metadata_list_E'] = info_E[1]

        # update the sampling_rate

        sampling_rate = info_N[1][0][0][0].sample_rate
        # take the azimuth
        # TODO REVIEW IS TAKING THE CORRECT AZIMUTH CLOCKWISE FROM NORTH
        self.az = info_N[1][0][0][0].azimuth

        if self.decimationCB:
            self.sampling_rate_new = self.factor
        else:
            self.sampling_rate_new = sampling_rate

        # self.sampling_rate_new = 5 # hacking
        self.dict_matrix_N['metadata_list_N'][0][0][0].sample_rate = self.sampling_rate_new
        self.dict_matrix_E['metadata_list_E'][0][0][0].sample_rate = self.sampling_rate_new

        # 3.- dict_matrix['data_matrix']

        num_minutes = self.num_minutes_dict_matrix  # 15min 96 incrementos
        self.num_rows = int((24 * 60) / num_minutes)
        num_columns_N = len(list_item_horizonrals["North"]) - 1
        num_columns_E = len(list_item_horizonrals["East"]) - 1
        N = num_minutes * 60 * self.sampling_rate_new  # segundos*fs
        self.dict_matrix_N['data_length'] = N
        self.dict_matrix_E['data_length'] = N
        # correction 16/01/2025
        #DD = 2 ** math.ceil(math.log2(N)) #Even Number of points
        DD = self.next_power_of_2(2 * N)
        self.list_item_N = list_item_horizonrals["North"]
        self.list_item_E = list_item_horizonrals["East"]
        """
        # f = [0, 1, ..., n / 2 - 1, n / 2] / (d * n) if n is even
        # f = [0, 1, ..., (n - 1) / 2 - 1, (n - 1) / 2] / (d * n) if n is odd
        """
        self.DD_half_point = int(((DD) / 2) + 1)

        if self.timenorm == "PCC":
            self.dict_matrix_N['data_matrix_N'] = np.zeros((self.num_rows, num_columns_N, DD),
                                                       dtype=np.complex64)
            self.dict_matrix_N['CC'] = "PCC"

            self.dict_matrix_E['data_matrix_E'] = np.zeros((self.num_rows, num_columns_E, DD),
                                                           dtype=np.complex64)
            self.dict_matrix_E['CC'] = "PCC"

            self.DD_half_point = DD

        else:
            self.dict_matrix_N['data_matrix_N'] = np.zeros((self.num_rows, num_columns_N, self.DD_half_point),
                                                           dtype=np.complex64)
            self.dict_matrix_E['data_matrix_E'] = np.zeros((self.num_rows, num_columns_E, self.DD_half_point),
                                                           dtype=np.complex64)
            self.dict_matrix_N['CC'] = "CC"
            self.dict_matrix_E['CC'] = "CC"



        self.inc_time = [i * 60 * num_minutes for i in range(self.num_rows + 1)]
        num_columns = min(num_columns_N, num_columns_E)
        with Pool(processes=self.cpuCount) as pool:
            r = pool.map(self.process_col_matrix_horizontals, range(num_columns))

        j = 0
        for pair in r:
            i = 0
            for N, E in zip(pair[0],pair[1]):
                if N is not None and N.size == self.DD_half_point and E is not None and E.size == self.DD_half_point:
                    self.dict_matrix_N['data_matrix_N'][i, j, :] = N
                    self.dict_matrix_E['data_matrix_E'][i, j, :] = E
                i += 1

            j += 1
            self.dict_matrix_N['date_list_N'].append(pair[2])
            self.dict_matrix_E['date_list_E'].append(pair[3])

            # compressing matrix
        # dict_matrix['data_matrix']=xr.DataArray(dictmatrix)
        if self.save_files:

            print(" -- File: " + self.output_files_path + '/' + list_item_horizonrals["North"][0][0] +
                  list_item_horizonrals["North"][0][1] + list_item_horizonrals["North"][0][2])

            path = self.output_files_path + '/' + list_item_horizonrals["North"][0][0] + \
                   list_item_horizonrals["North"][0][1] + (list_item_horizonrals["North"][0][2][0:2]+"N")
            print("Saving to ", path)

            file_to_store = open(path, "wb")
            pickle.dump(self.dict_matrix_N, file_to_store)

            print(" -- File: " + self.output_files_path + '/' + list_item_horizonrals["East"][0][0] +
                  list_item_horizonrals["East"][0][1] + list_item_horizonrals["East"][0][2])

            path = self.output_files_path + '/' + list_item_horizonrals["East"][0][0] + \
                   list_item_horizonrals["East"][0][1] + (list_item_horizonrals["East"][0][2][0:2]+"E")
            print("Saving to ", path)
            file_to_store = open(path, "wb")
            pickle.dump(self.dict_matrix_E, file_to_store)
            # free space

            # try:
            #     del self.dict_matrix_E
            #     del self.dict_matrix_N
            #     gc.collect()
            # except:
            #     pass

        #return self.dict_matrix

    def fftzeropad(self, data):
        current_size = len(data)
        new_size = 2 ** int(np.log2(current_size) + 1)
        pad_size = new_size - current_size
        if pad_size % 2 == 0:
            left_pad = int(pad_size / 2)
            right_pad = left_pad
        else:
            left_pad = int(pad_size / 2)
            right_pad = pad_size - left_pad
        datapad = np.pad(data, [left_pad, right_pad], 'constant')
        # print(str(np.log2(left_pad+right_pad+current_size)))
        return datapad

    def process_col_matrix(self, j):

        check_process = True
        res = []
        tr_raw = obspy.read(self.list_item[1 + j])[0]
        # ensure the starttime and endtime and to have 24 h
        tr, list_day = self.ensure_24(tr_raw)

        #list_day = str(tr.stats.starttime.julday) + "." + str(tr.stats.starttime.year)
        print("Processing", tr.stats.station, tr.stats.channel, str(tr.stats.starttime.julday)+"."+str(tr.stats.starttime.year))

        st = self.fill_gaps(Stream(traces=tr), tol=5*self.gaps_tol)

        if st == []:

            check_process = False
        else:

            tr = st[0]

        if self.remove_responseCB and check_process:
            #print("removing response ", tr.id)
            tr, check_process = self.__remove_response(tr, self.f1, self.f2, self.f3, self.f4, self.water_level,
                                                       self.unitsCB)

        if self.decimationCB and check_process:
            #print("decimating ", tr.id)
            try:
                # Anti-aliasing before decimation
                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.05)
                tr.filter(type="lowpass", freq=0.4 * self.factor, zerophase=True, corners=4)
                while (tr.stats.sampling_rate//self.factor) >= 16:

                        tr.resample(sampling_rate=tr.stats.sampling_rate//(tr.stats.sampling_rate*0.1), no_filter=True)
                        tr.detrend(type="simple")
                        tr.taper(type="blackman", max_percentage=0.05)

                tr.resample(sampling_rate=self.factor, no_filter=True)
                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.05)

            except:
                check_process = False
                print("Couldn't decimate")

        for i in range(len(self.inc_time) - 1):

            tr_test = tr.copy()

            if check_process:
                # ensure the start and end to have 24 h
                tr_test.trim(starttime=tr.stats.starttime + self.inc_time[i],
                                 endtime=tr.stats.starttime + self.inc_time[i + 1])


               # TODO "Make sense to check gaps every time window??"
               #  if fill_gaps:
               #      st = self.fill_gaps(Stream(traces=tr_test), tol=self.gaps_tol)
               #      if st == []:
               #          tr_test.data = np.zeros(len(tr_test.data), dtype=np.complex64)
               #      else:
               #          tr_test = st[0]

                if (self.data_domain == "frequency") and len(tr[:]) > 0:
                    n = tr_test.count()
                    if n > 0:

                        # Corrrection 16/01/2025 --> preparing data por later crop in time domain
                        # D = 2 ** math.ceil(math.log2(n))
                        D = self.next_power_of_2(2*n)

                        # Automatic Pre-filt
                        # filter the signal between 150 seconds and 1/4 the sampling rate

                        tr_test.detrend(type='simple')
                        tr_test.taper(max_percentage=0.05)
                        tr_test.filter(type="bandpass", freqmin=0.005, freqmax=0.4*self.sampling_rate_new,
                                       zerophase=True, corners=4)


                        process = noise_processing(tr_test)

                        if self.time_normalizationCB and self.timenorm == "running avarage":
                            process.normalize(norm_win=self.timewindow, norm_method=self.timenorm)
                            if self.whitheningCB:
                                process.whiten_new(freq_width=self.freqbandwidth)


                        elif self.time_normalizationCB and self.timenorm == "1 bit":
                            if self.whitheningCB:
                                process.whiten_new(freq_width=self.freqbandwidth)
                            if self.time_normalizationCB:
                                process.normalize(norm_win=self.timewindow, norm_method=self.timenorm)

                        if self.preFilter:
                            process.tr.detrend(type='simple')
                            process.tr.taper(max_percentage=0.05)
                            process.tr.filter(type="bandpass", freqmin=self.freqminFilter, freqmax=self.freqmaxFilter,
                                           zerophase=True, corners=self.cornersFilter)

                        if self.timenorm == "PCC":

                            xaZ = hilbert(process.tr.data, N=D)
                            xaZ = xaZ[0:n]
                            process.tr.data = xaZ/np.abs(xaZ) # normalize

                        try:
                            if self.timenorm == "PCC":
                                fft = np.fft.fft(process.tr.data, D)
                                res.append(fft)
                            else:
                                res.append(np.fft.rfft(process.tr.data, D))
                        except:
                            if self.timenorm == "PCC":
                                res.append(np.zeros(D, dtype=np.complex64))
                            else:
                                res.append(np.zeros(self.DD_half_point, dtype=np.complex64))

                            print("dimensions does not agree")
                    else:
                        res.append(np.zeros(self.DD_half_point, dtype=np.complex64))
                else:
                    res.append(np.zeros(self.DD_half_point, dtype=np.complex64))
            else:
                res.append(np.zeros(self.DD_half_point, dtype=np.complex64))

        return res, list_day


    def process_col_matrix_horizontals(self, j):

        check_process = True
        res_N = []
        res_E = []
        tr_N_raw = obspy.read(self.list_item_N[1 + j])[0]
        tr_E_raw = obspy.read(self.list_item_E[1 + j])[0]
        tr_N, list_days_N  = self.ensure_24(tr_N_raw)
        tr_E, list_days_E = self.ensure_24(tr_E_raw)
        #print("Checking N", tr_N)
        #print("Checking E", tr_E)

        #list_days_N = str(tr_N.stats.starttime.julday) + "." + str(tr_N.stats.starttime.year)
        #list_days_E = str(tr_E.stats.starttime.julday) + "." + str(tr_E.stats.starttime.year)

        print("Processing", tr_N.stats.station, tr_N.stats.channel, str(tr_N.stats.starttime.julday) + "." +
              str(tr_N.stats.starttime.year))
        print("Processing", tr_E.stats.station, tr_E.stats.channel,
              str(tr_E.stats.starttime.julday) + "." + str(tr_E.stats.starttime.year))

        st1 = self.fill_gaps(Stream(traces=tr_N), tol=5 * self.gaps_tol)
        st2 = self.fill_gaps(Stream(traces=tr_E), tol=5 * self.gaps_tol)

        if st1 == [] or st2 == []:
            check_process = False
        else:

            tr_N = st1[0]
            tr_E = st2[0]

        # Very important, data is process as pairs (N,E) just if belongs to the same day!!!!
        if list_days_N == list_days_E:

            if self.remove_responseCB and check_process:
                #print("removing response ", tr_N.id, tr_E.id)
                tr_N, check_process = self.__remove_response(tr_N, self.f1, self.f2, self.f3, self.f4, self.water_level,
                                                             self.unitsCB)
                tr_E, check_process = self.__remove_response(tr_E, self.f1, self.f2, self.f3, self.f4, self.water_level,
                                                             self.unitsCB)

            if self.decimationCB and check_process:
                #print("decimating ", tr_N.id, tr_E.id)
                try:
                    tr_N.detrend(type="simple")
                    tr_N.taper(type="blackman", max_percentage=0.05)
                    tr_N.filter(type="lowpass", freq=0.4 * self.factor, zerophase=True, corners=4)
                    tr_E.detrend(type="simple")
                    tr_E.taper(type="blackman", max_percentage=0.05)
                    tr_E.filter(type="lowpass", freq=0.4 * self.factor, zerophase=True, corners=4)

                    while (tr_N.stats.sampling_rate // self.factor) >= 16:
                        tr_N.resample(sampling_rate=tr_N.stats.sampling_rate // (tr_N.stats.sampling_rate * 0.1),
                                    no_filter=True)
                        tr_N.detrend(type="simple")
                        tr_N.taper(type="blackman", max_percentage=0.05)

                        tr_E.resample(sampling_rate=tr_N.stats.sampling_rate // (tr_N.stats.sampling_rate * 0.1),
                                      no_filter=True)
                        tr_E.detrend(type="simple")
                        tr_E.taper(type="blackman", max_percentage=0.05)

                    tr_N.resample(sampling_rate=self.factor, no_filter=True)
                    tr_N.detrend(type="simple")
                    tr_N.taper(type="blackman", max_percentage=0.05)

                    tr_E.resample(sampling_rate=self.factor, no_filter=True)
                    tr_E.detrend(type="simple")
                    tr_E.taper(type="blackman", max_percentage=0.05)

                except:
                    check_process = False
                    print("Couldn't Decimate")

            for i in range(len(self.inc_time) - 1):
                if check_process:
                    tr_test_N = tr_N.copy()
                    tr_test_E = tr_E.copy()
                    maxstart = np.max([tr_test_N.stats.starttime, tr_test_E.stats.starttime])
                    #minend = np.min([tr_test_N.stats.starttime, tr_test_E.stats.starttime])

                    tr_test_N.trim(starttime=maxstart + self.inc_time[i],
                                 endtime=maxstart + self.inc_time[i + 1], pad=True, nearest_sample=True,
                                   fill_value=0)

                    tr_test_E.trim(starttime=maxstart + self.inc_time[i],
                                   endtime=maxstart + self.inc_time[i + 1], pad=True, nearest_sample=True,
                                   fill_value=0)


                    #print("Checking Test N", tr_test_N)
                    #print("Checking Test E", tr_test_E)
                    # TODO Make sense to check trim data?
                    # if fill_gaps:
                    #     st_N = self.fill_gaps(Stream(traces=tr_test_N), tol=self.gaps_tol)
                    #     st_E = self.fill_gaps(Stream(traces=tr_test_E), tol=self.gaps_tol)
                    #
                    #     if st_N == []:
                    #         tr_test_N.data = np.zeros(len(tr_test_N.data), dtype=np.complex64)
                    #     elif st_E == []:
                    #         tr_test_E.data = np.zeros(len(tr_test_E.data), dtype=np.complex64)
                    #     else:
                    #         tr_test_N = st_N[0]
                    #         tr_test_E = st_E[0]

                    if (self.data_domain == "frequency") and len(tr_N[:]) and len(tr_E[:]) > 0 and \
                            len(tr_N[:]) == len(tr_E[:]):
                        n = tr_test_N.count()
                        if n > 0:
                            # Corrrection 16/01/2025 --> preparing data por later crop in time domain
                            # D = 2 ** math.ceil(math.log2(n))
                            D = self.next_power_of_2(2 * n)

                            #TODO Autorotate to azimuth in metadata


                            # Prefilt

                            # Automatic Pre-filt
                            # filter the signal between 150 seconds and 1/4 the sampling rate
                            tr_test_N.detrend(type='simple')
                            tr_test_E.detrend(type='simple')
                            tr_test_N.taper(max_percentage=0.05)
                            tr_test_E.taper(max_percentage=0.05)

                            tr_test_N.filter(type="bandpass", freqmin=0.005, freqmax=0.4*self.sampling_rate_new,
                                             zerophase=True, corners=4)

                            tr_test_E.filter(type="bandpass", freqmin=0.005, freqmax=0.4*self.sampling_rate_new,
                                             zerophase=True, corners=4)


                            process_horizontals = noise_processing_horizontals(tr_test_N, tr_test_E)

                            # rotate to N & E, designed specially for OBSs
                            process_horizontals.rotate2NE(self.az)

                            if self.time_normalizationCB and self.timenorm == "running avarage":
                                process_horizontals.normalize(norm_win=self.timewindow, norm_method=self.timenorm)
                                if self.whitheningCB:
                                    process_horizontals.whiten_new(freq_width=self.freqbandwidth)


                            elif self.time_normalizationCB and self.timenorm == "1 bit":
                                if self.whitheningCB:
                                    process_horizontals.whiten_new(freq_width=self.freqbandwidth)
                                if self.time_normalizationCB:
                                    process_horizontals.normalize(norm_win=self.timewindow, norm_method=self.timenorm)

                            if self.preFilter:
                                process_horizontals.tr_N.detrend(type='simple')
                                process_horizontals.tr_E.detrend(type='simple')
                                process_horizontals.tr_E.taper(max_percentage=0.05)
                                process_horizontals.tr_E.taper(max_percentage=0.05)

                                process_horizontals.tr_N.filter(type="bandpass", freqmin=self.freqminFilter,
                                                 freqmax=self.freqmaxFilter,
                                                 zerophase=True, corners=self.cornersFilter)
                                process_horizontals.tr_E.filter(type="bandpass", freqmin=self.freqminFilter,
                                                 freqmax=self.freqmaxFilter,
                                                 zerophase=True, corners=self.cornersFilter)

                            if self.time_normalizationCB and self.timenorm == "PCC":

                                xaN = hilbert(process_horizontals.tr_N.data, N=D)
                                xaN = xaN[0:n]
                                xaE = hilbert(process_horizontals.tr_E.data, N=D)
                                xaE = xaE[0:n]
                                process_horizontals.tr_N.data = xaN / np.abs(xaN)  # normalize
                                process_horizontals.tr_E.data = xaE / np.abs(xaE)  # normalize

                            try:
                                if self.timenorm == "PCC":
                                    fft = np.fft.fft(process_horizontals.tr_N.data, D)
                                    res_N.append(fft)
                                    fft = np.fft.fft(process_horizontals.tr_E.data, D)
                                    res_E.append(fft)
                                else:
                                    res_N.append(np.fft.rfft(process_horizontals.tr_N.data, D))
                                    res_E.append(np.fft.rfft(process_horizontals.tr_E.data, D))
                            except:
                                if self.timenorm == "PCC":
                                    res_N.append(np.zeros(D, dtype=np.complex64))
                                    res_E.append(np.zeros(D, dtype=np.complex64))
                                else:
                                    res_N.append(np.zeros(self.DD_half_point, dtype=np.complex64))
                                    res_E.append(np.zeros(self.DD_half_point, dtype=np.complex64))
                                print("dimensions does not agree")

                        else:
                            res_N.append(np.zeros(self.DD_half_point, dtype=np.complex64))
                            res_E.append(np.zeros(self.DD_half_point, dtype=np.complex64))
                else:
                    res_N.append(np.zeros(self.DD_half_point, dtype=np.complex64))
                    res_E.append(np.zeros(self.DD_half_point, dtype=np.complex64))

        res = [res_N, res_E, list_days_N, list_days_E]
        return res


    def get_all_values(self, nested_dictionary):
        list_raw = []
        for key, value in nested_dictionary.items():

            if type(value) is dict:
                list_raw.extend(self.get_all_values(value))

            else:
                list_raw.append(value)

        return list_raw

    def __check_starttimes(self, st1, st2, st3):
        endtime1 = st1[0].stats.endtime
        starttime1 = st2[0].stats.starttime
        endtime2 = st2[0].stats.endtime
        starttime2 = st3[0].stats.starttime
        if abs(starttime1 - endtime1) and abs(starttime2 - endtime2) < 600:
            check = True
        else:
            check = False
        return check


    def check_gaps(self, gaps, tol):
        time_gaps = []
        for i in gaps:
            time_gaps.append(i[6])

        sum_total = sum(time_gaps)

        if sum_total > tol:
            check = True
        else:
            check = False

        return check

    def fill_gaps(self, st, tol):
        gaps = st.get_gaps()

        if len(gaps) > 0 and self.check_gaps(gaps, tol):
            st.print_gaps()
            st = []

        elif len(gaps) > 0 and self.check_gaps(gaps, tol) == False:
            st.print_gaps()
            st.merge(fill_value="interpolate", interpolation_samples=-1)

        elif len(gaps) == 0 and self.check_gaps(gaps, tol) == False:
            pass

        return st

    def next_power_of_2(self, n):
        """
        Return next power of 2 greater than or equal to n
        """
        return 2 ** (n - 1).bit_length()


    def temporal_window(self, channel_list):
        starts = []
        ends = []

        for path in range(len(channel_list)):
            header = read(path, headlonly=True)
            starttime = header[0].stats.starttime
            endtime = header[0].stats.endtime
            starts.append(starttime)
            ends.append(endtime)
        min_start = min(starts)
        max_ends = max(ends)

        return min_start, max_ends

    def ensure_24(self, tr):
        # Ensure that this trace is set to have 24h points padding with zeros the starttime and endtime
        # take random numbers to ensure the day
        random_list = np.random.choice(len(tr), 100)
        times_posix = tr.times(type="timestamp")
        days_prob = times_posix[random_list.tolist()]
        days_prob_max = days_prob.tolist()
        max_prob = max(set(days_prob_max), key=days_prob_max.count)
        year = int(datetime.utcfromtimestamp(max_prob).strftime('%Y'))
        month = int(datetime.utcfromtimestamp(max_prob).strftime('%m'))
        day = int(datetime.utcfromtimestamp(max_prob).strftime('%d'))

        check_starttime = UTCDateTime(year=year, month=month, day=day, hour=00, minute=00, microsecond=00)
        check_endtime = check_starttime + 24 * 3600
        date = str(check_starttime.julday) + "." + str(check_starttime.year)

        tr.detrend(type="simple")
        tr.trim(starttime=check_starttime, endtime=check_endtime, pad=True, nearest_sample=False, fill_value=0)
        return tr, date


    def __remove_response(self, tr, f1, f2, f3, f4, water_level, units):

        done = True

        try:
            tr.remove_response(inventory=self.inventory, pre_filt=(f1, f2, f3, f4), output=units, water_level=water_level)
        except:
            print("Coudn't deconvolve", tr.stats)
            done = False

        return tr, done

    def __isleapyear(self, year):
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            return True
        return False

    def __list_days(self, year_ini, year_end, date_ini, date_end):

        year_list = [year for year in range(year_ini, year_end)]
        bi = []
        no_bi = []

        for year_ckeck in year_list:

            if self.__isleapyear(year_ckeck):

                bi.append(1)

            else:

                no_bi.append(1)

        n1 = len(no_bi)
        n2 = len(bi)
        date_end2 = date_end + 365 * n1 + 366 * n2 + 1
        def_list = [d for d in range(date_ini, date_end2)]

        return def_list

class disp_maps_tools:

      @classmethod
      def disp_maps_availables(cls):

          file_to_add = []
          file_checked = []

          for top_dir, _, files in os.walk(DISP_MAPS):
              for file in files:
                  file_to_add.append(os.path.join(top_dir, file))

          for file in file_to_add:
              if cls.is_valid_dsp_pickle(file):
                     file_checked.append(file)


          return file_checked


      @staticmethod
      def is_valid_file(file_path):
        """
        Return True if path is an existing regular file and a valid pickle. False otherwise.
        :param file_path: The full file's path.
        :return: True if path is an existing regular file and a valid mseed. False otherwise.
        """

        return os.path.isfile(file_path)

      @staticmethod
      def is_valid_dsp_pickle(file_path):

          try:

            dsp_map = pickle.load(open(file_path, "rb"))


            items = ['period', 'paths', 'rejected_paths', 'ref_velocity', 'alpha0', 'alpha1', 'beta', 'sigma',
                       'm_opt_relative',
                       'm_opt_absolute', 'grid', 'resolution_map', 'cov_map', 'residuals', 'rms']

            check_list = []
            for key in dsp_map[0].keys():
                check_list.append(key)


            return check_list == items

          except:

            return False


class clock_process:
    def __init__(self, stack_day, metadata, name, common_dates_list, dims):

        """
                Process ANT,



                :param No params required to initialize the class
        """

        self.stack_day = stack_day
        self.metadata = metadata
        self.name = name
        self.common_date_list = common_dates_list
        self.dims = dims

    def sort_dates(self):
        # extract years
        years = {}
        all_years = []
        list_iterate = self.common_date_list
        for date in list_iterate:
            date = date.split(".")
            julday = date[0]
            year = date[1]
            if year not in years.keys():
                years[year] = [julday+"."+year]
            else:
                years[year].append(julday+"."+year)

        for keys in years:
            date_index = years[keys]
            date_index = sorted(date_index, key=float)
            all_years = all_years+date_index

        old_index, new_index = self.help_swap(all_years, list_iterate)

        return all_years, old_index, new_index


    def help_swap(self, sort_list, raw_list):

        l1 = sort_list
        l2 = raw_list

        index1_list = []  # lista de indices incorrectos en la matriz original
        items1_list = []

        index2_list = []
        for index1, item1 in enumerate(l1):
            index_check = l2.index(item1)
            if index1 != index_check:
                index1_list.append(index1)
                items1_list.append(item1)
                index2_list.append(index_check)

        # arr[(:,index1_list)] = arr[:, index2_list] #swap index
        old_index = index1_list
        new_index = index2_list

        return old_index, new_index

    def daily_stack_part(self, part_day=20, type="Linear", power=2, overlap=75):

        all_years, old_index, new_index = self.sort_dates()
        self.common_date_list = all_years
        stack_day = self.stack_day.transpose()
        stack_day[:, [old_index]] = stack_day[:, [new_index]]
        stack_day = stack_day.transpose()
        stack_partial = []
        part_day_overlap = int(part_day*(1-overlap/100))
        numeration = [x for x in range(0, self.dims[0], part_day_overlap)]
        numeration_days = []

        for days in numeration[0:-1]:
            # conditions to proceed
            # 1. check that the list of partial items have no gaps
            if self.check_listdays_gaps(all_years, part_day, days):
                # take the day of self.common_date_list
                numeration_days.append(self.common_date_list[days])
                if type == "Linear":
                    data_new = np.zeros(self.dims[1])
                if type == "PWS":
                    data_new = np.zeros((part_day, self.dims[1]))
                index = 0
                for day in range(days, part_day+days):
                    if day <= self.dims[0]-1:
                        if type == "Linear":
                            data = stack_day[day, :]
                            data_new = data_new + data
                        else:
                            data_new[index , :] = stack_day[day, :]
                            index = index +1

                if type == "PWS":
                    stack_obj = array()
                    data_new = stack_obj.stack(data_new, stack_type='Phase Weigth Stack', order=power)

                # num = len(data_new)
                # if (num % 2) == 0:
                #
                #     # print(“Thenumber is even”)
                #     c = int(np.ceil(num / 2.) + 1)
                # else:
                #     # print(“The providednumber is odd”)
                #     c = int(np.ceil((num + 1) / 2))
                # data_new = (np.roll(data_new, c))/part_day
                #data_new = (np.roll(data_new, int(len(data_new) / 2)))/part_day
                self.metadata['location'] = str(days+int(part_day/2))
                stack_partial.append(Trace(data=data_new, header=self.metadata))
                np.zeros(self.dims[1])
                if type == "Linear":
                    del data

        st = Stream(stack_partial)
        numeration_days = self.extract_list_days(numeration_days)
        numeration_days = np.array(numeration_days) + int(part_day/2)
        data_to_save = {"dates": numeration_days.tolist(), "stream": st}

        file_to_store = open(self.name, "wb")
        pickle.dump(data_to_save, file_to_store)

        # clean memory
        try:
            del stack_day
            del self.stack_day
            gc.collect()
        except:
            pass



    def check_listdays_gaps(self, dates, part_day, days):

        # check if there are day gaps in consecutive days

        days = [day for day in range(days, part_day + days)]
        all_dates = []

        if len(dates) >= days[-1]:

            for day in days:
                if day <= len(dates)-1:
                    all_dates.append(dates[day])

            all_dates = np.array(self.extract_list_days(all_dates))
            sum_all = np.sum(np.diff(all_dates))

            if sum_all > 0.1*part_day:
                return True
            else:
                return False
        else:
            return False



    def extract_list_days(self, dates):
        # transform julday + "." + year --> ini_julday and  consecutive days
        days = []
        years = []
        leapyear = False
        # unwrap dates

        # extract years
        for year in dates:
            date_year = year.split(".")
            date_year = int(date_year[1])
            if date_year not in years:
                years.append(date_year)
        j = 0
        for year in years:
            for date in dates:
                date = date.split(".")
                day = int(date[0])
                year_list = int(date[1])
                if year == year_list:
                    if j > 0:
                        if years[j - 1] % 4 == 0 and (years[j - 1] % 100 != 0 or years[j - 1] % 400 == 0):
                            leapyear = True

                        if leapyear:
                            days.append(j * 366 + day)
                        elif leapyear == False:
                            days.append(j * 365 + day)
                    else:
                        days.append(day)

            j = j + 1

        return days