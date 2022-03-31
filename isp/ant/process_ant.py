import multiprocessing
import os
import pickle
from obspy import read
import numpy as np
import math
from multiprocessing import Pool
import obspy
from isp.ant.signal_processing_tools import noise_processing, noise_processing_horizontals
from obspy import Stream
from isp import DISP_MAPS

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
        self.num_hours_dict_matrix = 6
        self.num_minutes_dict_matrix = 15
        self.gaps_tol = 120

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

    def create_all_dict_matrix(self, list_raw, info):
        """
        :Description: Method that sort data Vertical / horizontals and send to create_dict_matrix to create the matrix
        with the spectrums rady to make later cross correlations

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

            if channel[2] == "Z":
                channels.append(list_item[0][2])
                key_info = list_item[0][0] + list_item[0][1] + list_item[0][2]  # Data, Metadata, dates 'XTCAPCBHE', ...
                self.create_dict_matrix(list_item, info[key_info])

            else:

                # check both channels belongs to the same station
                if channel[2] in ["N", "E", "1", "2", "Y","X"] and station_check == station:

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
                        self.create_dict_matrix_horizontals(list_item_horizontals, info_N, info_E)
                        check_N = False
                        check_E = False


        return all_dict_matrix

    def create_dict_matrix(self, list_item, info_item):
        # create an object to compress

        self.dict_matrix = {'data_matrix': [], 'metadata_list': [], 'date_list': []}
        print(" -- Matrix: " + list_item[0][0] + list_item[0][1] + list_item[0][2])

        # 1.- dict_matrix['date_list']
        # taking account changes of yars
        year_ini = info_item[0][0].year
        year_end = info_item[0][1].year
        date_ini = info_item[0][0].julday
        date_end = info_item[0][1].julday
        #self.dict_matrix['date_list'] = self.__list_days(year_ini, year_end, date_ini, date_end)

        # 2.- dict_matrix['metadata_list']
        self.dict_matrix['metadata_list'] = info_item[1]
        # update the sampling_rate
        sampling_rate = info_item[1][0][0][0].sample_rate
        sampling_rate_new = sampling_rate / self.factor
        self.dict_matrix['metadata_list'][0][0][0].sample_rate = sampling_rate_new

        # 3.- dict_matrix['data_matrix']

        num_minutes = self.num_minutes_dict_matrix  # 15min 96 incrementos
        self.num_rows = int((24 * 60) / num_minutes)
        num_columns = len(list_item) - 1
        N = num_minutes * 60 * sampling_rate_new + 1  # segundos*fs
        DD = 2 ** math.ceil(math.log2(N))
        self.list_item = list_item
        # ······
        # f = [0, 1, ..., n / 2 - 1, n / 2] / (d * n) if n is even
        # f = [0, 1, ..., (n - 1) / 2 - 1, (n - 1) / 2] / (d * n) if n is odd
        # ······
        self.DD_half_point = int(((DD) / 2) + 1)
        self.dict_matrix['data_matrix'] = np.zeros((self.num_rows, num_columns, self.DD_half_point), dtype=np.complex64)
        self.inc_time = [i * 60 * num_minutes for i in range(self.num_rows + 1)]
        with Pool(processes = multiprocessing.cpu_count()) as pool:
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


        print("Finalizado", info_item)

        # compressing matrix
        # dict_matrix['data_matrix']=xr.DataArray(dictmatrix)
        if self.save_files:
            print(" -- File: " + self.output_files_path + '/' + list_item[0][0] + list_item[0][1] + list_item[0][2])
            path = self.output_files_path + '/' + list_item[0][0] + list_item[0][1] + list_item[0][2]
            print("Saving to ", path)
            #print("Saving Days", self.dict_matrix['date_list'])
            file_to_store = open(path, "wb")
            pickle.dump(self.dict_matrix, file_to_store)

        return self.dict_matrix

    def create_dict_matrix_horizontals(self, list_item_horizonrals, info_N, info_E):
        # create an object to compress

        self.dict_matrix_N = {'data_matrix_N': [], 'metadata_list_N': [], 'date_list_N': []}
        self.dict_matrix_E = {'data_matrix_E': [], 'metadata_list_E': [], 'date_list_E': []}
        print(" -- Matrix: " + list_item_horizonrals["North"][0][0] + list_item_horizonrals["North"][0][1] + list_item_horizonrals["North"][0][2])
        print(" -- Matrix: " + list_item_horizonrals["East"][0][0] + list_item_horizonrals["East"][0][1] +
              list_item_horizonrals["East"][0][2])

        # 1.- dict_matrix['date_list']

        date_ini_N = info_N[0][0].julday
        year_ini_N = info_N[0][0].year

        date_ini_E = info_E[0][0].julday
        year_ini_E = info_E[0][0].year

        date_end_N = info_N[0][1].julday
        year_end_N = info_N[0][1].year

        date_end_E = info_E[0][1].julday
        year_end_E = info_E[0][1].year


        #self.dict_matrix_N['date_list_N'] = self.__list_days(year_ini_N, year_end_N, date_ini_N, date_end_N)
        #self.dict_matrix_E['date_list_E'] = self.__list_days(year_ini_E, year_end_E, date_ini_E, date_end_E)


        # 2.- dict_matrix['metadata_list']
        self.dict_matrix_N['metadata_list_N'] = info_N[1]
        self.dict_matrix_E['metadata_list_E'] = info_E[1]

        # update the sampling_rate

        sampling_rate = info_N[1][0][0][0].sample_rate
        sampling_rate_new = sampling_rate/self.factor
        self.dict_matrix_N['metadata_list_N'][0][0][0].sample_rate = sampling_rate_new
        self.dict_matrix_E['metadata_list_E'][0][0][0].sample_rate = sampling_rate_new
        #self.info_N[1].net.sampling_rate = sampling_rate_new
        #self.info_E[1].sampling_rate = sampling_rate_new

        # 3.- dict_matrix['data_matrix']

        num_minutes = self.num_minutes_dict_matrix  # 15min 96 incrementos
        self.num_rows = int((24 * 60) / num_minutes)
        num_columns_N = len(list_item_horizonrals["North"]) - 1
        num_columns_E = len(list_item_horizonrals["East"]) - 1
        N = num_minutes * 60 * sampling_rate_new + 1  # segundos*fs

        DD = 2 ** math.ceil(math.log2(N))
        self.list_item_N = list_item_horizonrals["North"]
        self.list_item_E = list_item_horizonrals["East"]
        # ······
        # f = [0, 1, ..., n / 2 - 1, n / 2] / (d * n) if n is even
        # f = [0, 1, ..., (n - 1) / 2 - 1, (n - 1) / 2] / (d * n) if n is odd
        # ······
        self.DD_half_point = int(((DD) / 2) + 1)
        self.dict_matrix_N['data_matrix_N'] = np.zeros((self.num_rows, num_columns_N, self.DD_half_point), dtype=np.complex64)
        self.dict_matrix_E['data_matrix_E'] = np.zeros((self.num_rows, num_columns_E, self.DD_half_point), dtype=np.complex64)
        self.inc_time = [i * 60 * num_minutes for i in range(self.num_rows + 1)]
        num_columns = min(num_columns_N, num_columns_E)
        with Pool(processes=6) as pool:
            r = pool.map(self.process_col_matrix_horizontals, range(num_columns))

        j = 0
        for pair in r:
            i = 0
            for N,E in zip(pair[0],pair[1]):
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
                   list_item_horizonrals["North"][0][1] + list_item_horizonrals["North"][0][2]
            print("Saving to ", path)
            #print("Saving Days", self.dict_matrix['date_list_N'])
            #print("Saving Days", self.dict_matrix['date_list_E'])
            file_to_store = open(path, "wb")
            pickle.dump(self.dict_matrix_N, file_to_store)

            print(" -- File: " + self.output_files_path + '/' + list_item_horizonrals["East"][0][0] +
                  list_item_horizonrals["East"][0][1] + list_item_horizonrals["East"][0][2])

            path = self.output_files_path + '/' + list_item_horizonrals["East"][0][0] + \
                   list_item_horizonrals["East"][0][1] + list_item_horizonrals["East"][0][2]
            print("Saving to ", path)
            file_to_store = open(path, "wb")
            pickle.dump(self.dict_matrix_E, file_to_store)

        return self.dict_matrix

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

    def process_col_matrix(self, j, fill_gaps=True):

        check_process = True
        res = []
        tr = obspy.read(self.list_item[1 + j])[0]
        list_day = str(tr.stats.starttime.julday)+"."+str(tr.stats.starttime.year)
        print("Processing", str(tr.stats.starttime.julday)+"."+str(tr.stats.starttime.year))

        st = self.fill_gaps(Stream(traces=tr), tol=5*self.gaps_tol)

        if st == []:

            check_process = False
        else:

            tr = st[0]

        if self.remove_responseCB and check_process:
            #print("removing response ", tr.id)
            tr, check_process = self.__remove_response(tr, self.f1, self.f2, self.f3, self.f4, self.water_level, self.unitsCB)

        if self.decimationCB and check_process:
            #print("decimating ", tr.id)
            try:
                tr.decimate(factor=self.factor, no_filter = False)
            except:
                check_process = False
                print("Couldn't decimate")

        # for i in range(self.num_rows):
        #print("Starting hard process ", tr.id)

        for i in range(len(self.inc_time) - 1):

            tr_test = tr.copy()

            if check_process:

                tr_test.trim(starttime=tr.stats.starttime + self.inc_time[i],
                                 endtime=tr.stats.starttime + self.inc_time[i + 1])

                if fill_gaps:
                    st = self.fill_gaps(Stream(traces=tr_test), tol=self.gaps_tol)
                    if st == []:
                        tr_test.data = np.zeros(len(tr_test.data), dtype=np.complex64)
                    else:
                        tr_test = st[0]
                if (self.data_domain == "frequency") and len(tr[:]) > 0:
                    n = tr_test.count()
                    if n > 0:
                        D = 2 ** math.ceil(math.log2(n))
                        #print(tr_test)
                        # tr_test.plot()
                        # Prefilt
                        tr_test.detrend(type='simple')
                        tr_test.taper(max_percentage=0.05)
                        process = noise_processing(tr_test)
                        if self.time_normalizationCB:
                            process.normalize(norm_win=3, norm_method='ramn')
                        if self.whitheningCB:
                            process.whiten_new(freq_width=self.freqbandwidth, taper_edge=True)
                        try:
                            # self.dict_matrix['data_matrix'][i, j, :] = np.fft.rfft(process.tr.data, D)
                            res.append(np.fft.rfft(process.tr.data, D))
                        except:
                            res.append(np.zeros(self.DD_half_point, dtype=np.complex64))
                            print("dimensions does not agree")
                    else:
                        res.append(np.zeros(self.DD_half_point, dtype=np.complex64))
                else:
                    res.append(np.zeros(self.DD_half_point, dtype=np.complex64))
            else:
                res.append(np.zeros(self.DD_half_point, dtype=np.complex64))

        return res, list_day


    def process_col_matrix_horizontals(self, j, fill_gaps=True):

        check_process = True
        res_N = []
        res_E = []
        tr_N = obspy.read(self.list_item_N[1 + j])[0]
        tr_E = obspy.read(self.list_item_E[1 + j])[0]

        list_days_N = str(tr_N.stats.starttime.julday) + "." + str(tr_N.stats.starttime.year)
        list_days_E = str(tr_E.stats.starttime.julday) + "." + str(tr_E.stats.starttime.year)

        print("Processing", str(tr_N.stats.starttime.julday) + "." + str(tr_N.stats.starttime.year))
        print("Processing", str(tr_E.stats.starttime.julday) + "." + str(tr_E.stats.starttime.year))

        st1 = self.fill_gaps(Stream(traces=tr_N), tol=5 * self.gaps_tol)
        st2 = self.fill_gaps(Stream(traces=tr_E), tol=5 * self.gaps_tol)

        if st1 == [] or st2 == []:
            check_process = False
        else:

            tr_N = st1[0]
            tr_E = st2[0]

        if tr_N.stats.starttime.julday == tr_E.stats.starttime.julday:

            if self.remove_responseCB and check_process:
                #print("removing response ", tr_N.id, tr_E.id)
                tr_N, check_process = self.__remove_response(tr_N, self.f1, self.f2, self.f3, self.f4, self.water_level, self.unitsCB)
                tr_E, check_process = self.__remove_response(tr_E, self.f1, self.f2, self.f3, self.f4, self.water_level, self.unitsCB)

            if self.decimationCB and check_process:
                #print("decimating ", tr_N.id, tr_E.id)
                try :
                    tr_N.decimate(factor=self.factor, no_filter = False)
                    tr_E.decimate(factor=self.factor, no_filter = False)
                except:
                    check_process = False
                    print("Couldn't Decimate")

            # Prepare Matrix for North Component
            #print("Starting hard process ", tr_N.id, tr_E.id)

            for i in range(len(self.inc_time) - 1):
                if check_process:
                    tr_test_N = tr_N.copy()
                    tr_test_E = tr_E.copy()
                    maxstart = np.max([tr_test_N.stats.starttime, tr_test_E.stats.starttime])
                    #minend = np.min([tr_test_N.stats.starttime, tr_test_E.stats.starttime])

                    tr_test_N.trim(starttime = maxstart + self.inc_time[i],
                                 endtime = maxstart + self.inc_time[i + 1])

                    tr_test_E.trim(starttime=maxstart + self.inc_time[i],
                                   endtime=maxstart + self.inc_time[i + 1])

                    if fill_gaps:
                        st_N = self.fill_gaps(Stream(traces=tr_test_N), tol=self.gaps_tol)
                        st_E = self.fill_gaps(Stream(traces=tr_test_E), tol=self.gaps_tol)

                        if st_N == []:
                            tr_test_N.data = np.zeros(len(tr_test_N.data), dtype=np.complex64)
                        elif st_E == []:
                            tr_test_E.data = np.zeros(len(tr_test_E.data), dtype=np.complex64)
                        else:
                            tr_test_N = st_N[0]
                            tr_test_E = st_E[0]

                    if (self.data_domain == "frequency") and len(tr_N[:]) and len(tr_E[:]) > 0:
                        n = tr_test_N.count()
                        if n > 0:
                            D = 2 ** math.ceil(math.log2(n))
                            #print(tr_test_N, tr_test_E)
                            # tr_test.plot()
                            # Prefilt
                            tr_test_N.detrend(type='simple')
                            tr_test_E.detrend(type='simple')
                            tr_test_N.taper(max_percentage=0.05)
                            tr_test_E.taper(max_percentage=0.05)
                            process_horizontals = noise_processing_horizontals(tr_test_N, tr_test_E)
                            if self.time_normalizationCB:
                                process_horizontals.normalize(norm_win=3, norm_method='ramn')
                            if self.whitheningCB:
                                process_horizontals.whiten_new(freq_width=self.freqbandwidth, taper_edge=True)
                            try:
                                # self.dict_matrix['data_matrix'][i, j, :] = np.fft.rfft(process.tr.data, D)
                                res_N.append(np.fft.rfft(process_horizontals.tr_N.data, D))
                                res_E.append(np.fft.rfft(process_horizontals.tr_E.data, D))
                            except:
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

    def __list_days(self, year_ini,year_end, date_ini, date_end):

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


