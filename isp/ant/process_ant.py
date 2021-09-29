
import pickle
from obspy import read
import numpy as np
import math
from multiprocessing import Pool
import obspy
from isp.ant.signal_processing_tools import noise_processing
from obspy import Stream

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
        all_dict_matrix = []

        for list_item in list_raw:
            key_info = list_item[0][0] + list_item[0][1] + list_item[0][2]  # 'XTCAPCBHE', ...
            dict_matrix = self.create_dict_matrix(list_item, info[key_info])
            all_dict_matrix.append(dict_matrix)

        return all_dict_matrix

    def create_dict_matrix(self, list_item, info_item):
        # create an object to compress

        self.dict_matrix = {'data_matrix': [], 'metadata_list': [], 'date_list': []}
        print(" -- Matrix: " + list_item[0][0] + list_item[0][1] + list_item[0][2])

        # 1.- dict_matrix['date_list']
        date_ini = info_item[0][0].julday
        date_end = info_item[0][1].julday
        self.dict_matrix['date_list'] = [date_ini + d for d in range(date_end - date_ini + 1)]

        # 2.- dict_matrix['metadata_list']
        self.dict_matrix['metadata_list'] = info_item[1]

        # 3.- dict_matrix['data_matrix']

        num_minutes = self.num_minutes_dict_matrix  # 15min 96 incrementos
        self.num_rows = int((24 * 60) / num_minutes)
        num_columns = len(list_item) - 1
        N = num_minutes * 60 * 5 + 1  # segundos*fs
        DD = 2 ** math.ceil(math.log2(N))
        self.list_item = list_item
        # ······
        # f = [0, 1, ..., n / 2 - 1, n / 2] / (d * n) if n is even
        # f = [0, 1, ..., (n - 1) / 2 - 1, (n - 1) / 2] / (d * n) if n is odd
        # ······
        DD_half_point = int(((DD) / 2) + 1)
        self.dict_matrix['data_matrix'] = np.zeros((self.num_rows, num_columns, DD_half_point), dtype=np.complex64)
        self.inc_time = [i * 60 * num_minutes for i in range(self.num_rows + 1)]
        with Pool(processes=6) as pool:
            r = pool.map(self.process_col_matrix, range(num_columns))

        j = 0
        for col in r:
            i = 0
            for row in col:
                if row is not None and row.size == DD_half_point:
                    self.dict_matrix['data_matrix'][i, j, :] = row
                i += 1
            j += 1

        print("Finalizado", info_item)

        # compressing matrix
        # dict_matrix['data_matrix']=xr.DataArray(dictmatrix)
        if self.save_files:
            print(" -- File: " + self.output_files_path + '/' + list_item[0][0] + list_item[0][1] + list_item[0][2])
            path = self.output_files_path + '/' + list_item[0][0] + list_item[0][1] + list_item[0][2]
            print("Saving to ", path)
            file_to_store = open(path, "wb")
            pickle.dump(self.dict_matrix, file_to_store)

        print(self.dict_matrix)
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

        res = []
        tr = obspy.read(self.list_item[1 + j])[0]

        if self.remove_responseCB:
            print("removing response ", tr.id)
            tr = self.__remove_response(tr, self.f1, self.f2, self.f3, self.f4, self.water_level, self.unitsCB)

        if self.decimationCB:
            print("decimating ", tr.id)
            tr.decimate(factor=self.factor, no_filter = False)

        # for i in range(self.num_rows):
        print("Starting hard process ", tr.id)
        for i in range(len(self.inc_time) - 1):
            tr_test = tr.copy()
            tr_test.trim(starttime=tr.stats.starttime + self.inc_time[i],
                         endtime=tr.stats.starttime + self.inc_time[i + 1])
            # TODO IMPLEMENT THE DECONVOLUTION AND THE CORRECT QUALITY CONTROL --> SEND TO BLACKLIST
            if fill_gaps:
                st = self.fill_gaps(Stream(traces=tr_test), tol=self.gaps_tol)
                if st == []:
                    tr_test.data = np.zeros(len(tr_test.data),dtype=np.complex64)
                else:
                    tr_test = st[0]
            if (self.data_domain == "frequency") and len(tr[:]) > 0:
                n = tr_test.count()
                if n > 0:
                    D = 2 ** math.ceil(math.log2(n))
                    print(tr_test)
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
                        res.append(None)
                        print("dimensions does not agree")
                else:
                    res.append(None)
            else:
                res.append(None)

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

        try:

            tr.remove_response(inventory=self.inventory, pre_filt=(f1, f2, f3, f4), output=units, water_level=water_level)

        except:

            print("Coudn't deconvolve", tr.stats)

        return tr