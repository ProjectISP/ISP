#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import multiprocessing
from multiprocessing import Pool
from obspy import Trace, Stream
from obspy.core import UTCDateTime
import numpy as np
import pickle
import os
from obspy import read
from obspy.geodetics import gps2dist_azimuth
from isp.ant.process_ant import clock_process
import gc


class noisestack:

    def __init__(self, output_files_path, stations, channels, stack, power, autocorr, min_distance,
                 dailyStacks, overlap):

        """
                Process ANT, Cross + Stack

                :param params required to initialize the class
        """

        self.__metadata_manager = None
        self.output_files_path = output_files_path
        self.channel = channels
        self.stations_whitelist = stations
        self.stack = stack
        self.power = power
        self.year = 2000
        self.autocorr = autocorr
        self.min_dist = min_distance
        self.dailyStacks = dailyStacks
        self.overlap = overlap

    def check_file(self, filename):

        return os.path.isfile(filename)

    def check_path(self):

        self.stack_files_path = os.path.join(self.output_files_path, "stack")
        self.stack_rotated_files_path = os.path.join(self.output_files_path, "stack_rotated")
        self.stack_daily_files_path = os.path.join(self.output_files_path, "stack_daily")

        if not os.path.exists(self.stack_files_path):
            os.makedirs(self.stack_files_path)

        if not os.path.exists(self.stack_rotated_files_path):
            os.makedirs(self.stack_rotated_files_path)

        if not os.path.exists(self.stack_daily_files_path):
            os.makedirs(self.stack_daily_files_path)

        # Ficheros de datos
        # self.pickle_files = [pickle_file for pickle_file in os.listdir(self.output_files_path) if self.channel in pickle_file]
        self.pickle_files = []
        for pickle_file in os.listdir(self.output_files_path):
            for jj in range(len(self.channel)):
                if self.channel[jj] in pickle_file:
                    self.pickle_files.append(pickle_file)

        self.stack_files_path_done = []
        for pickle_file in os.listdir(self.stack_files_path):
            for jj in range(len(self.channel)):
                self.stack_files_path_done.append(pickle_file)

    # Para cada pareja de ficheros, se cargan los ficheros y se multiplican las matrices de datos que contienen, sólo en los días comunes
    # Indices i,j: se refieren a ficheros de datos file_i, file_j que contiene las matrices que se multiplicarán.

    # Sólo hacer si i>= j, para hacer sólo triangular superior
    # y reducir el número de operaciones

    def run_cross_stack(self):
        if self.autocorr:
            self.hard_process_full()
            print("End Cross & Stack")
        else:
            self.hard_process_simple()
            print("End Cross & Stack")

    def hard_process_simple(self):
        self.check_path()
        with Pool(processes=3) as pool:
            pool.map(self.hard_process_simple_parallel, range(len(self.pickle_files)))

    def hard_process_full(self):
        self.check_path()
        with Pool(processes=3) as pool:
            pool.map(self.hard_process_full_parallel, range(len(self.pickle_files)))

    def hard_process_simple_parallel(self, i):
        file_i = self.pickle_files[i]
        x_station = len(file_i) - 3

        try:
            if file_i[-1] in ["N", "E", "X", "Y", "1", "2"]:
                key1_i = "data_matrix" + "_" + file_i[-1]
                key2_i = 'metadata_list' + "_" + file_i[-1]
                key3_i = 'date_list' + "_" + file_i[-1]
            else:
                key1_i = "data_matrix"
                key2_i = 'metadata_list'
                key3_i = 'date_list'

            for j, file_j in enumerate(self.pickle_files):
                y_station = len(file_j) - 3
                filename = file_i[:2] + "." + file_i[2:x_station] + "_" + file_j[2:y_station] + "." + file_i[-1] + \
                           file_j[-1]
                if (
                        filename not in self.stack_files_path_done
                        and (
                        not self.stations_whitelist  # no whitelist = allow all
                        or file_i[2:x_station] in self.stations_whitelist
                        or file_j[2:y_station] in self.stations_whitelist
                )):
                    if i < j:
                        if file_j[-1] in ["N", "E", "X", "Y", "1", "2"]:
                            key1_j = "data_matrix" + "_" + file_j[-1]
                            key2_j = 'metadata_list' + "_" + file_j[-1]
                            key3_j = 'date_list' + "_" + file_j[-1]
                        else:
                            key1_j = "data_matrix"
                            key2_j = 'metadata_list'
                            key3_j = 'date_list'

                        print("(i=" + str(i) + ",j=" + str(j) + ") -> (" + file_i + "," + file_j + ")")

                        with open(os.path.join(self.output_files_path, file_i), 'rb') as h_i, open(
                                os.path.join(self.output_files_path, file_j),
                                'rb') as h_j:

                            # Cada fichero file_i y file_i contiene:
                            # dict_matrix ={ 'data_matrix': [] , 'metadata_list': [], 'date_list': []}
                            dict_matrix_file_i = pickle.load(h_i)
                            dict_matrix_file_j = pickle.load(h_j)

                            normalization_i = dict_matrix_file_i["CC"]
                            normalization_j = dict_matrix_file_j["CC"]
                            if normalization_i == normalization_j and normalization_i == "PCC":
                                normalization = "PCC"
                            else:
                                normalization = "CC"

                            # 28-05-2024, important 2n - 1
                            cross_length = int(dict_matrix_file_i["data_length"] +
                                               dict_matrix_file_j["data_length"] - 1)

                            data_matrix_file_i_corr = dict_matrix_file_i[key1_i]
                            data_matrix_file_j_corr = dict_matrix_file_j[key1_j]
                            metadata_list_file_i = dict_matrix_file_i[key2_i]
                            metadata_list_file_j = dict_matrix_file_j[key2_j]
                            date_list_file_i = dict_matrix_file_i[key3_i]
                            date_list_file_j = dict_matrix_file_j[key3_j]
                            # realease memory
                            del dict_matrix_file_i
                            del dict_matrix_file_j
                            gc.collect()
                            # coordinates
                            net_i = metadata_list_file_i[0]
                            net_j = metadata_list_file_j[0]
                            sta_i = net_i[0]
                            sta_j = net_j[0]
                            lat_i = sta_i.latitude
                            lon_i = sta_i.longitude
                            lat_j = sta_j.latitude
                            lon_j = sta_j.longitude

                            # sampling rate

                            self.sampling_rate = metadata_list_file_i[0][0][0].sample_rate

                            dist, bazim, azim = self.__coords2azbazinc(lat_i, lon_i, lat_j, lon_j)
                            if (dist / 1000) <= self.min_dist:

                                # Lista de días de cada fichero
                                print("dict_matrix_file_i['date_list']: " + str(date_list_file_i))
                                print("dict_matrix_file_j['date_list']: " + str(date_list_file_j))

                                if (len(date_list_file_i) > 0 and len(date_list_file_j) > 0):
                                    date_list_file_i = self.check_header(date_list_file_i)
                                    date_list_file_j = self.check_header(date_list_file_j)

                                    # check for duplicate days
                                    elements_i_to_delete = self.checkIfDuplicates(date_list_file_i)
                                    elements_j_to_delete = self.checkIfDuplicates(date_list_file_j)

                                    # refress date_list without repeated days
                                    if len(elements_i_to_delete) > 0:
                                        index_set = set(elements_i_to_delete)
                                        date_list_file_i_common = [x for i, x in enumerate(date_list_file_i) if
                                                                   i not in index_set]
                                    else:
                                        date_list_file_i_common = date_list_file_i

                                    if len(elements_j_to_delete) > 0:
                                        index_set = set(elements_j_to_delete)
                                        date_list_file_j_common = [x for i, x in enumerate(date_list_file_j) if
                                                                   i not in index_set]
                                    else:
                                        date_list_file_j_common = date_list_file_j

                                    # eliminate non common days
                                    common_dates_list = [value for value in date_list_file_i_common if
                                                         value in date_list_file_j_common]

                                    for date_i in date_list_file_i:
                                        if (not date_i in common_dates_list):
                                            print("Delete day: " + str(date_i) + " from " + file_i)
                                            elements_i_to_delete.append(date_list_file_i.index(date_i))

                                    if len(elements_i_to_delete) > 0:
                                        data_matrix_file_i_corr = np.delete(data_matrix_file_i_corr,
                                                                            elements_i_to_delete, 1)

                                    for date_j in date_list_file_j:
                                        if (not date_j in common_dates_list):
                                            print("Delete day: " + str(date_j) + " from " + file_j)
                                            elements_j_to_delete.append(date_list_file_j.index(date_j))

                                    if len(elements_j_to_delete) > 0:
                                        data_matrix_file_j_corr = np.delete(data_matrix_file_j_corr,
                                                                            elements_j_to_delete, 1)

                                    # ###########
                                    # Correlación: multiplicación de matrices elemento a elemento
                                    # ###########

                                    # introduce sort matrix columns
                                    ######
                                    common_dates_list, old_index, new_index = self.sort_dates(common_dates_list)
                                    data_matrix_file_i_corr[:, [old_index], :] = data_matrix_file_i_corr[:, [new_index],
                                                                                 :]
                                    data_matrix_file_j_corr[:, [old_index], :] = data_matrix_file_j_corr[:, [new_index],
                                                                                 :]
                                    corr_ij_freq = data_matrix_file_i_corr * np.conj(data_matrix_file_j_corr)
                                    ######

                                    # La matriz resultante se pasa al dominio del tiempo
                                    # Se reserva el espacio para la matriz de correlaciones en el dominio del tiempo
                                    size_1d = corr_ij_freq.shape[0]
                                    size_2d = corr_ij_freq.shape[1]
                                    size_2d_all = size_1d + size_2d
                                    corr_ij_freq[np.isnan(corr_ij_freq)] = 0.0 + 0.0j
                                    zero_vectors = np.all(corr_ij_freq == 0.0 + 0.0j, axis=2)
                                    count_zero_vectors = np.sum(zero_vectors)
                                    size_2d_all = size_2d_all - count_zero_vectors

                                    # Crop in time domain 16/01/2025

                                    if normalization == "PCC":
                                        corr_ij_time = np.real(np.fft.ifft(corr_ij_freq, axis=2))
                                        corr_ij_time = np.fft.ifftshift(corr_ij_time)
                                    else:

                                        corr_ij_time = np.real(np.fft.irfft(corr_ij_freq, axis=2))
                                        corr_ij_time = np.fft.ifftshift(corr_ij_time)

                                    # Crop in time domain 16/01/2025
                                    pad_length = corr_ij_time.shape[2]

                                    start_idx = (pad_length - cross_length) // 2
                                    end_idx = start_idx + cross_length
                                    corr_ij_time = corr_ij_time[:, :, start_idx:end_idx]

                                    # save memory
                                    if self.stack != "PWS":
                                        try:
                                            del data_matrix_file_i_corr
                                            del data_matrix_file_j_corr
                                            del corr_ij_freq
                                            gc.collect()
                                        except:
                                            pass
                                    else:
                                        try:
                                            del data_matrix_file_i_corr
                                            del data_matrix_file_j_corr
                                            gc.collect()
                                        except:
                                            pass

                                    if self.stack == "nrooth":
                                        corr_ij_time = (np.abs(corr_ij_time) ** (1 / self.power)) * np.sign(
                                            corr_ij_time)
                                        c_stack = np.sum(np.sum(corr_ij_time, axis=1), axis=0) / size_2d_all

                                    elif self.stack == "Linear":
                                        c_stack = np.sum(np.sum(corr_ij_time, axis=1), axis=0) / size_2d_all


                                    elif self.stack == "PWS":

                                        # estimate the analytic function and then the instantaneous phase matrix
                                        # analytic_signal = np.zeros((size_1d, size_2d, size_3d), dtype=np.complex64)
                                        # if normalization == "CC":
                                        f, c, d = corr_ij_freq.shape
                                        dim_full = 2 * d - 1
                                        non_real_dim = (dim_full - d)

                                        c = np.zeros((f, c, non_real_dim), dtype=np.complex64)
                                        signal_rfft_mod = np.concatenate((corr_ij_freq, c), axis=2)
                                        signal_rfft_mod[((non_real_dim // 2) + 1):] = signal_rfft_mod[((non_real_dim
                                                                                                        // 2) + 1):] * 0
                                        signal_rfft_mod[1:non_real_dim // 2] = 2 * signal_rfft_mod[1:non_real_dim // 2]

                                        # Generate the analytic function matrix
                                        analytic_signal = np.fft.ifft(signal_rfft_mod, axis=2)
                                        analytic_signal = np.fft.ifftshift(analytic_signal)

                                        # elif normalization != "CC":
                                        #
                                        #     # Generate the analytic function matrix
                                        #     analytic_signal = np.fft.ifft(corr_ij_freq, cross_length, axis=2)
                                        #     analytic_signal = np.fft.ifftshift(analytic_signal)

                                        # Compute linear stack

                                        # Crop the analytic signal 16/01/2025
                                        analytic_signal = analytic_signal[:, :, start_idx:end_idx]

                                        c_stack = np.sum(np.sum(corr_ij_time, axis=1), axis=0) / size_2d_all
                                        c_stack_max = np.max(c_stack)

                                        phase_stack = np.sum(np.sum(analytic_signal, axis=1), axis=0) / size_2d_all

                                        # this point proceed to the PWS
                                        phase_stack = (np.abs(phase_stack)) ** self.power
                                        c_stack = c_stack * phase_stack
                                        c_stack = (c_stack * c_stack_max) / np.max(c_stack)

                                    # c_stack par, impar ...
                                    # num = len(c_stack)
                                    # if (num % 2) == 0:
                                    #
                                    #     #print(“Thenumber is even”)
                                    #     c = int(np.ceil(num / 2.) + 1)
                                    # else:
                                    #     #print(“The providednumber is odd”)
                                    #     c = int(np.ceil((num + 1)/2))
                                    #
                                    # c_stack = np.roll(c_stack, c)

                                    print("stack[" + str(i) + "," + str(j) + "]:")
                                    # print(c_stack)

                                    # Guardar fichero
                                    # print(metadata_list_file_i)
                                    # print(metadata_list_file_j)
                                    stats = {}
                                    x_station = len(file_i) - 3
                                    y_station = len(file_j) - 3
                                    stats['network'] = file_i[:2]
                                    stats['station'] = file_i[2:x_station] + "_" + file_j[2:y_station]
                                    stats['channel'] = file_i[-1] + file_j[-1]
                                    stats['sampling_rate'] = self.sampling_rate
                                    stats['npts'] = len(c_stack)
                                    stats['mseed'] = {'dataquality': 'D', 'geodetic': [dist, bazim, azim],
                                                      'cross_channels': file_i[-1] + file_j[-1],
                                                      'coordinates': [lat_i, lon_i, lat_j, lon_j]}
                                    stats['starttime'] = UTCDateTime("2000-01-01T00:00:00.0")
                                    # stats['info'] = {'geodetic': [dist, bazim, azim],'cross_channels':file_i[-1]+file_j[-1]}
                                    st = Stream([Trace(data=c_stack, header=stats)])
                                    # Nombre del fichero = XT.STA1_STA2.ZE
                                    filename = file_i[:2] + "." + file_i[2:x_station] + "_" + file_j[
                                                                                              2:y_station] + "." + \
                                               file_i[-1] + file_j[-1]
                                    path_name = os.path.join(self.stack_files_path, filename)
                                    print(path_name)
                                    st.write(path_name, format='H5')
                                    #
                                    if self.dailyStacks:
                                        path_name = os.path.join(self.stack_daily_files_path, filename + "_daily")
                                        dims = [corr_ij_time.shape[1], corr_ij_time.shape[2]]
                                        stack_day = np.sum(corr_ij_time, axis=0)

                                        # here I can delete corr_ij_time
                                        del corr_ij_time
                                        gc.collect()

                                        clock = clock_process(stack_day, stats, path_name, common_dates_list, dims)
                                        clock.daily_stack_part(type=self.stack, power=self.power, overlap=self.overlap)
                                else:
                                    print("Empty date_list.")
                                print("-----")
                            else:
                                del metadata_list_file_i
                                del metadata_list_file_j
                                gc.collect()
                                print("Excluded cross correlations for being out of maximum distance ", dist * 1E-3,
                                      "<", self.min_dist)
        except:
            print("Something went wrong at:", file_i)

    def hard_process_full_parallel(self, i):
        file_i = self.pickle_files[i]
        x_station = len(file_i) - 3
        try:
            if file_i[-1] in ["N", "E", "X", "Y", "1", "2"]:
                key1_i = "data_matrix" + "_" + file_i[-1]
                key2_i = 'metadata_list' + "_" + file_i[-1]
                key3_i = 'date_list' + "_" + file_i[-1]
            else:
                key1_i = "data_matrix"
                key2_i = 'metadata_list'
                key3_i = 'date_list'

            for j, file_j in enumerate(self.pickle_files):
                y_station = len(file_j) - 3
                filename = file_i[:2] + "." + file_i[2:x_station] + "_" + file_j[2:y_station] + "." + file_i[-1] + \
                           file_j[-1]
                if (
                        filename not in self.stack_files_path_done
                        and (
                        not self.stations_whitelist  # no whitelist = allow all
                        or file_i[2:x_station] in self.stations_whitelist
                        or file_j[2:y_station] in self.stations_whitelist
                )):
                    if file_j[-1] in ["N", "E", "X", "Y", "1", "2"]:
                        key1_j = "data_matrix" + "_" + file_j[-1]
                        key2_j = 'metadata_list' + "_" + file_j[-1]
                        key3_j = 'date_list' + "_" + file_j[-1]
                    else:
                        key1_j = "data_matrix"
                        key2_j = 'metadata_list'
                        key3_j = 'date_list'

                    print("(i=" + str(i) + ",j=" + str(j) + ") -> (" + file_i + "," + file_j + ")")

                    with open(os.path.join(self.output_files_path, file_i), 'rb') as h_i, open(
                            os.path.join(self.output_files_path, file_j),
                            'rb') as h_j:

                        # Cada fichero file_i y file_i contiene:
                        # dict_matrix ={ 'data_matrix': [] , 'metadata_list': [], 'date_list': []}
                        dict_matrix_file_i = pickle.load(h_i)
                        dict_matrix_file_j = pickle.load(h_j)

                        normalization_i = dict_matrix_file_i["CC"]
                        normalization_j = dict_matrix_file_j["CC"]
                        if normalization_i == normalization_j and normalization_i == "PCC":
                            normalization = "PCC"
                        else:
                            normalization = "CC"

                        # 28-05-2024, important 2n - 1
                        cross_length = int(dict_matrix_file_i["data_length"] +
                                           dict_matrix_file_j["data_length"] - 1)

                        data_matrix_file_i_corr = dict_matrix_file_i[key1_i]
                        data_matrix_file_j_corr = dict_matrix_file_j[key1_j]
                        metadata_list_file_i = dict_matrix_file_i[key2_i]
                        metadata_list_file_j = dict_matrix_file_j[key2_j]
                        date_list_file_i = dict_matrix_file_i[key3_i]
                        date_list_file_j = dict_matrix_file_j[key3_j]
                        # realease memory
                        del dict_matrix_file_i
                        del dict_matrix_file_j
                        gc.collect()

                        # coordinates
                        net_i = metadata_list_file_i[0]
                        net_j = metadata_list_file_j[0]
                        sta_i = net_i[0]
                        sta_j = net_j[0]
                        lat_i = sta_i.latitude
                        lon_i = sta_i.longitude
                        lat_j = sta_j.latitude
                        lon_j = sta_j.longitude

                        # sampling rate

                        self.sampling_rate = metadata_list_file_i[0][0][0].sample_rate

                        dist, bazim, azim = self.__coords2azbazinc(lat_i, lon_i, lat_j, lon_j)

                        if (dist / 1000) <= self.min_dist:

                            # Lista de días de cada fichero
                            print("dict_matrix_file_i['date_list']: " + str(date_list_file_i))
                            print("dict_matrix_file_j['date_list']: " + str(date_list_file_j))

                            if (len(date_list_file_i) > 0 and len(date_list_file_j) > 0):
                                date_list_file_i = self.check_header(date_list_file_i)
                                date_list_file_j = self.check_header(date_list_file_j)
                                # check for duplicate days
                                elements_i_to_delete = self.checkIfDuplicates(date_list_file_i)
                                elements_j_to_delete = self.checkIfDuplicates(date_list_file_j)

                                # refress date_list without repeated days
                                if len(elements_i_to_delete) > 0:
                                    index_set = set(elements_i_to_delete)
                                    date_list_file_i_common = [x for i, x in enumerate(date_list_file_i) if
                                                               i not in index_set]
                                else:
                                    date_list_file_i_common = date_list_file_i

                                if len(elements_j_to_delete) > 0:
                                    index_set = set(elements_j_to_delete)
                                    date_list_file_j_common = [x for i, x in enumerate(date_list_file_j) if
                                                               i not in index_set]
                                else:
                                    date_list_file_j_common = date_list_file_j

                                # eliminate non common days

                                common_dates_list = [value for value in date_list_file_i_common if
                                                     value in date_list_file_j_common]

                                for date_i in date_list_file_i:
                                    if not date_i in common_dates_list:
                                        print("Delete day: " + str(date_i) + " from " + file_i)
                                        elements_i_to_delete.append(date_list_file_i.index(date_i))

                                if len(elements_i_to_delete) > 0:
                                    data_matrix_file_i_corr = np.delete(data_matrix_file_i_corr,
                                                                        elements_i_to_delete, 1)

                                for date_j in date_list_file_j:
                                    if not date_j in common_dates_list:
                                        print("Delete day: " + str(date_j) + " from " + file_j)
                                        elements_j_to_delete.append(date_list_file_j.index(date_j))

                                if len(elements_j_to_delete) > 0:
                                    data_matrix_file_j_corr = np.delete(data_matrix_file_j_corr, elements_j_to_delete,
                                                                        1)

                                # ###########
                                # Correlación: multiplicación de matrices elemento a elemento
                                # ###########

                                # introduce sort columns
                                common_dates_list, old_index, new_index = self.sort_dates(common_dates_list)
                                data_matrix_file_i_corr[:, [old_index], :] = data_matrix_file_i_corr[:, [new_index], :]
                                data_matrix_file_j_corr[:, [old_index], :] = data_matrix_file_j_corr[:, [new_index], :]

                                corr_ij_freq = data_matrix_file_i_corr * np.conj(data_matrix_file_j_corr)

                                # La matriz resultante se pasa al dominio del tiempo
                                # Se reserva el espacio para la matriz de correlaciones en el dominio del tiempo
                                size_1d = corr_ij_freq.shape[0]
                                size_2d = corr_ij_freq.shape[1]
                                size_2d_all = size_1d + size_2d

                                corr_ij_freq[np.isnan(corr_ij_freq)] = 0.0 + 0.0j
                                zero_vectors = np.all(corr_ij_freq == 0.0 + 0.0j, axis=2)
                                count_zero_vectors = np.sum(zero_vectors)
                                size_2d_all = size_2d_all - count_zero_vectors

                                if normalization == "PCC":
                                    corr_ij_time = np.real(np.fft.ifft(corr_ij_freq, axis=2))
                                    corr_ij_time = np.fft.ifftshift(corr_ij_time)
                                else:
                                    corr_ij_time = np.real(np.fft.irfft(corr_ij_freq, axis=2))
                                    corr_ij_time = np.fft.ifftshift(corr_ij_time)

                                # Crop in time domain 16/01/2025
                                pad_length = corr_ij_time.shape[2]

                                start_idx = (pad_length - cross_length) // 2
                                end_idx = start_idx + cross_length
                                corr_ij_time = corr_ij_time[:, :, start_idx:end_idx]

                                # save memory
                                if self.stack != "PWS":
                                    try:
                                        del data_matrix_file_i_corr
                                        del data_matrix_file_j_corr
                                        del corr_ij_freq
                                        gc.collect()
                                    except:
                                        pass
                                else:
                                    try:
                                        del data_matrix_file_i_corr
                                        del data_matrix_file_j_corr
                                        gc.collect()
                                    except:
                                        pass

                                if self.stack == "nrooth":
                                    corr_ij_time = (np.abs(corr_ij_time) ** (1 / self.power)) * np.sign(corr_ij_time)
                                    c_stack = np.sum(np.sum(corr_ij_time, axis=1), axis=0) / size_2d_all

                                elif self.stack == "Linear":
                                    # Stack: Linear stack
                                    c_stack = np.sum(np.sum(corr_ij_time, axis=1), axis=0) / size_2d_all

                                elif self.stack == "PWS":

                                    # estimate the analytic function and then the instantaneous phase matrix
                                    # analytic_signal = np.zeros((size_1d, size_2d, size_3d), dtype=np.complex64)
                                    if normalization == "CC":
                                        f, c, d = corr_ij_freq.shape
                                        dim_full = 2 * d - 1
                                        non_real_dim = (dim_full - d)
                                        c = np.zeros((f, c, non_real_dim), dtype=np.complex64)
                                        signal_rfft_mod = np.concatenate((corr_ij_freq, c), axis=2)
                                        signal_rfft_mod[((non_real_dim // 2) + 1):] = signal_rfft_mod[
                                                                                      ((non_real_dim // 2) + 1):] * 0
                                        signal_rfft_mod[1:non_real_dim // 2] = 2 * signal_rfft_mod[1:non_real_dim // 2]

                                        # Generate the analytic function matrix
                                        analytic_signal = np.fft.ifft(signal_rfft_mod, axis=2)
                                        analytic_signal = np.fft.ifftshift(analytic_signal)

                                    # elif normalization != "CC":
                                    #
                                    #     # Generate the analytic function matrix
                                    #     analytic_signal = np.fft.ifft(corr_ij_freq,  axis=2)
                                    #     analytic_signal = np.fft.ifftshift(analytic_signal)

                                    # Crop the analytic signal 16/01/2025
                                    analytic_signal = analytic_signal[:, :, start_idx:end_idx]

                                    # Compute linear stack
                                    c_stack = np.sum(np.sum(corr_ij_time, axis=1), axis=0) / size_2d_all
                                    c_stack_max = np.max(c_stack)

                                    phase_stack = np.sum(np.sum(analytic_signal, axis=1), axis=0) / size_2d_all

                                    # this point proceed to the PWS
                                    phase_stack = (np.abs(phase_stack)) ** self.power
                                    c_stack = c_stack * phase_stack
                                    c_stack = (c_stack * c_stack_max) / np.max(c_stack)

                                # c_stack par, impar ...
                                # num = len(c_stack)
                                # if (num % 2) == 0:
                                #
                                #     #print(“Thenumber is even”)
                                #     c = int(np.ceil(num / 2.) + 1)
                                # else:
                                #     #print(“The providednumber is odd”)
                                #     c = int(np.ceil((num + 1)/2))

                                # c_stack = np.roll(c_stack, c)
                                print("stack[" + str(i) + "," + str(j) + "]:")
                                # print(c_stack)

                                # Guardar fichero
                                # print(metadata_list_file_i)
                                # print(metadata_list_file_j)
                                stats = {}
                                x_station = len(file_i) - 3
                                y_station = len(file_j) - 3
                                stats['network'] = file_i[:2]
                                stats['station'] = file_i[2:x_station] + "_" + file_j[2:y_station]
                                stats['channel'] = file_i[-1] + file_j[-1]
                                stats['sampling_rate'] = self.sampling_rate
                                stats['npts'] = len(c_stack)

                                stats['mseed'] = {'dataquality': 'D', 'geodetic': [dist, bazim, azim],
                                                  'cross_channels': file_i[-1] + file_j[-1],
                                                  'coordinates': [lat_i, lon_i, lat_j, lon_j]}
                                stats['starttime'] = UTCDateTime("2000-01-01T00:00:00.0")
                                # stats['info'] = {'geodetic': [dist, bazim, azim],'cross_channels':file_i[-1]+file_j[-1]}
                                st = Stream([Trace(data=c_stack, header=stats)])
                                # Nombre del fichero = XT.STA1_STA2.BHZE
                                filename = file_i[:2] + "." + file_i[2:x_station] + "_" + file_j[2:y_station] + "." + \
                                           file_i[-1] + file_j[
                                               -1]
                                path_name = os.path.join(self.stack_files_path, filename)
                                print(path_name)
                                st.write(path_name, format='H5')

                            if self.dailyStacks:
                                path_name = os.path.join(self.stack_daily_files_path, filename + "_daily")
                                dims = [corr_ij_time.shape[1], corr_ij_time.shape[2]]
                                stack_day = np.sum(corr_ij_time, axis=0)

                                # here I can delete corr_ij_time
                                del corr_ij_time
                                gc.collect()

                                clock = clock_process(stack_day, stats, path_name, common_dates_list, dims)
                                clock.daily_stack_part(type=self.stack, power=self.power, overlap=self.overlap)

                            else:
                                print("Empty date_list.")
                            print("-----")

                        else:
                            del metadata_list_file_i
                            del metadata_list_file_j
                            gc.collect()
                            print("Excluded cross correlations for being out of maximum distance ", dist * 1E-3, "<",
                                  self.min_dist)
        except:
            print("Something went wrong at:", file_i)

    def list_directory(self, path):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(path):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

    def check_header(self, list_files):
        list_files_new = []
        check_elem = list_files[0]
        date_check = check_elem.split(".")

        if len(date_check[0]) == 4:
            for index, element in enumerate(list_files):
                check_elem = element.split(".")
                date = check_elem[1] + "." + check_elem[0]
                list_files_new.append(date)
        else:
            list_files_new = list_files

        return list_files_new

    def checkIfDuplicates(self, listOfElems):

        ''' Check if given list contains any duplicates '''
        # dupes = []

        elements_to_delete = []
        dupes = [item for item, count in collections.Counter(listOfElems).items() if count > 1]
        if len(dupes) > 0:
            for elements in dupes:
                indices = [i for i, x in enumerate(listOfElems) if x == elements]
                for index in indices:
                    elements_to_delete.append(index)
                #    elements_to_delete.append(listOfElems[index])

        return elements_to_delete

    def rotate_horizontals(self):
        # self.check_path()

        obsfiles = self.list_directory(self.stack_files_path)
        station_list = self.list_stations(self.stack_files_path)
        channel_check = ["EE", "EN", "NN", "NE"]
        matrix_data = {}

        for station_pair in station_list:

            def_rotated = {}
            info = station_pair.split("_")
            sta1 = info[0]
            sta2 = info[1]

            if sta1 != sta2:
                for file in obsfiles:

                    try:
                        st = read(file)
                        tr = st[0]
                        station_i = tr.stats.station

                        chn = tr.stats.mseed['cross_channels']
                        # tr.stats['mseed']

                        if station_i == station_pair and chn in channel_check:
                            data = tr.data
                            matrix_data["net"] = tr.stats.network
                            matrix_data[chn] = data
                            matrix_data['geodetic'] = tr.stats.mseed['geodetic']
                            matrix_data['coordinates'] = tr.stats.mseed['coordinates']
                            matrix_data["sampling_rate"] = tr.stats.sampling_rate

                            # method to rotate the dictionary
                    except:
                        pass

            def_rotated["rotated_matrix"] = self.__rotate(matrix_data)

            if len(matrix_data) > 0 and def_rotated["rotated_matrix"] is not None:
                def_rotated["geodetic"] = matrix_data['geodetic']
                def_rotated["net"] = matrix_data["net"]
                def_rotated["station_pair"] = station_pair
                def_rotated['sampling_rate'] = matrix_data["sampling_rate"]
                def_rotated['coordinates'] = matrix_data["coordinates"]
                print(station_pair, "rotated")
                self.save_rotated(def_rotated)
                print(station_pair, "saved")

    def rotate_specific_daily(self):

        obsfiles = self.list_directory(self.stack_daily_files_path)
        station_list = self.list_stations_daily(self.stack_daily_files_path)
        channel_check = ["EE", "EN", "NN", "NE"]
        matrix_data = {}

        for station_pair in station_list:

            def_rotated = {}
            info = station_pair.split("_")
            sta1 = info[0]
            sta2 = info[1]

            if sta1 != sta2:
                for file in obsfiles:
                    try:
                        file_pickle = pickle.load(open(file, "rb"))
                        st = file_pickle["stream"]

                        # just for checking
                        station_i = st[0].stats.station
                        chn = st[0].stats.mseed['cross_channels']
                        net = st[0].stats.network
                        geodetic = st[0].stats.mseed['geodetic']
                        location = st[0].stats.location
                        coordinates = st[0].stats.mseed['coordinates']
                        fs = st[0].stats.sampling_rate
                        if station_i == station_pair and chn in channel_check:
                            dates = file_pickle["dates"]
                            matrix_data["net"] = net
                            matrix_data['geodetic'] = geodetic
                            matrix_data["sampling_rate"] = fs
                            matrix_data["location"] = location
                            matrix_data["dates"] = dates
                            matrix_data["coordinates"] = coordinates
                            data = []
                            for tr in st:
                                # loop to take data from all traces
                                data.append(tr.data)
                            matrix_data[chn] = data

                            # method to rotate the dictionary
                    except:
                        pass

                def_rotated["rotated_matrix"] = self.__rotate_specific(matrix_data)

                try:
                    if len(matrix_data) > 0 and len(def_rotated["rotated_matrix"]) > 0:
                        def_rotated["geodetic"] = matrix_data['geodetic']
                        def_rotated["net"] = matrix_data["net"]
                        def_rotated["station_pair"] = station_pair
                        def_rotated['sampling_rate'] = matrix_data["sampling_rate"]
                        def_rotated['location'] = matrix_data['location']
                        def_rotated['dates'] = matrix_data['dates']
                        def_rotated['coordinates'] = matrix_data['coordinates']
                        print(station_pair, "rotated")

                        self.save_rotated_specific(def_rotated)

                        print(station_pair, "saved")
                except:
                    print("Coudn't save ", station_pair)

    def __validation(self, data_matrix, specific=False):

        channel_check = ["EE", "EN", "NN", "NE"]
        check1 = False
        check2 = True
        check = False
        dims = []

        for chn in channel_check:
            if chn in data_matrix:
                check1 = True
                if specific:
                    dims.append(len(data_matrix[chn][0]))
                else:
                    dims.append(len(data_matrix[chn]))
            else:
                check1 = False

        try:
            ele = dims[0]
            for item in dims:
                if ele != item:
                    check2 = False
                    break
        except:
            check2 = False

        if check1 and check2:
            check = True

        return check, dims

    def __rotate(self, data_matrix):

        rotated = None

        validation, dim = self.__validation((data_matrix))

        if validation:
            data_array_ne = np.zeros((dim[0], 4, 1))

            data_array_ne[:, 0, 0] = data_matrix["EE"][:]
            data_array_ne[:, 1, 0] = data_matrix["EN"][:]
            data_array_ne[:, 2, 0] = data_matrix["NN"][:]
            data_array_ne[:, 3, 0] = data_matrix["NE"][:]

            rotate_matrix = self.__generate_matrix_rotate(data_matrix['geodetic'], dim)

            rotated = np.matmul(rotate_matrix, data_array_ne)

        return rotated

    def __rotate_specific(self, data_matrix):

        rotated = None
        all_rotated = []
        validation, dim = self.__validation((data_matrix), specific=True)

        if validation:
            n = len(data_matrix["NN"])
            rotate_matrix = self.__generate_matrix_rotate(data_matrix['geodetic'], dim)

            for iter in range(n):
                data_array_ne = np.zeros((dim[0], 4, 1))
                data_array_ne[:, 0, 0] = data_matrix["EE"][iter][:]
                data_array_ne[:, 1, 0] = data_matrix["EN"][iter][:]
                data_array_ne[:, 2, 0] = data_matrix["NN"][iter][:]
                data_array_ne[:, 3, 0] = data_matrix["NE"][iter][:]

                rotated = np.matmul(rotate_matrix, data_array_ne)
                all_rotated.append(rotated)

        return all_rotated

    def __generate_matrix_rotate(self, geodetic, dim):

        baz = geodetic[1] * np.pi / 180
        az = geodetic[2] * np.pi / 180

        rotate_matrix = np.zeros((4, 4))
        rotate_matrix[0, 0] = -1 * np.cos(az) * np.cos(baz)
        rotate_matrix[0, 1] = np.cos(az) * np.sin(baz)
        rotate_matrix[0, 2] = -1 * np.sin(az) * np.sin(baz)
        rotate_matrix[0, 3] = np.sin(az) * np.cos(baz)

        rotate_matrix[1, 0] = -1 * np.sin(az) * np.sin(baz)
        rotate_matrix[1, 1] = -1 * np.sin(az) * np.cos(baz)
        rotate_matrix[1, 2] = -1 * np.cos(az) * np.cos(baz)
        rotate_matrix[1, 3] = -1 * np.cos(az) * np.sin(baz)

        rotate_matrix[2, 0] = -1 * np.cos(az) * np.sin(baz)
        rotate_matrix[2, 1] = -1 * np.cos(az) * np.cos(baz)
        rotate_matrix[2, 2] = np.sin(az) * np.cos(baz)
        rotate_matrix[2, 3] = np.sin(az) * np.sin(baz)

        rotate_matrix[3, 0] = -1 * np.sin(az) * np.cos(baz)
        rotate_matrix[3, 1] = np.sin(az) * np.sin(baz)
        rotate_matrix[3, 2] = np.cos(az) * np.sin(baz)
        rotate_matrix[3, 3] = -1 * np.cos(az) * np.cos(baz)

        rotate_matrix = np.repeat(rotate_matrix[np.newaxis, :, :], dim[0], axis=0)

        return rotate_matrix

    def list_stations(self, path):
        stations = []
        files = self.list_directory(path)
        for file in files:
            try:
                st = read(file)
                name = st[0].stats.station
                info = name.split("_")
                flip_name = info[1] + "_" + info[0]
                if name not in stations and flip_name not in stations and info[0] != info[1]:
                    stations.append(name)
            except:
                pass

        return stations

    def list_stations_daily(self, path):
        stations = []
        files = self.list_directory(path)
        for file in files:
            try:
                file_pickle = pickle.load(open(file, "rb"))
                st = file_pickle["stream"]
                name = st[0].stats.station
                info = name.split("_")
                flip_name = info[1] + "_" + info[0]
                if name not in stations and flip_name not in stations and info[0] != info[1]:
                    stations.append(name)
            except:
                pass

        return stations

    def __coords2azbazinc(self, station1_latitude, station1_longitude, station2_latitude,
                          station2_longitude):

        """
        Returns azimuth, backazimuth and incidence angle from station coordinates
        given in first trace of stream and from event location specified in origin
        dictionary.
        """

        dist, bazim, azim = gps2dist_azimuth(station1_latitude, station1_longitude, station2_latitude,
                                             station2_longitude)
        return dist, bazim, azim

    def info_extract_name(self, path):
        name = os.path.basename(path)
        info = name.split("_")
        list1 = info[0].split(".")
        list2 = info[1].split(".")
        net = list1[0]
        sta1 = list1[1]
        sta2 = list2[0]
        channels = list2[1]
        return net, sta1, sta2, channels

    def save_rotated(self, def_rotated):
        stats = {}
        channels = ["TT", "RR", "TR", "RT"]
        stats['network'] = def_rotated["net"]
        stats['station'] = def_rotated["station_pair"]
        stats['sampling_rate'] = def_rotated['sampling_rate']
        j = 0
        for chn in channels:
            stats['channel'] = chn
            stats['npts'] = len(def_rotated["rotated_matrix"][:, j, 0])
            stats['mseed'] = {'dataquality': 'D', 'geodetic': def_rotated["geodetic"],
                              'cross_channels': def_rotated["station_pair"], 'coordinates': def_rotated['coordinates']}
            stats['starttime'] = UTCDateTime("2000-01-01T00:00:00.0")
            # stats['info'] = {'geodetic': [dist, bazim, azim],'cross_channels':file_i[-1]+file_j[-1]}
            st = Stream([Trace(data=def_rotated["rotated_matrix"][:, j, 0], header=stats)])
            # Nombre del fichero = XT.STA1_STA2.BHZE
            filename = def_rotated["net"] + "." + def_rotated["station_pair"] + "." + chn
            path_name = os.path.join(self.stack_rotated_files_path, filename)
            print(path_name)
            st.write(path_name, format='H5')
            j = j + 1

    def save_rotated_specific(self, def_rotated):

        stats = {}
        channels = ["TT", "RR", "TR", "RT"]
        stats['network'] = def_rotated["net"]
        stats['station'] = def_rotated["station_pair"]
        stats['sampling_rate'] = def_rotated['sampling_rate']
        stats['location'] = def_rotated['location']

        for i, chn in enumerate(channels):
            stack_partial = []
            stats['channel'] = chn
            # stats['npts'] = len(def_rotated["rotated_matrix"][:, j, 0][0])
            stats['mseed'] = {'dataquality': 'D', 'geodetic': def_rotated["geodetic"],
                              'cross_channels': def_rotated["station_pair"], "coordinates": def_rotated['coordinates']}
            stats['starttime'] = UTCDateTime("2000-01-01T00:00:00.0")

            for iter in def_rotated["rotated_matrix"]:
                data = iter[:, i, 0]
                stack_partial.append(Trace(data=data, header=stats))

            st = Stream(stack_partial)
            # Nombre del fichero = XT.STA1_STA2.BHZE
            filename = def_rotated["net"] + "." + def_rotated["station_pair"] + "." + chn + "_" + "daily"
            path_name = os.path.join(self.stack_daily_files_path, filename)
            print(path_name)
            data_to_save = {"dates": def_rotated['dates'], "stream": st}

            file_to_store = open(path_name, "wb")
            pickle.dump(data_to_save, file_to_store)

        # j = 0
        # for chn in channels:
        #     stats['channel'] = chn
        #     stats['npts'] = len(def_rotated["rotated_matrix"][:, j, 0])
        #     stats['mseed'] = {'dataquality': 'D', 'geodetic': def_rotated["geodetic"],
        #                       'cross_channels': def_rotated["station_pair"]}
        #     stats['starttime'] = UTCDateTime("2000-01-01T00:00:00.0")
        #
        #     for data in def_rotated:
        #
        #     st = Stream([Trace(data=def_rotated["rotated_matrix"][:, j, 0], header=stats)])
        #     # Nombre del fichero = XT.STA1_STA2.BHZE
        #     filename = def_rotated["net"] + "." + def_rotated["station_pair"] + "." + chn
        #     path_name = os.path.join(self.stack_rotated_files_path, filename)
        #     print(path_name)
        #     st.write(path_name, format='H5')
        #     j = j + 1

    def sort_dates(self, common_date_list):
        # extract years
        years = {}
        all_years = []
        list_iterate = common_date_list
        for date in list_iterate:
            date = date.split(".")
            julday = date[0]
            year = date[1]
            if year not in years.keys():
                years[year] = [julday + "." + year]
            else:
                years[year].append(julday + "." + year)

        for keys in years:
            date_index = years[keys]
            date_index = sorted(date_index, key=float)
            all_years = all_years + date_index

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
