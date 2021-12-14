#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from configparser import ConfigParser

from obspy import Trace, Stream
from obspy.core import UTCDateTime
import numpy as np
import pickle
import os
from obspy import read
from obspy.geodetics import gps2dist_azimuth


class noisestack:

    def __init__(self, output_files_path):

        """
                Process ANT, Cross + Stack



                :param No params required to initialize the class
        """
        self.__metadata_manager = None
        self.output_files_path = output_files_path
        self.channel = ["BHZ", "BHN", "BHE"]

        self.year = 2000

    def check_path(self):
    # Se crea la carpeta si no existe:

        self.stack_files_path = os.path.join(self.output_files_path, "stack")
        self.stack_rotated_files_path = os.path.join(self.output_files_path, "stack_rotated")

        if not os.path.exists(self.stack_files_path):
           os.makedirs(self.stack_files_path)

        if not os.path.exists(self.stack_rotated_files_path):

            os.makedirs(self.stack_rotated_files_path)



    # Ficheros de datos
        #self.pickle_files = [pickle_file for pickle_file in os.listdir(self.output_files_path) if self.channel in pickle_file]
        self.pickle_files = []
        for pickle_file in os.listdir(self.output_files_path):
            for jj in range(len(self.channel)):
                if self.channel[jj] in pickle_file:
                    self.pickle_files.append(pickle_file)
        print(self.pickle_files)
    # Para cada pareja de ficheros, se cargan los ficheros y se multiplican las matrices de datos que contienen, sólo en los días comunes
    # Indices i,j: se refieren a ficheros de datos file_i, file_j que contiene las matrices que se multiplicarán.

    # Sólo hacer si i>= j, para hacer sólo triangular superior
    # y reducir el número de operaciones
    def run_cross_stack(self):
        self.check_path()
        for i, file_i in enumerate(self.pickle_files):
            if file_i[-1] in ["N", "E", "X", "Y", "1","2"]:
                key1_i = "data_matrix" + "_" + file_i[-1]
                key2_i = 'metadata_list' + "_" + file_i[-1]
                key3_i = 'date_list' + "_" + file_i[-1]
            else:
                key1_i = "data_matrix"
                key2_i = 'metadata_list'
                key3_i = 'date_list'

            for j, file_j in enumerate(self.pickle_files):

                if file_j[-1] in ["N", "E", "X", "Y", "1", "2"]:
                    key1_j = "data_matrix" + "_" + file_j[-1]
                    key2_j = 'metadata_list' + "_" + file_j[-1]
                    key3_j = 'date_list' + "_" + file_j[-1]
                else:
                    key1_j = "data_matrix"
                    key2_j = 'metadata_list'
                    key3_j = 'date_list'

                print("(i=" + str(i) + ",j=" + str(j) + ") -> (" + file_i + "," + file_j + ")")

                with open(os.path.join(self.output_files_path, file_i), 'rb') as h_i, open(os.path.join(self.output_files_path, file_j),
                                                                                      'rb') as h_j:

                    # Cada fichero file_i y file_i contiene:
                    # dict_matrix ={ 'data_matrix': [] , 'metadata_list': [], 'date_list': []}
                    dict_matrix_file_i = pickle.load(h_i)
                    dict_matrix_file_j = pickle.load(h_j)

                    data_matrix_file_i = dict_matrix_file_i[key1_i]
                    data_matrix_file_j = dict_matrix_file_j[key1_j]
                    metadata_list_file_i = dict_matrix_file_i[key2_i]
                    metadata_list_file_j = dict_matrix_file_j[key2_j]

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

                    date_list_file_i = dict_matrix_file_i[key3_i]
                    date_list_file_j = dict_matrix_file_j[key3_j]

                    # Lista de días de cada fichero
                    print("dict_matrix_file_i['date_list']: " + str(date_list_file_i))
                    print("dict_matrix_file_j['date_list']: " + str(date_list_file_j))

                    # Lista de días comunes a las dos matrices que se van a multiplicar
                    # Antes de multiplicarl, se eliminarán de ellas las columnas correspondientes a los días que no aparecen en esta lista
                    common_dates_list = [value for value in date_list_file_i if value in date_list_file_j]

                    if (len(date_list_file_i) > 0 and len(date_list_file_j) > 0):

                        # Sufijo _corr: para matrices que se multiplicarán después de eliminar días no comunes a la matriz original
                        data_matrix_file_i_corr = data_matrix_file_i
                        data_matrix_file_j_corr = data_matrix_file_j

                        # Se eliminan de las matrices los días no comunes con numpy.delete()
                        # >numpy.delete(arr, obj, axis=None)
                        # >arr refers to the input array,
                        # >obj refers to which sub-arrays (e.g. column/row no. or slice of the array) and
                        # >axis refers to either column wise (axis = 1) or row-wise (axis = 0) delete operation
                        elements_i_to_delete = []
                        elements_j_to_delete = []

                        for date_i in date_list_file_i:
                            if (not date_i in common_dates_list):
                                print("Delete day: " + str(date_i) + " from " + file_i)
                                elements_i_to_delete.append(date_list_file_i.index(date_i))

                        if len(elements_i_to_delete)>0:

                            data_matrix_file_i_corr = np.delete(data_matrix_file_i, elements_i_to_delete, 1)

                        for date_j in date_list_file_j:
                            if (not date_j in common_dates_list):
                                print("Delete day: " + str(date_j) + " from " + file_j)
                                elements_j_to_delete.append(date_list_file_j.index(date_j))

                        if len(elements_j_to_delete) > 0:
                            data_matrix_file_j_corr = np.delete(data_matrix_file_j, elements_j_to_delete, 1)

                        # ###########
                        # Correlación: multiplicación de matrices elemento a elemento
                        # ###########
                        #if j >= i:
                        corr_ij_freq = data_matrix_file_i_corr * np.conj(data_matrix_file_j_corr)

                        # La matriz resultante se pasa al dominio del tiempo
                        # Se reserva el espacio para la matriz de correlaciones en el dominio del tiempo
                        size_1d = corr_ij_freq.shape[0]
                        size_2d = corr_ij_freq.shape[1]
                        # 7-7-2021, importante 2n - 1
                        size_3d = 2 * corr_ij_freq.shape[2] - 1

                        corr_ij_time = np.zeros((size_1d, size_2d, size_3d), dtype=np.float64)

                        # Se rellena la matriz en el dominio del tiempo
                        for m in range(corr_ij_freq.shape[0]):
                            for n in range(corr_ij_freq.shape[1]):
                                # irfft para pasar al dominio del tiempo

                                corr_ij_time[m, n, :] = np.fft.irfft(corr_ij_freq[m, n, :], size_3d)

                        # Stack: se suman los intervalos de la matriz
                        c_stack = np.zeros(size_3d, dtype=np.float64)
                        for m in range(corr_ij_time.shape[0]):
                            for n in range(corr_ij_time.shape[1]):
                                c_stack = c_stack + corr_ij_time[m, n, :]
                        # c_stack par, impar ...
                        c_stack = np.roll(c_stack, int(len(c_stack) / 2))
                        print("stack[" + str(i) + "," + str(j) + "]:")
                        print(c_stack)

                        # Guardar fichero
                        # print(metadata_list_file_i)
                        # print(metadata_list_file_j)
                        stats = {}
                        stats['network'] = file_i[:2]
                        stats['station'] = file_i[2:6] + "_" + file_j[2:6]
                        stats['channel'] = file_i[-1]
                        stats['sampling_rate'] = self.sampling_rate
                        stats['npts'] = len(c_stack)
                        stats['mseed'] = {'dataquality': 'D', 'geodetic': [dist, bazim, azim], 'cross_channels':file_i[-1]+file_j[-1],
                                          'coordinates': [lat_i, lon_i, lat_j, lon_j]}
                        stats['starttime'] = UTCDateTime("2000-01-01T00:00:00.0")
                        #stats['info'] = {'geodetic': [dist, bazim, azim],'cross_channels':file_i[-1]+file_j[-1]}
                        st = Stream([Trace(data=c_stack, header=stats)])
                        # Nombre del fichero = XT.STA1_STA2.BHZE
                        filename = file_i[:2] + "." + file_i[2:6] + "_" + file_j[2:6] + "." + file_i[-1]+file_j[-1]
                        path_name = os.path.join(self.stack_files_path, filename)
                        print(path_name)
                        st.write(path_name, format='H5')
                        #

                    else:
                        print("Empty date_list.")
                    print("-----")



    def list_directory(self, path):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(path):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles


    def rotate_horizontals(self):

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
                        st  = read(file)
                        tr = st[0]
                        station_i = tr.stats.station

                        chn = tr.stats.mseed['cross_channels']

                        if station_i == station_pair and chn in channel_check :

                            data = tr.data
                            matrix_data["net"] = tr.stats.network
                            matrix_data[chn] = data
                            matrix_data['geodetic'] = tr.stats.mseed['geodetic']
                            # method to rotate the dictionary
                    except:
                        pass

            def_rotated["rotated_matrix"] = self.__rotate(matrix_data)

            if len(matrix_data) > 0 and def_rotated["rotated_matrix"] is not None:

                def_rotated["geodetic"] = matrix_data['geodetic']
                def_rotated["net"] = matrix_data["net"]
                def_rotated["station_pair"] = station_pair
                print(station_pair, "rotated")
                self.save_rotated(def_rotated)
                print(station_pair, "saved")


            station_list.remove(station_pair)

    def __validation(self, data_matrix):

        channel_check = ["EE", "EN", "NN", "NE"]
        check1 = False
        check2 = True
        check = False
        dims = []

        for j in channel_check:
            if j in data_matrix:
                check1 = True
                dims.append(len(data_matrix[j]))
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


    def __generate_matrix_rotate(self, geodetic, dim):

        baz = geodetic[1]*np.pi/180
        az = geodetic[2]*np.pi/180

        rotate_matrix = np.zeros((4,4))
        rotate_matrix[0, 0] = -1*np.cos(az)*np.cos(baz)
        rotate_matrix[0, 1] = np.cos(az)*np.sin(baz)
        rotate_matrix[0, 2] = -1*np.sin(az)*np.sin(baz)
        rotate_matrix[0, 3] = np.sin(az)*np.cos(baz)

        rotate_matrix[1, 0] = -1*np.sin(az)*np.sin(baz)
        rotate_matrix[1, 1] = -1*np.sin(az)*np.cos(baz)
        rotate_matrix[1, 2] = -1*np.cos(az)*np.cos(baz)
        rotate_matrix[1, 3] = -1*np.cos(az)*np.sin(baz)

        rotate_matrix[2, 0] = -1 * np.cos(az)*np.sin(baz)
        rotate_matrix[2, 1] = -1 * np.cos(az)*np.cos(baz)
        rotate_matrix[2, 2] = np.sin(az)*np.cos(baz)
        rotate_matrix[2, 3] = np.sin(az)*np.sin(baz)

        rotate_matrix[3, 0] = -1 * np.sin(az) * np.cos(baz)
        rotate_matrix[3, 1] = np.sin(az) * np.sin(baz)
        rotate_matrix[3, 2] = np.cos(az) * np.sin(baz)
        rotate_matrix[3, 3] = -1 * np.cos(az) * np.cos(baz)

        rotate_matrix = np.repeat(rotate_matrix[np.newaxis, :, :], dim[0], axis=0)

        return rotate_matrix


    def list_stations(self, path):
        stations =[]
        files = self.list_directory(path)
        for file in files:
            try:
                st = read(file)
                if st[0].stats.station not in stations:
                    stations.append(st[0].stats.station)
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
        return net,sta1,sta2,channels

    def save_rotated(self, def_rotated):
        stats = {}
        channels = ["TT", "RR", "TR", "RT"]
        stats['network'] = def_rotated["net"]
        stats['station'] = def_rotated["station_pair"]
        stats['sampling_rate'] = self.sampling_rate

        j = 0
        for chn in channels:

            stats['channel'] = chn
            stats['npts'] = len(def_rotated["rotated_matrix"][:,j,0])
            stats['mseed'] = {'dataquality': 'D', 'geodetic': def_rotated["geodetic"],
                              'cross_channels': def_rotated["station_pair"]}
            stats['starttime'] = UTCDateTime("2000-01-01T00:00:00.0")
            # stats['info'] = {'geodetic': [dist, bazim, azim],'cross_channels':file_i[-1]+file_j[-1]}
            st = Stream([Trace(data=def_rotated["rotated_matrix"][:,j,0], header=stats)])
            # Nombre del fichero = XT.STA1_STA2.BHZE
            filename = def_rotated["net"] + "." + def_rotated["station_pair"] + "." + chn
            path_name = os.path.join(self.stack_rotated_files_path, filename)
            print(path_name)
            st.write(path_name, format='H5')
            j = j+1