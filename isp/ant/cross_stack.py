#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from configparser import ConfigParser

from obspy import Trace, Stream
from obspy.core import UTCDateTime
import numpy as np
import pickle
import os


class cross_stack:

    def __init__(self, output_files_path, stack_files_path):

        """
                Process ANT, Cross + Stack



                :param No params required to initialize the class
        """

        self.output_files_path = output_files_path
        self.stack_files_path = stack_files_path
        self.channel = "BHZ, BHN, BHE"
        self.year = 2000
        check_nan = True

    # Se crea la carpeta si no existe:
    #if not os.path.exists(stack_files_path):
    #    os.makedirs(stack_files_path)

    # Ficheros de datos
        pickle_files = [pickle_file for pickle_file in os.listdir(self.output_files_path) if self.channel in pickle_file]

    # Para cada pareja de ficheros, se cargan los ficheros y se multiplican las matrices de datos que contienen, sólo en los días comunes
    # Indices i,j: se refieren a ficheros de datos file_i, file_j que contiene las matrices que se multiplicarán.

    # Sólo hacer si i>= j, para hacer sólo triangular superior
    # y reducir el número de operaciones
    def run_cross_stack(self):

        for i, file_i in enumerate(self.pickle_files):

            for j, file_j in enumerate(self.pickle_files):

                print("(i=" + str(i) + ",j=" + str(j) + ") -> (" + file_i + "," + file_j + ")")

                with open(os.path.join(self.output_files_path, file_i), 'rb') as h_i, open(os.path.join(self.output_files_path, file_j),
                                                                                      'rb') as h_j:

                    # Cada fichero file_i y file_i contiene:
                    # dict_matrix ={ 'data_matrix': [] , 'metadata_list': [], 'date_list': []}
                    dict_matrix_file_i = pickle.load(h_i)
                    dict_matrix_file_j = pickle.load(h_j)
                    data_matrix_file_i = dict_matrix_file_i['data_matrix']
                    data_matrix_file_j = dict_matrix_file_j['data_matrix']
                    metadata_list_file_i = dict_matrix_file_i['metadata_list']
                    metadata_list_file_j = dict_matrix_file_j['metadata_list']
                    date_list_file_i = dict_matrix_file_i['date_list']
                    date_list_file_j = dict_matrix_file_j['date_list']

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

                        for date_i in date_list_file_i:
                            if (not date_i in common_dates_list):
                                print("Delete day: " + str(date_i) + " from " + file_i)
                                data_matrix_file_i_corr = np.delete(data_matrix_file_i, date_list_file_i.index(date_i), 1)

                        for date_j in date_list_file_j:
                            if (not date_j in common_dates_list):
                                print("Delete day: " + str(date_j) + " from " + file_j)
                                data_matrix_file_j_corr = np.delete(data_matrix_file_j, date_list_file_j.index(date_j), 1)

                        # ###########
                        # Correlación: multiplicación de matrices elemento a elemento
                        # ###########
                        if j >= i:
                            corr_ij_freq = data_matrix_file_i_corr * np.conj(data_matrix_file_j_corr)

                            # La matriz resultante se pasa al dominio del tiempo
                            # Se reserva el espacio para la matriz de correlaciones en el dominio del tiempo
                            size_1d = corr_ij_freq.shape[0]
                            size_2d = corr_ij_freq.shape[1]
                            # 7-7-2021, importante 2n - 1
                            size_3d = 2 * corr_ij_freq.shape[2] - 1

                            corr_ij_time = np.zeros((size_1d, size_2d, size_3d), dtype=np.float64)
                            corr_ij_time = np.zeros((size_1d, size_2d, size_3d), dtype=np.float64)

                            # Se rellena la matriz en el dominio del tiempo
                            for m in range(corr_ij_freq.shape[0]):
                                for n in range(corr_ij_freq.shape[1]):
                                    # irfft para pasar al dominio del tiempo

                                    corr_ij_time[m, n, :] = np.fft.irfft(corr_ij_freq[m, n, :], size_3d)

                            # Stack: se suman los intervalos de la matriz
                            c_stack = np.zeros(size_3d, dtype=np.float64)  # y si hubo reshape?
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
                            stats['channel'] = self.channel
                            stats['sampling_rate'] = 5.0
                            stats['npts'] = len(c_stack)
                            stats['mseed'] = {'dataquality': 'D'}
                            stats['starttime'] = UTCDateTime(year=self.year, julday=common_dates_list[0], hour=0, minute=0)
                            st = Stream([Trace(data=c_stack, header=stats)])
                            # Nombre del fichero = XT.STA1_STA2.BHZ
                            filename = file_i[:2] + "." + file_i[2:6] + "_" + file_j[2:6] + "." + self.channel
                            st.write(os.path.join(self.stack_files_path, filename), format='MSEED')

                    else:
                        print("Empty date_list.")
                    print("-----")


