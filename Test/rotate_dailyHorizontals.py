import collections
import os
import pickle

from obspy import read, UTCDateTime, Stream, Trace
import numpy as np
from obspy.geodetics import gps2dist_azimuth


class RotataDailyHorizontals:
    def __init__(self, stack_files_path, stack_rotated_files_path):
        self.stack_files_path = stack_files_path
        self.stack_rotated_files_path = stack_rotated_files_path

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
                list_st = []

                for file in obsfiles:

                    try:
                        st, dates = self.read_daily(file)
                        list_st.append([st, dates])
                        tr = st[0]
                        station_i = tr.stats.station

                        chn = tr.stats.mseed['cross_channels']

                        if station_i == station_pair and chn in channel_check:
                            data = []

                            for j in range(len(st)):
                                tr1 = st[j]
                                data.append([tr1.data])
                            matrix_data["net"] = tr.stats.network
                            matrix_data[chn] = data
                            matrix_data['geodetic'] = tr.stats.mseed['geodetic']
                            matrix_data["sampling_rate"] = tr.stats.sampling_rate
                            # method to rotate the dictionary
                    except:
                        pass
            stacked_days = len(st)
            def_rotated["rotated_matrix"] = self.__rotate(matrix_data, stacked_days)

            if len(matrix_data) > 0 and def_rotated["rotated_matrix"] is not None:
                def_rotated["geodetic"] = matrix_data['geodetic']
                def_rotated["net"] = matrix_data["net"]
                def_rotated["station_pair"] = station_pair
                def_rotated['sampling_rate'] = matrix_data["sampling_rate"]
                print(station_pair, "rotated")
                self.save_rotated_new(def_rotated, list_st)
                #print(station_pair, "saved")


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


    def __rotate(self, data_matrix, stacked_days):
        rotated = None
        rotated_complete = None
        data_matrix_check = data_matrix.copy()
        try:
            data_matrix_check["EE"] = data_matrix["EE"][0][0]
            data_matrix_check["EN"] = data_matrix["EN"][0][0]
            data_matrix_check["NE"] = data_matrix["NE"][0][0]
            data_matrix_check["NN"] = data_matrix["NN"][0][0]
            validation, dim = self.__validation((data_matrix_check))
        except:
            validation = False

        if validation:
            # Here we make the loop
            rotated_complete = []
            for day in range(stacked_days):
                data_array_ne = np.zeros((dim[0], 4, 1))
                data_array_ne[:, 0, 0] = data_matrix["EE"][day][0][:]
                data_array_ne[:, 1, 0] = data_matrix["EN"][day][0][:]
                data_array_ne[:, 2, 0] = data_matrix["NN"][day][0][:]
                data_array_ne[:, 3, 0] = data_matrix["NE"][day][0][:]

                rotate_matrix = self.__generate_matrix_rotate(data_matrix['geodetic'], dim)

                rotated = np.matmul(rotate_matrix, data_array_ne)
                rotated_complete.append(rotated)
        return rotated_complete


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
                st, _ = self.read_daily(file)
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

    def save_rotated_new(self, def_rotated, list_st):
        print("saving")
        for item in list_st:
            date = item[1]
            st = item[0]
            network = st[0].stats.network
            channel = st[0].stats.channel
            stations = st[0].stats.station

            if channel == "EE":
                channel_new = "TT"
                for index, trace in enumerate(st):
                    st[index].data = def_rotated["rotated_matrix"][index][:, 0, 0]
                    st[index].stats.channel = channel_new
            elif channel == "EN":
                channel_new = "RR"
                for index, trace in enumerate(st):
                    st[index].data = def_rotated["rotated_matrix"][index][:, 1, 0]
                    st[index].stats.channel = channel_new
            elif channel == "NN":
                channel_new = "TR"
                for index, trace in enumerate(st):
                    st[index].data = def_rotated["rotated_matrix"][index][:, 2, 0]
                    st[index].stats.channel = channel_new
            elif channel == "NE":
                channel_new = "RT"
                for index, trace in enumerate(st):
                    st[index].data = def_rotated["rotated_matrix"][index][:, 3, 0]
                    st[index].stats.channel = channel_new

            # saving
            data_to_save = {"dates": date, "stream": st}
            name = network+"."+stations+"."+channel_new+"_"+"daily"
            path_name = os.path.join(self.stack_rotated_files_path,name)
            file_to_store = open(path_name, "wb")
            pickle.dump(data_to_save, file_to_store)


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
                date = check_elem[1]+"."+check_elem[0]
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

    def read_daily(self, path_file):

        with open(path_file, 'rb') as handle:
            mapping = pickle.load(handle)

        dates = mapping["dates"]
        st = mapping["stream"]

        return st, dates

if __name__ == "__main__":
    #stack_path_files = "/Volumes/LaCie/UPFLOW_resample/EGFs_test/NEW/horizontals/stack"
    stack_path_files = "/Volumes/LaCie/UPFLOW_resample/EGFs_test/NEW/horizontals/daily"
    stack_rotated_files_path = "/Volumes/LaCie/UPFLOW_resample/EGFs_test/NEW/horizontals/daily_rotated"
    rth = RotataDailyHorizontals(stack_path_files, stack_rotated_files_path)
    rth.rotate_horizontals()