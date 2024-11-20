import os
import shutil
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth
import time
from isp import GREEN_SOURCE, GREEN, ROOT_DIR


class MTIManager:

    def __init__(self, st, inv, lat0, lon0, min_dist, max_dist):
        """
        Manage MTI files for run isola class program.
        st: stream of seismograms
        in: inventory
        """
        self.__st = st
        self.__inv = inv
        self.lat = lat0
        self.lon = lon0
        self.min_dist = min_dist
        self.max_dist = max_dist

    @staticmethod
    def __validate_file(file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError("The file {} doesn't exist.".format(file_path))

    @staticmethod
    def __validate_dir(dir_path):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(dir_path))

    @property
    def root_path(self):
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.__validate_dir(root_path)
        return root_path

    @property
    def get_stations_dir(self):
        stations_dir = os.path.join(self.root_path, "input")
        self.__validate_dir(stations_dir)
        return stations_dir

    # @staticmethod
    # def copy_and_clean_folder():
    #     """
    #     Copies all files from the source folder to the destination folder.
    #     Cleans the destination folder before copying.
    #
    #     :param source_folder: Path to the source folder
    #     :param destination_folder: Path to the destination folder
    #     """
    #     try:
    #         # Ensure both folders exist
    #         if not os.path.exists(GREEN_SOURCE):
    #             raise FileNotFoundError(f"Source folder does not exist: {GREEN_SOURCE}")
    #
    #         if not os.path.exists(GREEN):
    #             os.makedirs(GREEN)
    #
    #         # Clean the destination folder
    #         for item in os.listdir(GREEN):
    #             item_path = os.path.join(GREEN, item)
    #             if os.path.isfile(item_path) or os.path.islink(item_path):
    #                 os.unlink(item_path)  # Remove file or symbolic link
    #             elif os.path.isdir(item_path):
    #                 shutil.rmtree(item_path)  # Remove directory and contents
    #
    #         # Copy files from source to destination
    #         for file_name in os.listdir(GREEN):
    #             source_file_path = os.path.join(GREEN, file_name)
    #             destination_file_path = os.path.join(GREEN, file_name)
    #
    #             if os.path.isfile(source_file_path):  # Only copy files, not directories
    #                 shutil.copy2(source_file_path, destination_file_path)
    #         print("Waiting for 10 seconds before starting the copy process...")
    #
    #         time.sleep(10)  # Wait for 10 seconds
    #         print(f"All files from '{GREEN_SOURCE}' have been copied to '{GREEN}' successfully.")
    #     except Exception as e:
    #         print(f"An error occurred: {e}")


    @staticmethod
    def move_files(destination_folder):
        """
        Moves all files from the source_folder to the destination_folder.
        Creates the destination_folder if it doesn't exist.
        """

        source_folder = os.path.join(ROOT_DIR, 'mti/output/')

        if not os.path.exists(source_folder):
            print(f"Source folder '{source_folder}' does not exist.")
            return

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for filename in os.listdir(source_folder):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            # Check if it's a file
            if os.path.isfile(source_path):
                try:
                    shutil.move(source_path, destination_path)
                    print(f"Moved: {source_path} -> {destination_path}")
                except Exception as e:
                    print(f"Error moving file {filename}: {e}")
    @staticmethod
    def clean_and_create_symlinks():
        """
        Cleans the destination folder and creates symbolic links for all files in the source folder.

        Args:
            destination_folder (str): Path to the destination folder.
            source_folder (str): Path to the source folder.

        Raises:
            ValueError: If source_folder does not exist.
        """
        if not os.path.exists(GREEN_SOURCE):
            raise ValueError(f"The source folder '{GREEN_SOURCE}' does not exist.")
        if not os.path.exists(GREEN):
            os.makedirs(GREEN)  # Create destination folder if it doesn't exist

        # Clean the destination folder
        for item in os.listdir(GREEN):
            item_path = os.path.join(GREEN, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove files or symbolic links
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directories

        # Create symbolic links for all files in the source folder
        for item in os.listdir(GREEN_SOURCE):
            source_path = os.path.join(GREEN_SOURCE, item)
            destination_path = os.path.join(GREEN, item)
            if os.path.isfile(source_path):
                os.symlink(source_path, destination_path)  # Create symbolic link

        print(f"Symbolic links created for all files in '{GREEN_SOURCE}' to '{GREEN}'.")
    def get_stations_index(self):

        ind = []
        file_list = []
        dist1 = []
        for tr in self.__st:
            net = tr.stats.network
            station = tr.stats.station
            channel = tr.stats.channel
            coords = self.__inv.get_coordinates(tr.id)
            lat = coords['latitude']
            lon = coords['longitude']
            if ind.count(station):
                pass
            else:
                [dist, _, _] = gps2dist_azimuth(self.lat, self.lon, lat, lon, a=6378137.0, f=0.0033528106647474805)
                ind.append(station)
                item = '{net}:{station}::{channel}    {lat}    {lon}'.format(net=net,
                        station=station, channel=channel[0:2], lat=lat, lon=lon)

            # filter by distance
                if self.min_dist and self.max_dist > 0:
                    # do the distance filter
                    if dist >= self.min_dist:
                        file_list.append(item)
                        dist1.append(dist)
                        keydict = dict(zip(file_list, dist1))
                        file_list.sort(key=keydict.get)
                    if dist <= self.max_dist:
                        file_list.append(item)
                        dist1.append(dist)
                        keydict = dict(zip(file_list, dist1))
                        file_list.sort(key=keydict.get)
                # do not filter by distance
                else:
                    file_list.append(item)
                    dist1.append(dist)
                    keydict = dict(zip(file_list, dist1))
                    file_list.sort(key=keydict.get)

        self.stations_index = ind
        self.stream = self.sort_stream(dist1)


        deltas = self.get_deltas()

        data = {'item': file_list}

        df = pd.DataFrame(data, columns=['item'])
        #print(df)
        outstations_path = os.path.join(self.get_stations_dir, "stations.txt")
        #print(outstations_path)
        df.to_csv(outstations_path, header=False, index=False)
        return self.stream , deltas, outstations_path


    def sort_stream(self, dist1):
        stream = []
        stream_sorted_order = []

        for station in self.stations_index:
            st2 = self.__st.select(station=station)
            stream.append(st2)

        # Sort by Distance

        stream_sorted = [x for _, x in sorted(zip(dist1, stream))]
        # reverse from E N Z --> Z N E
        # reverse from 1 2 Z --> Z 2 1

        for stream_sort in stream_sorted:
            stream_sorted_order.append(stream_sort.reverse())

        return stream_sorted_order

    def get_deltas(self):
        deltas = []
        n =len(self.stream)
        for j in range(n):
            stream_unique = self.stream[j]
            delta_unique = stream_unique[0].stats.delta
            deltas.append(delta_unique)

        return deltas




