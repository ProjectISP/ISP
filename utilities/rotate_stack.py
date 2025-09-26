#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rotate_stack
"""

import os
import pickle
from obspy import read, UTCDateTime, Trace, Stream
import numpy as np
import re
from pathlib import Path

class RotateStack:

    def __init__(self, stack_files_path, stack_daily_files_path=None, stack_rotated_files_path=None, output_folder = None):
        self.stack_files_path = stack_files_path
        self.stack_daily_files_path = stack_daily_files_path
        self.stack_rotated_files_path = stack_rotated_files_path
        self.output_folder = output_folder

    def list_stations_daily(self, path):
        _DAILY_NAME_RE = re.compile(
            r"^[^.]+\.(?P<sta1>[A-Z0-9]+)_(?P<sta2>[A-Z0-9]+)\.[A-Z]{2}_daily(?:\.[^.]+)?$"
        )
        stations = []
        files = self.list_directory(path)
        for file in files:
            try:
                base = os.path.basename(file)
                if not (base.endswith("_daily") or "_daily." in base):
                    continue
                else:
                    file_name = os.path.basename(file)
                    name = file_name.split("_")
                    sta1 = name[0].split(".")[1]
                    sta2 = name[1].split(".")[0]
                    name = sta1 + "_" + sta2
                    flip_name = sta2 + "_" + sta1

                    if name not in stations and flip_name not in stations and sta1 != sta2:
                        stations.append(name)
            except:
                pass

        return stations

    def list_directory(self, path):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(path):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

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
            path_name = os.path.join(self.output_folder, filename)
            print(path_name)
            data_to_save = {"dates": def_rotated['dates'], "stream": st}

            file_to_store = open(path_name, "wb")
            pickle.dump(data_to_save, file_to_store)


    def _normalize_pair(self, a: str, b: str) -> str:
        """Return canonical 'sta1_sta2' with sorted ends so a_b == b_a."""
        s1, s2 = sorted((a, b))
        return f"{s1}_{s2}"

    def _parse_filename(self, fname: str):
        FILENAME_RE = re.compile(
            r"^(?P<net>[^.]+)\.(?P<sta1>[A-Z0-9]+)_(?P<sta2>[A-Z0-9]+)\.(?P<chn>[A-Z]{2})_daily$"
        )
        """Return (net, sta1, sta2, chn) or None if it doesn't match."""
        m = FILENAME_RE.match(fname)
        if not m:
            return None
        d = m.groupdict()
        return d["net"], d["sta1"], d["sta2"], d["chn"]

    def _extract_matrix_from_pickle(self, pkl):
        """Extract common metadata and channel data from a loaded pickle."""
        st = pkl["stream"]
        dates = pkl.get("dates")
        # common meta from first trace
        tr0 = st[0]
        meta = {
            "net": tr0.stats.network,
            "geodetic": tr0.stats.mseed.get("geodetic"),
            "sampling_rate": tr0.stats.sampling_rate,
            "location": tr0.stats.location,
            "coordinates": tr0.stats.mseed.get("coordinates"),
            "dates": dates,
        }
        # collect data arrays for all traces in this pickle
        data = [tr.data for tr in st]
        chn = tr0.stats.mseed.get("cross_channels")
        return meta, chn, data

    def rotate_specific_daily(self):


        CHANNELS_WANTED = {"EE", "EN", "NN", "NE"}
        base = Path(self.stack_daily_files_path)

        # 1) Collect all files once
        obsfiles = list(self.list_directory(self.stack_daily_files_path))  # keep your method
        # If list_directory returns strings, turn into Path for parsing/display
        obsfiles = [Path(p) for p in obsfiles]

        # 2) Build an index: {(pair_key, chn): filepath}
        files_index = {}
        for p in obsfiles:
            key = self._parse_filename(p.name)
            if not key:
                continue  # skip files that don't match expected pattern
            net, sta1, sta2, chn = key
            if chn not in CHANNELS_WANTED:
                continue
            pair_key = self._normalize_pair(sta1, sta2)
            files_index[(pair_key, chn)] = p

        # 3) Deduplicate & normalize station pairs the user cares about
        raw_pairs = self.list_stations_daily(self.stack_daily_files_path)
        pair_keys = sorted({self._normalize_pair(*sp.split("_")) for sp in raw_pairs if "_" in sp})

        for pair_key in pair_keys:
            sta1, sta2 = pair_key.split("_")
            if sta1 == sta2:
                # skip self-pairs if not meaningful
                continue

            matrix_data = {}
            meta_set = False

            # 4) Load the per-channel pickles only if they exist for this pair
            for chn in CHANNELS_WANTED:
                p = files_index.get((pair_key, chn))
                if p is None:
                    continue

                try:
                    # Load once per needed file
                    file_pickle = pickle.load(open(p, "rb"))
                except (OSError, pickle.UnpicklingError) as e:
                    # skip broken/unreadable file
                    # (log if you have a logger)
                    continue

                try:
                    meta, chn_from_file, data = self._extract_matrix_from_pickle(file_pickle)

                    # Sanity: ensure channel matches expectation
                    if chn_from_file not in CHANNELS_WANTED or chn_from_file != chn:
                        # channel mismatch — skip
                        continue

                    # Set common meta once (all channels should agree)
                    if not meta_set:
                        matrix_data.update(meta)
                        meta_set = True

                    matrix_data[chn] = data

                except Exception as e:
                    # Narrowly skip this file if it doesn't have expected structure
                    # (log if desired)
                    continue

            # 5) Rotate and save if we have enough data
            try:
                if meta_set and any(k in matrix_data for k in CHANNELS_WANTED):
                    rotated = self.__rotate_specific(matrix_data)  # your rotation method
                else:
                    continue  # nothing to rotate for this pair

                if rotated:
                    payload = {
                        "rotated_matrix": rotated,
                        "geodetic": matrix_data.get("geodetic"),
                        "net": matrix_data.get("net"),
                        "station_pair": pair_key,  # canonical name
                        "sampling_rate": matrix_data.get("sampling_rate"),
                        "location": matrix_data.get("location"),
                        "dates": matrix_data.get("dates"),
                        "coordinates": matrix_data.get("coordinates"),
                    }

                    print(pair_key, "rotated")
                    self.save_rotated_specific(payload)
                    print(pair_key, "saved")

            except Exception as e:
                print("Couldn't rotate/save", pair_key, "->", e)

if __name__ == "__main__":
    stack_files_path = "/Volumes/LaCie/UPFLOW_NEW_MATRIX/upflow_new_horizontals/stack"
    stack_daily_files_path = "/Volumes/LaCie/UPFLOW_NEW_MATRIX/upflow_new_horizontals/stack_daily"
    stack_rotated_files_path = "/Volumes/LaCie/UPFLOW_NEW_MATRIX/upflow_new_horizontals/stack_rotated"
    output_folder = "/Volumes/LaCie/UPFLOW_NEW_MATRIX/upflow_new_horizontals/stack_daily_rotated"
    RS = RotateStack(stack_files_path, stack_daily_files_path=stack_daily_files_path,
                     stack_rotated_files_path=stack_rotated_files_path, output_folder=output_folder)

    RS.rotate_specific_daily()