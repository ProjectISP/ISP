import os
import pickle
from obspy import UTCDateTime
from isp.Utils import MseedUtil
import numpy as np
from Test.compliance import RemoveComplianceTilt
from obspy import read

class CheckDataBaseCoherence:

    def __init__(self):

        """

        :param obs_file_path: The file path of pick observations.
        """
        self.project = {}
        self.files_path = []
        self.transfer_functions = {}
        self.transfer_functions['transfer_ne'] = []
        self.transfer_functions['transfer_nz'] = []
        self.transfer_functions['transfer_ez'] = []
        self.transfer_functions['transfer_hz'] = []

    def load_project(self, project_path):
        self.project = MseedUtil.load_project(project_path)

    def find_time_project_limits(self):

        starttimes = []
        endtimes = []

        for key in self.project:
            for k in range(len(self.project[key])):

                starttimes.append(self.project[key][k][1].starttime)
                endtimes.append(self.project[key][k][1].endtime)
                #print(self.project[key][k][0], self.project[key][k][1].starttime, self.project[key][k][1].endtime)
        self.min_start = np.min(starttimes)
        self.max_end = np.max(endtimes)

    def sort_channels(self, selection):
        if len(selection) ==4:
            id_def = ["N", "E", "Z", "H"]
            for id in selection:
                if id[2][2] == "1":
                    id_def[0] = id
                if id[2][2] == "2":
                    id_def[1] = id
                if id[2][2] == "Z":
                    id_def[2] = id
                if id[2][2] == "H":
                    id_def[3] = id
        if len(selection) == 3:
            id_def = ["N", "E", "Z"]
            for id in selection:
                if id[2][2] == "1":
                    id_def[0] = id
                if id[2][2] == "2":
                    id_def[1] = id
                if id[2][2] == "Z":
                    id_def[2] = id

        return id_def

    def filter_project(self, selection, start, end):

        files_selected = []
        for item in selection:

            net = item[0]
            station = item[1]
            channel = item[2]
            _, self.files_path = MseedUtil.filter_project_keys(self.project, net=net, station=station,
                                                               channel=channel)

            diff = end-start

            if diff > 0:
                files_path = MseedUtil.filter_time(list_files=self.files_path, starttime=start, endtime=end)
            else:
                files_path = MseedUtil.filter_time(list_files=self.files_path)

            files_selected.append(files_path[0])

        return files_selected


    def load_transfer(self, path_file, name):

        path_file = os.path.join(path_file, name)
        try:
            if os.path.exists(path_file):
                with open(path_file, 'rb') as handle:
                    mapping = pickle.load(handle)
                    print("Project Loaded")
                    return mapping

        except:
            print("Warning Something went wrong")


    def __get_selections(self):
        station_list = []
        selection = []
        channel_check = []
        self.full_selection = []
        for key in self.project:
            sta = key.split(".")[1]
            if sta not in station_list:
                station_list.append(sta)
        print(station_list)
        # extract channels
        for sta_check in station_list:
            for key in self.project:
                net = key.split(".")[0]
                sta = key.split(".")[1]
                chn = key.split(".")[2]
                if sta_check == sta and chn not in channel_check:
                    channel_check.append(chn)
                    selection.append([net, sta, chn])

            self.full_selection.append(selection)
            selection = []
            channel_check = []
        print(self.full_selection)


    def daily_denoise(self, selection, start, end, output_path, plot_path):
        selection = self.sort_channels(selection)
        secs = 24*3600
        days = int((self.max_end-self.min_start)/secs)
        start = UTCDateTime(start)
        stop = UTCDateTime(end)
        for j in range(days):
            starttime = start + (j - 1) * secs
            endtime = start + j * secs
            if starttime < stop:
                try:
                    files_selected = self.filter_project(selection, starttime, endtime)
                    print(files_selected)
                    if len(files_selected) == 4:
                        N = read(files_selected[0])
                        E = read(files_selected[1])
                        Z = read(files_selected[2])
                        H = read(files_selected[3])
                    else:
                        N = read(files_selected[0])
                        E = read(files_selected[1])
                        Z = read(files_selected[2])
                        H = [""]
                    noise = RemoveComplianceTilt(N[0], E[0], Z[0], H[0])
                    channels = {}
                    # First Tilt (between horizontal components)
                    # Y' = Y - Tyx*X
                    channels["source"] = N[0]
                    channels["response"] = E[0]
                    noise.transfer_function(channels)
                    noise.plot_coherence_transfer(channels, save_fig=True, path_save=plot_path)
                    noise.plot_transfer_function(channels, save_fig=True, path_save=plot_path)
                    #transfer_ne = noise.transfer_info
                    #self.transfer_functions['transfer_ne'].append(transfer_ne)
                    noise.remove_noise(channels)

                    # First Tilt (between horizontal/vertical components)
                    # Z' = Z- Tzx*X
                    channels["source"] = E[0]
                    channels["response"] = Z[0]
                    noise.transfer_function(channels)
                    noise.plot_coherence_transfer(channels, save_fig=True, path_save=plot_path)
                    noise.plot_transfer_function(channels, save_fig=True, path_save=plot_path)
                    noise.remove_noise(channels)
                    #transfer_ez = noise.transfer_info
                    #self.transfer_functions['transfer_ez'].append(transfer_ez)

                    # Second Tilt Noise (horizontal - Vertical)
                    # Z'' = Z' - Tz'y'*Y'
                    channels["source"] = noise.Nnew
                    channels["response"] = noise.Znew
                    noise.transfer_function(channels)
                    noise.plot_coherence_transfer(channels, save_fig=True, path_save=plot_path)
                    noise.plot_transfer_function(channels, save_fig=True, path_save=plot_path)
                    noise.remove_noise(channels)
                    noise.plot_compare_spectrums(channels, save_fig=True, path_save=plot_path)
                    #transfer_nz = noise.transfer_info
                    #self.transfer_functions['transfer_nz'].append(transfer_nz)

                    # Second Compliance Noise (between horizontal components)
                    if len(files_selected) == 4:
                        channels["source"] = H[0]
                        channels["response"] = noise.Znew
                        Ztilt = noise.Znew.copy()
                        noise.transfer_function(channels)
                        noise.plot_coherence_transfer(channels, save_fig=True, path_save=plot_path)
                        noise.plot_transfer_function(channels, save_fig=True, path_save=plot_path)
                        #transfer_hz = noise.transfer_info
                        #self.transfer_functions['transfer_hz'].append(transfer_hz)
                        noise.remove_noise(channels)
                        noise.plot_compare_spectrums(channels, save_fig=True, path_save=plot_path)
                        noise.plot_compare_spectrums_full(channels, Ztilt, save_fig=True, path_save=plot_path)
                        # save in time domain
                        tr_znew = noise.Znew
                        #tr_znew.stats.channel = 'SCZ'
                        year = tr_znew.stats.starttime.year
                        month = tr_znew.stats.starttime.month
                        day = tr_znew.stats.starttime.day
                        name = tr_znew.id + "_" +str(year)+"-"+str(month)+"-"+str(day)+"."+"mseed"
                        out = os.path.join(output_path, name)
                        tr_znew.write(out, format="mseed")

                    del noise
                except:
                    print("warning nothing to do for", starttime)

    def full_denoise(self, start, end, output_path, plots_path):

        self.__get_selections()

        # sta_done = ["X25", "UP01", "UP02", "UP03", "UP04", "UP05", "UP06", "UP07", "UP08", "UP09", "UP11", "UP12",
        #             "UP13", "UP14", "UP15", "UP16", "UP17", "UP18", "UP20", "UP21", "UP22", "UP24", "UP23", "UP25",
        #             "UP26", "UP27", "UP29", "UP30", "UP31", "UP32", "UP33", "UP34", "UP35", "UP36", "UP37",
        #             "UP38", "UP39", "UP40", "UP41", "UP42", "UP43", "UP44", "UP45", "UP46"]
        sta_done = []
        for selection in self.full_selection:
            sta = selection[0][1]
            print(sta)
            if sta not in sta_done:
                self.daily_denoise(selection, start, end, output_path, plots_path)


if __name__ == "__main__":

    output_path = "/Volumes/LaCie/UPFLOW_denoise/clean"
    plots_path = "/Volumes/LaCie/UPFLOW_denoise/plots"
    project_file_path = '/Users/robertocabieces/Documents/UPFLOW_denoise'
    start = "2021-06-21TT00:00:00.00"
    end = "2022-09-08TT00:00:00.00"
    Cdb = CheckDataBaseCoherence()
    Cdb.load_project(project_file_path)
    Cdb.find_time_project_limits()
    Cdb.full_denoise(start, end, output_path, plots_path)