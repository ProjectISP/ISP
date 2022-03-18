import itertools
import pickle


class tomotools:

    #def __init__(self, root_path):

    @classmethod
    def read_dispersion(cls, file, wave_type, dispersion_type):

        disp_data = {}
        dict_ = pickle.load(open(file, "rb"))

        for pair in dict_.keys():
            stnm1, stnm2, wave, type_ = pair.split("_")
            if type_ in dispersion_type and wave in wave_type:
                disp_data.setdefault((stnm1, stnm2), {})
                disp_data[(stnm1, stnm2)]['c(f)'] = [float(x) for x in dict_[pair]['velocity']]
                disp_data[(stnm1, stnm2)]['f'] = [float(x) for x in dict_[pair]['period']]

        return disp_data

    @classmethod
    def get_station_info(cls, station_csv):
        st_info = {}
        with open(station_csv, 'r') as f:
            for line in f.readlines():
                line = [x for x in line.split(' ') if x != '']
                stnm = line[0]
                lat = float(line[2])
                lon = float(line[3])
                st_info[stnm] = [lon, lat]

        stations = sorted([stnm for stnm in st_info.keys()])
        pairs = list(itertools.combinations(stations, 2))

        return st_info, pairs