import pickle

def read_load_project(file):
    project = pickle.load(open(file, "rb"))
    return project

def sort_channels(selection):
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

    return id




file ='/Users/robertocabieces/Documents/UPFLOW_denoise'
project = read_load_project(file)
#print(project)
# 1 extract stations
station_list = []
#selection = [["WM", "OBS05", "SHY"], ["WM", "OBS05", "SHX"], ["WM", "OBS05", "SHZ"], ["WM", "OBS05", "SDH"]]
selection = []
channel_check = []
full_selection = []
for key in project:
    net = key.split(".")[0]
    sta = key.split(".")[1]
    chn = key.split(".")[2]
    if sta not in station_list:
        station_list.append(sta)
#print(station_list)
# extract channels
for sta_check in station_list:
    for key in project:
        net = key.split(".")[0]
        sta = key.split(".")[1]
        chn = key.split(".")[2]
        if sta_check == sta and chn not in channel_check:
            channel_check.append(chn)
            selection.append([net, sta, chn])
    #print(selection)
    full_selection.append(selection)
    selection = []
    channel_check = []

for selection in full_selection:
    print(sort_channels(selection))

