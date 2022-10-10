import os
from obspy import read_inventory


def list_directory(data_path):
    obsfiles = []
    for top_dir, sub_dir, files in os.walk(data_path):
        for file in files:
            obsfiles.append(os.path.join(top_dir, file))
    obsfiles.sort()
    return obsfiles

path = '/Volumes/NO NAME/continous/metadata'
path_UPFLOW_OBS = '/Volumes/NO NAME/Metadata_Upflow/UPFLOWprojectxml'


path_UPFLOW_OBS_output = '/Volumes/NO NAME/Metadata_Upflow/UPFLOWproject_full_xml'

meta_origin = read_inventory(path_UPFLOW_OBS)
files = list_directory(path)
inv_land = {}
for meta in files:
    inv = read_inventory(meta)
    #print(inv)
    meta_origin += inv

print(meta_origin)
meta_origin.write(path_UPFLOW_OBS_output, format = "stationxml")


