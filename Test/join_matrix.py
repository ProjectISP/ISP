import os
import pickle
import numpy as np
import gc
from isp.Utils import MseedUtil

def list_directory(data_path):
    obsfiles = []
    for top_dir, sub_dir, files in os.walk(data_path):
        for file in files:
            obsfiles.append(os.path.join(top_dir, file))
    obsfiles.sort()
    return obsfiles

def check_header(list_files):

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




if __name__ == "__main__":
    path_mini_matrix = "/Volumes/LaCie/UPFLOW_resample/EGFs_Horizontals/mini_matrix"
    path_partial_matrix= "/Volumes/LaCie/UPFLOW_resample/EGFs_Horizontals/horizontals_land"
    output_path = "/Volumes/LaCie/UPFLOW_resample/EGFs_Horizontals/full_matrices_def"
    obsfiles_mini = list_directory(path_mini_matrix)
    obsfiles_partial = list_directory(path_partial_matrix)

    for mini_file in obsfiles_mini:
        full_matrix = {}
        mini_name = os.path.basename(mini_file)

        for partial_file in obsfiles_partial:
            partial_name = os.path.basename(partial_file)
            if partial_name == mini_name and mini_name != ".DS_Store":
                print("working on ", mini_name)
                mini_matrix = MseedUtil.load_project(file=mini_file)
                partial_matrix = MseedUtil.load_project(file=partial_file)

                # concatenate header
                key1 = "data_matrix" + "_" + mini_name[-1]
                key2 = 'metadata_list' + "_" + mini_name[-1]
                key3 = 'date_list' + "_" + mini_name[-1]

                # concatenate matrix
                #try:
                full_matrix[key1] = np.concatenate((partial_matrix[key1], mini_matrix[key1]), axis=1)
                full_matrix[key2] = partial_matrix[key2]+mini_matrix[key2]
                partial_matrix[key3] = check_header(partial_matrix[key3])
                full_matrix[key3] = partial_matrix[key3] + mini_matrix[key3]
                del mini_matrix
                del partial_matrix
                gc.collect()
                # write full matrix in output directory
                file_to_store = open(os.path.join(output_path, mini_name), "wb")
                pickle.dump(full_matrix, file_to_store)
                #except:
                #    print("An exception ocurred at ", mini_name)
