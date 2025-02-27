from multiprocessing import freeze_support
from surfquakecore.project.surf_project import SurfProject


path_to_data = "/Users/robertocabiecesdiaz/Documents/test_surfquake/inputs/waveforms_cut"
path_to_project = "/Users/robertocabiecesdiaz/Documents/test_surfquake/project/project.pkl"

if __name__ == '__main__':

    freeze_support()
    sp = SurfProject(path_to_data)
    sp.search_files()
    print(sp)
    sp.save_project(path_file_to_storage=path_to_project)

