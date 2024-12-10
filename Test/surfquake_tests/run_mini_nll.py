from surfquakecore.earthquake_location.run_nll import Nllcatalog, NllManager

if __name__ == "__main__":

    # Basic input: working_directory, inventory file path and config_file input
    working_directory = '/Volumes/LaCie/all_andorra/mini_test/outputs/nll_test'
    inventory_path = "/Volumes/LaCie/all_andorra/mini_test/inputs/metadata/inv_all.xml"
    path_to_configfiles = '/Volumes/LaCie/all_andorra/mini_test/inputs/configs/nll_config.ini'
    nll_manager = NllManager(path_to_configfiles, inventory_path, working_directory)
    #nll_manager.vel_to_grid()
    #nll_manager.grid_to_time()
    for iter in range(1, 5):
        nll_manager.run_nlloc(num_iter=iter)
    nll_catalog = Nllcatalog(working_directory)
    nll_catalog.run_catalog(working_directory)

    # command version
    """
    surfquake locate -i /Volumes/LaCie/all_andorra/mini_test/inputs/metadata/inv_all.xml 
    -c /Volumes/LaCie/all_andorra/mini_test/inputs/configs/nll_config.ini 
    -o /Volumes/LaCie/all_andorra/mini_test/outputs/nll_test -s -n 4
    """