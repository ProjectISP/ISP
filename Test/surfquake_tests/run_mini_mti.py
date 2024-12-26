import os
from surfquakecore.moment_tensor.mti_parse import read_isola_result, WriteMTI
from surfquakecore.moment_tensor.sq_isola_tools.sq_bayesian_isola import BayesianIsolaCore
from surfquakecore.project.surf_project import SurfProject

def list_files_with_iversion_json(root_folder):
    iversion_json_files = []

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "inversion.json":
                iversion_json_files.append(os.path.join(foldername, filename))

    return iversion_json_files

if __name__ == "__main__":

    inventory_path = "/test_surfquake/inputs/metadata/inv_all.xml"
    path_to_project = "/test_surfquake/outputs/project/project.pkl"
    path_to_configfiles = '/test_surfquake/inputs/configs/mti_config_mini_test.ini'
    output_directory = '/test_surfquake/outputs/mti'
    #
    # # Load the Project
    sp = SurfProject.load_project(path_to_project_file=path_to_project)
    print(sp)
    #
    # Build the class
    bic = BayesianIsolaCore(project=sp, inventory_file=inventory_path, output_directory=output_directory,
                             save_plots=True)
    #
    # Run Inversion
    bic.run_inversion(mti_config=path_to_configfiles)
    print("Finished Inversion")
    iversion_json_files = list_files_with_iversion_json(output_directory)

    for result_file in iversion_json_files:
        result = read_isola_result(result_file)
        print(result)

    print("Writting MTI summary")

    wm = WriteMTI(output_directory)
    wm.mti_summary()

    print("End of process, please review output directory")
