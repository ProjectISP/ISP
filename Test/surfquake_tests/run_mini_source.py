from surfquakecore.magnitudes.run_magnitudes import Automag
from surfquakecore.magnitudes.source_tools import ReadSource
from surfquakecore.project.surf_project import SurfProject
import os

if __name__ == "__main__":

    cwd = os.path.dirname(__file__)
    ## Project definition ##
    project_path_file = "/test_surfquake/outputs/project/project.pkl"
    print("project:", project_path_file)
    sp_loaded = SurfProject.load_project(path_to_project_file=project_path_file)
    print(sp_loaded)

    ## Basic input: working_directory, invenoty file path and config_file input
    working_directory = 'test_surfquake/outputs/source'

    # inventory path must be placed inside config_file
    inventory_path = "/test_surfquake/inputs/metadata/inv_all.xml"
    path_to_configfiles = '/test_surfquake/inputs/configs/source_spec_narrow.conf'
    locations_directory = '/test_surfquake/outputs/nll/all_loc'
    output_directory = os.path.join(working_directory, "output")
    summary_path = '/test_surfquake/outputs/source/source_summary.txt'

    # Running stage
    mg = Automag(sp_loaded, locations_directory, inventory_path, path_to_configfiles, output_directory, "regional")
    mg.estimate_source_parameters()

    # Now we can read the output and even write a txt summarizing the results
    rs = ReadSource(output_directory)
    summary = rs.generate_source_summary()
    rs.write_summary(summary, summary_path)