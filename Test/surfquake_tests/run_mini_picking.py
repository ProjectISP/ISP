import os
from multiprocessing import freeze_support
from surfquakecore.phasenet.phasenet_handler import PhasenetUtils
from surfquakecore.phasenet.phasenet_handler import PhasenetISP
from surfquakecore.project.surf_project import SurfProject

### LOAD PROJECT ###
path_to_project = "/test_surfquake/outputs/project"
project_name = 'project.pkl'
output_picks = '/test_surfquake/outputs/picks'
project_file = os.path.join(path_to_project, project_name)

if __name__ == '__main__':
    freeze_support()

    # Load project
    sp_loaded = SurfProject.load_project(path_to_project_file=project_file)
    print(sp)
    # Instantiate the class PhasenetISP
    phISP = PhasenetISP(sp.project, amplitude=True, min_p_prob=0.30, min_s_prob=0.30)
    # Running Stage
    picks = phISP.phasenet()
    # """ PHASENET OUTPUT TO REAL INPUT """
    picks_results = PhasenetUtils.split_picks(picks)
    PhasenetUtils.convert2real(picks_results, output_picks)
    PhasenetUtils.save_original_picks(picks_results, output_picks)