import logging
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCATION_OUTPUT_PATH = os.path.join(ROOT_DIR, 'earthquakeAnalysis/location_output/loc')
ALL_LOCATIONS = os.path.join(ROOT_DIR, 'earthquakeAnalysis/location_output/all_locations')
LOC_STRUCTURE = os.path.join(ROOT_DIR, "earthquakeAnalysis", "loc_structure")
PICKING_DIR = os.path.join(ROOT_DIR, 'earthquakeAnalysis/location_output/obs')
MOMENT_TENSOR_OUTPUT = os.path.join(ROOT_DIR, 'mti/output')
RESOURCE_PATH = os.path.join(ROOT_DIR, 'resources')
IMAGES_PATH = os.path.join(RESOURCE_PATH, 'images')
UIS_PATH = os.path.join(RESOURCE_PATH, 'designer_uis')
FOC_MEC_PATH = os.path.join(ROOT_DIR, 'focmec/bin')
FOC_MEC_BASH_PATH = os.path.join(ROOT_DIR, 'earthquakeAnalysis/loc_structure/first_polarity/focmec_run')
MACROS_PATH = os.path.join(ROOT_DIR, 'macros')
HELP_PATH = os.path.join(ROOT_DIR, 'ISP_Documentation')
EVENTS_DETECTED = os.path.join(ROOT_DIR, 'earthquakeAnalysis/auto_detect/events')
AUTOMATIC_PHASES = os.path.join(ROOT_DIR, 'earthquakeAnalysis/auto_detect/phases')
DISP_MAPS = os.path.join(ROOT_DIR, 'ant/data_tomo')
DISP_REF_CURVES = os.path.join(ROOT_DIR, 'ant/disp_curv_ref')
CLOCK_PATH = os.path.join(ROOT_DIR, 'ant/clock_dir')
MAGNITUDE_DICT_PATH = os.path.join(ROOT_DIR, 'DataProcessing/MagnitudesConfig')
GREEN_SOURCE = os.path.join(ROOT_DIR, 'mti/green_source')
GREEN = os.path.join(ROOT_DIR, 'mti/green')
LOGMTI = os.path.join(ROOT_DIR, 'mti/output/log.txt')
nll_templates = os.path.join(ROOT_DIR, "earthquakeAnalysis", "loc_structure/run")
nll_ak135 = os.path.join(ROOT_DIR, "earthquakeAnalysis", "loc_structure/ak135")
BINARY_NLL_DIR = os.path.join(ROOT_DIR, 'NLL7/src/bin')
MAP_SERVICE_URL = "https://wms.gebco.net/mapserv?"
MAP_LAYER = "GEBCO_LATEST"
source_config = os.path.join(ROOT_DIR, "source_spec_parse/config/source_spec.conf")
def create_logger():

    # create logger.
    logger = logging.getLogger('logger')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # create console handler and set level to debug.
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create file handler.
        file_log = logging.FileHandler(filename="app.log")
        file_log.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)
        file_log.setFormatter(formatter)

        # add ch and file_log to logger
        logger.addHandler(ch)
        logger.addHandler(file_log)

    return logger


app_logger = create_logger()
