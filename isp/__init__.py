import logging
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_PATH = os.path.join(ROOT_DIR, 'resources')
IMAGES_PATH = os.path.join(RESOURCE_PATH, 'images')
UIS_PATH = os.path.join(RESOURCE_PATH, 'designer_uis')


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
