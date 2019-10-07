import os

from PyQt5 import uic

from isp import UIS_PATH


def load_ui_designers(ui_file):

    ui_path = os.path.join(UIS_PATH, ui_file)
    if not os.path.isfile(ui_path):
        raise FileNotFoundError("The file {} can't be found at the location {}".
                                format(ui_file, UIS_PATH))
    ui_class, _ = uic.loadUiType(ui_path, from_imports=True, import_from='isp.resources')
    return ui_class
