import pickle
from isp import CLOCK_PATH
import os
file_name = "UP40_PMAR"
file_path = os.path.join(CLOCK_PATH, file_name)
map = pickle.load(file_path)