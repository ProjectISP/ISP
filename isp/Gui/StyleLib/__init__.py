import os
import matplotlib.pyplot as plt


def get_isp_mpl_style_file():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isp.mplstyle")
    if os.path.isfile(file_path):
        return file_path
    return None


def set_isp_mpl_style_file():
    file_path = get_isp_mpl_style_file()
    plt.style.use(file_path)
