import numpy as np
from matplotlib import pyplot as plt

class PDFmanger:

    def __init__(self, scatter_x, scatter_y, scatter_z, pdf):
        """
        Plot stations map fro dictionary (key = stations name, coordinates)

        :param
        """
        self.x = scatter_x
        self.y = scatter_y
        self.z = scatter_z
        self.pdf = pdf

    def plot_scatter(self):

            from isp.Gui.Frames import MatplotlibFrame
            print("Plotting PDF")
            pdf = np.array(self.pdf) / np.max(self.pdf)

            left, width = 0.06, 0.65
            bottom, height = 0.1, 0.65
            spacing = 0.02
            rect_scatter = [left, bottom, width, height]
            rect_scatterlon = [left, bottom + height + spacing, width, 0.2]
            rect_scatterlat = [left + width + spacing, bottom, 0.2, height]


            fig = plt.figure(figsize=(10, 8))
            self.mpf = MatplotlibFrame(fig)
            ax_scatter = plt.axes(rect_scatter)
            ax_scatter.tick_params(direction='in', top=True, right=True, labelsize=10)
            plt.scatter(self.x, self.y, s=10, c=pdf, alpha=0.5, marker=".", cmap=plt.cm.jet)
            plt.xlabel("Longitude", fontsize=10)
            plt.ylabel("Latitude", fontsize=10)
            ax_scatx = plt.axes(rect_scatterlon)
            ax_scatx.tick_params(direction='in', labelbottom=False, labelsize=10)
            plt.scatter(self.x, self.z, s=10, c=pdf, alpha=0.5, marker=".", cmap=plt.cm.jet)
            plt.ylabel("Depth (km)", fontsize=10)
            plt.gca().invert_yaxis()
            ax_scatx = plt.axes(rect_scatterlat)
            ax_scatx.tick_params(direction='in', labelleft=False, labelsize=10)
            ax_scaty = plt.axes(rect_scatterlat)
            ax_scaty.tick_params(direction='in')
            ax_scaty.tick_params(which='major', labelsize=10)
            plt.scatter(self.z, self.y, s=10, c=pdf, alpha=0.5, marker=".", cmap=plt.cm.jet)
            ax_scaty = plt.axes(rect_scatterlat)
            ax_scaty.tick_params(direction='in', labelsize=10)
            plt.xlabel("Depth (km)", fontsize=10)
            cax = plt.axes([0.95, 0.1, 0.02, 0.8])
            plt.colorbar(cax=cax)
            self.mpf.show()