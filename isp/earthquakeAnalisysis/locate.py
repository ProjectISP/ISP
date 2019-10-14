#import matplotlib
#matplotlib.use('QT5Agg')
from isp.Structures.structures import TracerStats
from isp.Utils import MseedUtil, ObspyUtil
import matplotlib.pyplot as plt

class EarthquakeLocation:

    def __init__(self, file_path):
        self.file_path = file_path
        self.mseed_files = MseedUtil.get_mseed_files(file_path)

    def locate_earthquake(self, val):
        print(val)
        #data = np.array([0.7, 0.7, 0.7, 0.8, 0.9, 0.9, 1.5, 1.5, 1.5, 1.5])
        #fig, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True)

        #bins = np.arange(0.6, 1.62, 0.02)

        #n1, bins1, patches1 = ax1[0].hist(data, bins, alpha=0.6, density=False, cumulative=False)
        #n1, bins1, patches1 = ax1[1].hist(data, bins, alpha=0.6, density=False, cumulative=False)

    def paginate(self, files_per_page, current_page=1):
        index_0 = (current_page - 1)*files_per_page
        index_1 = index_0 + files_per_page
        return self.mseed_files[index_0:index_1]

    def plot(self, files_per_page,current_page, show = False):

        list_files = self.paginate(files_per_page,current_page)
        nfilas = len(list_files)
        if nfilas == 0:
            return None
        k = 0
        fig, axes = plt.subplots(nrows=nfilas, ncols=1, sharex=True)
        for file in list_files:
            tr = ObspyUtil.get_tracer_from_file(file)
            stats = TracerStats.from_dict(tr.stats)
            # Obs: after doing tr.detrend it add processing key to stats
            tr.detrend(type="demean")

            data=tr.data
            if nfilas > 1:
                ax=axes[k]
            else:
                ax=axes
            ax.plot(data)
            #plt.title("Multi Taper Spectrogram" + stats.Station)
            #plt.xlabel("Time after %s [s]" % stats.StartTime)
            #plt.ylabel("Frequency [Hz]")
            k += 1
        if show:
            plt.plot()
        else:
            return fig