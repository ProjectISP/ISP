from datetime import datetime
from multiprocessing import Pool
import numpy as np
import obspy
from obspy import UTCDateTime, Stream, read_inventory
import os
import time
from os import path
import random
from obspy.core.trace import Trace

start_time = time.time()


class DecimateSmart():

    def __init__(self, input_folder, output_files_path, new_sampling_rate=20, inventory=None, remove_glitches=False,
                 cores=None):

        """
        Process decimation,

        :param input data where is places mseed files,
         output folder for desimated files
        """

        self.input_folder = input_folder
        self.output_folder = output_files_path
        self.gaps_tol = 120
        self.obsfiles = []
        self.new_sampling_rate = new_sampling_rate
        self.remove_glitches = remove_glitches
        if cores is None:
            self.cpuCount = os.cpu_count() - 1
        else:
            self.cpuCount = cores

        if inventory != None:
            self.inventory = read_inventory(inventory)

        self.list_dir()

    def list_dir(self):

        for top_dir, _, files in os.walk(self.input_folder):

            for file in files:
                self.obsfiles.append(os.path.join(top_dir, file))

        self.obsfiles.sort()
        print(self.obsfiles)

    # get_julday

    def ensure_24(self, tr):
        # Ensure that this trace is set to have 24h points padding with zeros the starttime and endtime
        # take random numbers to ensure the day
        random_list = np.random.choice(len(tr), 100)
        times_posix = tr.times(type="timestamp")
        days_prob = times_posix[random_list.tolist()]
        days_prob_max = days_prob.tolist()
        max_prob = max(set(days_prob_max), key=days_prob_max.count)
        year = int(datetime.utcfromtimestamp(max_prob).strftime('%Y'))
        month = int(datetime.utcfromtimestamp(max_prob).strftime('%m'))
        day = int(datetime.utcfromtimestamp(max_prob).strftime('%d'))
        key = str(year) + "." + str(month) + "." + str(day) + "." + str(random.randint(0, 9))
        check_starttime = UTCDateTime(year=year, month=month, day=day, hour=00, minute=00, microsecond=00)
        check_endtime = check_starttime + 24 * 3600
        tr.detrend(type="simple")
        tr.trim(starttime=check_starttime, endtime=check_endtime, pad=True, nearest_sample=True, fill_value=0)
        return tr, key

    def fill_gaps(self, st, tol):
        gaps = st.get_gaps()

        if len(gaps) > 0 and self.check_gaps(gaps, tol):
            st.print_gaps()
            st = []

        elif len(gaps) > 0 and self.check_gaps(gaps, tol) == False:
            st.print_gaps()
            st.merge(fill_value="interpolate", interpolation_samples=-1)

        elif len(gaps) == 0 and self.check_gaps(gaps, tol) == False:
            pass

        return st

    def check_gaps(self, gaps, tol):
        time_gaps = []
        for i in gaps:
            time_gaps.append(i[6])

        sum_total = sum(time_gaps)

        if sum_total > tol:
            check = True
        else:
            check = False

        return check

    def remove_response(self, tr, f1, f2, f3, f4, water_level, units):

        try:

            tr.remove_response(inventory=self.inventory, pre_filt=(f1, f2, f3, f4), output=units,
                               water_level=water_level, taper_fraction=0.025)
        except:

            tr = None
            print("Coudn't deconvolve")

        return tr

    def hampel(self, tr, window_size, n_sigmas=3):

        """
            Median absolute deviation (MAD) outlier in Time Series
            :param ts: a trace obspy object representing the timeseries
            :param window_size: total window size in seconds
            :param n: threshold, default is 3 (Pearson's rule)
            :return: Returns the corrected timeseries
            """

        size = tr.count()
        input_series = tr.data
        window_size = int(window_size * tr.stats.sampling_rate)
        tr.data = self.__hampel_aux(input_series, window_size, size, n_sigmas)

        return tr

    def __hampel_aux(self, input_series, window_size, size, n_sigmas):

        k = 1.4826  # scale factor for Gaussian distribution
        # indices = []
        new_series = input_series.copy()
        # possibly use np.nanmedian
        for i in range((window_size), (size - window_size)):
            x0 = np.median(input_series[(i - window_size):(i + window_size)])
            S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
            if (np.abs(input_series[i] - x0) > n_sigmas * S0):
                new_series[i] = x0
        return new_series

    def process_data(self, j):

        tr_raw = None
        tr = None

        try:
            print(self.obsfiles[j])
            # tr_raw = obspy.read(self.obsfiles[j], format = "MSEED")[0]
            tr_raw = obspy.read(self.obsfiles[j])[0]

        except:

            print("Something went wrong during reading!!!", self.obsfiles[j])
            tr_raw = None

        if isinstance(tr_raw, Trace):

            try:
                # ensure the starttime and endtime and to have 24 h
                # tr, key = self.ensure_24(tr_raw)
                print("Processing", tr_raw.stats.station, tr_raw.stats.channel,
                      str(tr_raw.stats.starttime.julday) + "." + str(tr_raw.stats.starttime.year))
                st = self.fill_gaps(Stream(traces=tr_raw), tol=5 * self.gaps_tol)
                tr = st[0]
            except:

                print("Something went wrong during getting gaps step", self.obsfiles[j])
                tr = None

        if isinstance(tr, Trace):

            #try:

            tr.detrend(type="simple")
            tr.taper(type="blackman", max_percentage=0.025)
            sampling_rate = tr.stats.sampling_rate
            f4 = (sampling_rate / 3) * 0.5
            f3 = (sampling_rate / 4) * 0.5
            tr = self.remove_response(tr, f1=0.005, f2=0.008, f3=f3, f4=f4, water_level=90, units="VEL")

            # Anti-aliasing filter
            tr.filter("lowpass", freq=0.4 * self.new_sampling_rate, zerophase=True, corners=4)

            if tr.stats.sampling_rate == 100:
                steps = [10, 1]
                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.025)
                tr.filter(type="lowpass", freq=0.4 * self.new_sampling_rate, zerophase=True, corners=4)

                if self.new_sampling_rate <= 10:
                    tr.detrend(type="simple")
                    tr.taper(type="blackman", max_percentage=0.025)
                    tr.resample(sampling_rate=10, no_filter=True)

                if self.new_sampling_rate <= 1:
                    tr.detrend(type="simple")
                    tr.taper(type="blackman", max_percentage=0.025)
                    tr.resample(sampling_rate=1, no_filter=True)

                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.025)
                if self.new_sampling_rate not in steps:
                    tr.resample(sampling_rate=self.new_sampling_rate, no_filter=True)

            elif tr.stats.sampling_rate == 50:
                steps = [5, 1]
                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.025)
                tr.filter(type="lowpass", freq=0.4 * self.new_sampling_rate, zerophase=True, corners=4)

                if self.new_sampling_rate <= 5:
                    tr.detrend(type="simple")
                    tr.taper(type="blackman", max_percentage=0.025)
                    tr.resample(sampling_rate=5, no_filter=True)

                if self.new_sampling_rate <= 1:
                    tr.detrend(type="simple")
                    tr.taper(type="blackman", max_percentage=0.025)
                    tr.resample(sampling_rate=1, no_filter=True)

                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.025)
                if self.new_sampling_rate not in steps:
                    tr.resample(sampling_rate=self.new_sampling_rate, no_filter=True)

            elif tr.stats.sampling_rate <= 40:
                steps = [5, 1]
                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.025)
                tr.filter(type="lowpass", freq=0.4 * self.new_sampling_rate, zerophase=True, corners=4)

                if self.new_sampling_rate <= 5:
                    tr.detrend(type="simple")
                    tr.taper(type="blackman", max_percentage=0.025)
                    tr.resample(sampling_rate=5, no_filter=True)

                if self.new_sampling_rate <= 1:
                    tr.detrend(type="simple")
                    tr.taper(type="blackman", max_percentage=0.025)
                    tr.resample(sampling_rate=1, no_filter=True)

                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.025)
                if self.new_sampling_rate not in steps:
                    tr.resample(sampling_rate=self.new_sampling_rate, no_filter=True)

            elif tr.stats.sampling_rate == 250:
                steps = [25, 5, 1]
                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.025)
                tr.filter(type="lowpass", freq=0.4 * self.new_sampling_rate, zerophase=True, corners=4)

                if self.new_sampling_rate <= 25:
                    tr.detrend(type="simple")
                    tr.taper(type="blackman", max_percentage=0.025)
                    tr.resample(sampling_rate=25, no_filter=True)

                if self.new_sampling_rate <= 5:
                    tr.detrend(type="simple")
                    tr.taper(type="blackman", max_percentage=0.025)
                    tr.resample(sampling_rate=5, no_filter=True)

                if self.new_sampling_rate <= 1:
                    tr.detrend(type="simple")
                    tr.taper(type="blackman", max_percentage=0.025)
                    tr.resample(sampling_rate=1, no_filter=True)

                tr.detrend(type="simple")
                tr.taper(type="blackman", max_percentage=0.025)
                if self.new_sampling_rate not in steps:
                    tr.resample(sampling_rate=self.new_sampling_rate, no_filter=True)

                if self.remove_glitches:
                    tr = self.hampel(tr, window_size=0.7)

            # Convert the data to 'float32'
            data = tr.data.astype('float32')
            tr.data = data
            starttime = (tr.stats.starttime + 1300)
            day = str(starttime.julday)
            year = str(starttime.year)
            name = tr.id + "." + day + "." + year
            destination = os.path.join(self.output_folder, name)
            tr.write(destination, format="MSEED", encoding='FLOAT32')

            #except:
            #    print("Couldn't decimate (couldn't write out?) ", self.obsfiles[j])

    def run_process(self):
        parallel = True
        if parallel:

            with Pool(processes=self.cpuCount) as pool:
                pool.map(self.process_data, range(len(self.obsfiles)))
        else:
            for j in range(len(self.obsfiles)):
                self.process_data(j)


if __name__ == "__main__":

    input_path = "/Users/robertocabiecesdiaz/Desktop/toy_noise/data_selected"
    output_path = "/Users/robertocabiecesdiaz/Desktop/toy_noise/decimated"
    inventory_path = "/Users/robertocabiecesdiaz/Desktop/toy_noise/metadata/metadata_test"

    if not path.exists(output_path):
        os.mkdir(output_path)

    decimate_run = DecimateSmart(input_path, output_path, new_sampling_rate=5, inventory=inventory_path,
                                 remove_glitches=False, cores=10)
    decimate_run.run_process()

    print("Process finished --- %s seconds ---" % (time.time() - start_time))
