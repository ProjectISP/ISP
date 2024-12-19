from typing import Union
from datetime import datetime
from obspy import UTCDateTime, read, Stream
from obspy.signal.trigger import coincidence_trigger
from multiprocessing import Pool
import pandas as pd
from surfquakecore.project.surf_project import SurfProject
from isp.DataProcessing import ConvolveWaveletScipy
from isp.Utils import MseedUtil


class CoincidenceTrigger:
    def __init__(self, project: Union[SurfProject, str], parameters: dict):

        if isinstance(project, str):
            self.project = SurfProject.load_project(project)
        else:
            self.project = project

        self.method: str = parameters.pop('method', 'classicstalta')
        self.the_on = parameters.pop('the_on', 20)
        self.the_off = parameters.pop('the_off', 10)
        self.fmin = parameters.pop('fmin', 1)  # freq_min
        self.fmax = parameters.pop('fmax', 40)  # freq_max
        self.centroid_radio = parameters.pop('centroid_radio', 60)  # centroid radious
        self.coincidence = parameters.pop("coincidence", 3)  # coincidence

        if self.method == "classicstalta":

            self.sta = parameters.pop("time_window", 1)  # time_window_kurtosis
            self.lta = parameters.pop("time_window", 40)

        elif self.method == "kurtosis":

            self.time_window = parameters.pop("time_window", 5)  # time_window_kurtosis
            self.decimate_sampling_rate: Union[bool, int] = parameters.pop("decimate_sampling_rate", False)

    def fill_gaps(self, tr, tol=5):

        tol_seconds_percentage = int((tol / 100) * len(tr.data)) * tr.stats.delta
        st = Stream(traces=tr)
        gaps = st.get_gaps()

        if len(gaps) > 0 and self._check_gaps(gaps, tol_seconds_percentage):
            st.print_gaps()
            st.merge(fill_value="interpolate", interpolation_samples=-1)
            return st[0]

        elif len(gaps) > 0 and not self._check_gaps(gaps, tol_seconds_percentage):
            st.print_gaps()
            return None
        elif len(gaps) == 0 and self._check_gaps(gaps, tol_seconds_percentage):
            return tr
        else:
            return tr

    def _check_gaps(self, gaps, tol):
        time_gaps = []
        for i in gaps:
            time_gaps.append(i[6])

        sum_total = sum(time_gaps)

        return sum_total <= tol

    def _extract_event_info(self, trigger):
        events_times = []
        for k in range(len(trigger)):
            detection = trigger[k]
            for key in detection:

                if key == 'time':
                    time = detection[key]
                    events_times.append(time)
        return events_times

    def thresholding_sta_lta(self, files_list, start, end):

        traces = []
        for file in files_list:
            try:
                tr = read(file)[0]
                tr = self.fill_gaps(tr)
                if tr is not None:
                    traces.append(tr)
            except:
                pass
        st = Stream(traces)
        #st.select(station="CARF", channel="*Z") # no necessary, better filter project before
        st.trim(starttime=start, endtime=end)
        st.merge()
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)
        st.filter(type="bandpass", freqmin=self.fmin, freqmax=self.fmax)
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)
        events = coincidence_trigger('classicstalta', self.the_on, self.the_off, st,
                                     thr_coincidence_sum=self.coincidence,
                                     max_trigger_length=self.centroid_radio, sta=self.sta, lta=self.lta)
        triggers = self._extract_event_info(events)
        events_times_cluster, _ = MseedUtil.cluster_events(triggers, eps=self.centroid_radio)
        print("Number of events detected", len(events_times_cluster))
        return events, events_times_cluster

    def thresholding_cwt_kurt(self, files_list, start, end):

            traces = []

            for file in files_list:
                try:
                    tr = read(file)[0]

                    tr = self.fill_gaps(tr)
                    if tr is not None:

                       traces.append(tr)
                       print(tr)
                except:
                    print("File not included: ", file)

            st = Stream(traces)
            st.merge()

            if self.decimate_sampling_rate:
                st.resample(self.decimate_sampling_rate)
                cw = ConvolveWaveletScipy(st, decimate_stream=True)
                tt = int(self.decimate_sampling_rate / 2)
            else:
                cw = ConvolveWaveletScipy(st, decimate_stream=True)
                tt = int(st[0].stats.sampling_rate / 2)

            cw.setup_wavelet(start_time=start, end_time=end, wmin=6, wmax=6, tt=tt, fmin=self.fmin, fmax=self.fmax,
                             nf=60, use_rfft=False, decimate=False, setup_wavelets_stream=True)

            st_kurt = cw.charachteristic_function_kurt_stream(window_size_seconds=self.time_window)

            events = coincidence_trigger(trigger_type=None, thr_on=self.the_on, thr_off=self.the_off,
                      trigger_off_extension=self.centroid_radio, thr_coincidence_sum=self.coincidence, stream=st_kurt,
                                          similarity_threshold=0.8, details=True)

            triggers = self._extract_event_info(events)
            events_times_cluster, _ = MseedUtil.cluster_events(triggers, eps=self.centroid_radio)
            print("Number of events detected", len(events_times_cluster))

            return events, events_times_cluster

    def separate_picks_by_events(self, input_file, output_file, centroids):
        """
        Separates picks by events based on centroids and their radius.

        :param input_file: Path to the input file with pick data.
        :param output_file: Path to the output file for separated picks.
        :param centroids: List of UTCDateTime objects representing centroids.
        :param radius: Radius in seconds for each centroid.
        """
        # Load the input data
        df = pd.read_csv(input_file, delimiter='\s+')
        # Ensure columns are properly typed
        # df['Date'] = df['Date'].astype(str)  # Date as string for slicing
        # df['Hourmin'] = df['Hourmin'].astype(int)  # Hourmin as integer
        # df['Seconds'] = df['Seconds'].astype(float)  # Seconds as float for fractional handling
        # Parse date and time columns into a single datetime object
        df['FullTime'] = df.apply(lambda row: UTCDateTime(
            f"{row['Date']}T{str(row['Hour_min']).zfill(4)}:{row['Seconds']}"
        ), axis=1)

        # Create a new column for event assignment
        df['Event'] = None

        # Assign picks to the closest centroid within the radius
        for i, centroid in enumerate(centroids):
            within_radius = df['FullTime'].apply(lambda t: abs((t - centroid)) <= self.centroid_radio / 2)
            df.loc[within_radius, 'Event'] = f"Event_{i + 1}"

        # Write grouped picks into the output file
        with open(output_file, 'w') as f:
            for event, group in df.groupby('Event'):
                if pd.isna(event):
                    continue  # Skip unassigned picks
                f.write(f"{event}\n")
                group.drop(columns=['Event', 'FullTime']).to_csv(f, sep='\t', index=False)
                f.write("\n")

    def process_day(self, args):
        """Process a single day's data and return events or 'empty'."""
        sp, start, end = args
        # Filter files for the given time range
        filtered_files = sp.filter_time(starttime=start, endtime=end, tol=3600, use_full=True)
        if filtered_files:
            if self.method == 'classicstalta':
                return self.thresholding_sta_lta(filtered_files, start, end)
            else:
                return self.thresholding_cwt_kurt(filtered_files, start, end)
        else:
            return "empty"

    def optimized_project_processing(self, **kwargs):


        input_file: str = kwargs.pop('input_file', None)
        output_file: str = kwargs.pop('output_file', None)

        info = self.project.get_project_basic_info()
        print(info['Start'], info['End'])

        # Parse start and end times
        start_time = UTCDateTime(datetime.strptime(info['Start'], '%Y-%m-%d %H:%M:%S'))
        end_time = UTCDateTime(datetime.strptime(info['End'], '%Y-%m-%d %H:%M:%S'))

        # Generate daily time ranges
        daily_ranges = [(start_time + i * 86400, start_time + (i + 1) * 86400)
                        for i in range(int((end_time - start_time) // 86400))]

        if len(daily_ranges) == 0 and (end_time-start_time) < 86400:
            daily_ranges = [(start_time, end_time)]
        # Prepare arguments for multiprocessing
        tasks = [(self.project, start, end) for start, end in daily_ranges]

        # Use multiprocessing to parallelize
        with Pool() as pool:
            results = pool.map(self.process_day, tasks)

        # Join the output of all days

        final__filtered_results = []
        details = []
        for item in results:
            times = item[1]
            details.extend(item[0])
            final__filtered_results.extend(times)

        if len(results[0][1]) > 0 and input_file is not None and output_file is not None:
            self.separate_picks_by_events(input_file, output_file, centroids=final__filtered_results)

        return final__filtered_results, details
    
if __name__ == '__main__':

    # Example of parametrization and association pf event picks
    path_to_project = "/Users/robertocabiecesdiaz/Documents/ISP/Andorra_test"
    input_file = '/Users/robertocabiecesdiaz/Documents/test_surfquake/my_test/picks/nll_picks.txt'
    output_file = '/Users/robertocabiecesdiaz/Documents/test_surfquake/my_test/picks/nll_picks_associated.txt'

    kurtosis = {"method": "kurtosis", "fmin": 0.5, "fmax": 12.0, "the_on": 20, "the_off": 5, "time_window": 15,
                "coincidence": 4, "centroid_radio": 60, "decimate_sampling_rate": 40}

    classic_sta_lta = {"method": "classicstalta", "fmin": 2.0, "fmax": 8.0, "the_on": 15, "the_off": 10,
                       "sta": 1, "lta": 40, "coincidence": 4, "centroid_radio": 60}

    ct = CoincidenceTrigger(project=path_to_project, parameters=classic_sta_lta)
    ct.optimized_project_processing(input_file = input_file, output_file=output_file)