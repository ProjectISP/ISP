from typing import Union
from datetime import datetime
from obspy import UTCDateTime, read, Stream
from obspy.signal.trigger import coincidence_trigger
from multiprocessing import Pool
import pandas as pd
from surfquakecore.project.surf_project import SurfProject

class CoincidenceTrigger:
    def __init__(self, project: Union[SurfProject, str], **kwargs):
        if isinstance(project, str):
            self.project = SurfProject.load_project(project)
        else:
            self.project = project

        self.method: str = kwargs.pop('input_file', 'classicstalta')
        self.the_on = kwargs.pop('the_on', 3)
        self.the_off = kwargs.pop('the_on', 3)

        self.param1 = kwargs.pop('param1', 1) # sta in seconds
        self.param2 = kwargs.pop('param2', 40) # lta in seconds
        self.param3 = kwargs.pop('param3', 60)  # centroid radious
        self.param4 = kwargs.pop('param4', 3) # coincidence

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
        #st.select(channel="*Z") # no necessary, better filter project before
        st.trim(starttime=start, endtime=end)
        st.merge()
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)
        st.filter(type="bandpass", freqmin=0.5, freqmax=8)
        st.detrend(type='simple')
        st.taper(max_percentage=0.05)
        fs = 20
        st.resample(20)
        events = coincidence_trigger('classicstalta', self.the_on, self.the_off, st,
                                     thr_coincidence_sum=self.param4,
                                     max_trigger_length=self.param3, sta=self.param1*fs, lta=self.param2*fs)
        print(events)
        return events

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
        df['Date'] = df['Date'].astype(str)  # Date as string for slicing
        df['Hourmin'] = df['Hourmin'].astype(int)  # Hourmin as integer
        df['Seconds'] = df['Seconds'].astype(float)  # Seconds as float for fractional handling
        # Parse date and time columns into a single datetime object
        df['FullTime'] = df.apply(lambda row: UTCDateTime(
            f"{row['Date']}T{str(row['Hourmin']).zfill(4)}:{row['Seconds']}"
        ), axis=1)

        # Create a new column for event assignment
        df['Event'] = None

        # Assign picks to the closest centroid within the radius
        for i, centroid in enumerate(centroids):
            within_radius = df['FullTime'].apply(lambda t: abs((t - centroid)) <= (self.param3)/2)
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
            return self.thresholding_sta_lta(filtered_files, start, end)
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

        if len(results) > 0 and input_file is not None and output_file is not None:
            events = []
            for event in results:
                events.append(event("time"))
            self.separate_picks_by_events(input_file, output_file, centroids=results)


if __name__ == '__main__':
    path_to_project = "/Users/robertocabiecesdiaz/Documents/test_surfquake/project/project.pkl"
    ct = CoincidenceTrigger(project=path_to_project)
    ct.optimized_project_processing()