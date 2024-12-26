from datetime import datetime
from multiprocessing import freeze_support
from obspy import UTCDateTime, read, Stream
from obspy.signal.trigger import coincidence_trigger
from surfquakecore.project.surf_project import SurfProject
from multiprocessing import Pool

path_to_project = "/Volumes/LaCie/all_andorra/project/surfquake_project_year.pkl"


def thresholding(files_list, start, end):

    traces = []
    for file in files_list:
        try:
            tr = read(file)[0]
            traces.append(tr)
        except:
            pass
    st = Stream(traces)
    st.select(channel="*Z")
    st.trim(starttime=start, endtime=end)
    st.merge()
    st.filter(type="bandpass", freqmin=0.5, freqmax=8)


    events = coincidence_trigger('classicstalta', 5,5, st, thr_coincidence_sum=3,
                                 max_trigger_length=60, sta=500, lta=3000)
    print(events)
    return events


def process_day(args):
    """Process a single day's data and return events or 'empty'."""
    sp, start, end = args
    # Filter files for the given time range
    filtered_files = sp.filter_time(starttime=start, endtime=end, tol=3600, use_full=True)
    if filtered_files:
        return thresholding(filtered_files, start, end)
    else:
        return "empty"


def optimized_project_processing(path_to_project):
    sp = SurfProject.load_project(path_to_project)
    info = sp.get_project_basic_info()
    print(info['Start'], info['End'])

    # Parse start and end times
    start_time = UTCDateTime(datetime.strptime(info['Start'], '%Y-%m-%d %H:%M:%S'))
    end_time = UTCDateTime(datetime.strptime(info['End'], '%Y-%m-%d %H:%M:%S'))

    # Generate daily time ranges
    daily_ranges = [(start_time + i * 86400, start_time + (i + 1) * 86400)
                    for i in range(int((end_time - start_time) // 86400))]

    # Prepare arguments for multiprocessing
    tasks = [(sp, start, end) for start, end in daily_ranges]

    # Use multiprocessing to parallelize
    with Pool() as pool:
        results = pool.map(process_day, tasks)

    return results


if __name__ == '__main__':

    freeze_support()
    optimized_project_processing(path_to_project)

