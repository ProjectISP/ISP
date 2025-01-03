from typing import Union
from copy import deepcopy
from obspy import UTCDateTime, read, Trace, Stream
import numpy as np
import tensorflow as tf
from datetime import datetime
from isp.Utils import MseedUtil
from surfquakecore.project.surf_project import SurfProject

# def init_worker():
#     """Initialize TensorFlow settings for each worker process."""
#     tf.compat.v1.enable_eager_execution()

class Polarity:
    def __init__(self, project: Union[SurfProject, str],  model_path, arrivals_path, threshold, output_path):

        self.model = tf.keras.models.load_model(model_path)

        if isinstance(project, str):
            self.project = SurfProject.load_project(project)
        else:
            self.project = project

        self.model_path = model_path
        self.arrivals_path = arrivals_path
        self.threshold = threshold
        self.output_path = output_path
        self.import_picks()

    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        return model_summary

    def import_picks(self):
        self.pick_times_imported = MseedUtil.get_NLL_phase_picks(input_file=self.arrivals_path)
        self.pick_times_imported_modify = deepcopy(self.pick_times_imported)


    def _filter_trace_by_picks(self, tr: Trace):
        """
        Filters an ObsPy Trace based on a pick dictionary.

        Parameters:
        - trace (obspy.Trace): The input seismic trace.
        - pick_dict (dict): Dictionary containing pick information in the form:
          {
              "station.channel": [["Phase_name", UTCDateTime, "channel", ...], ...],
              ...
          }

        Returns:
        - filtered_picks (list): List of picks matching the trace criteria.
        """
        # Get the trace's metadata
        trace_id = f"{tr.stats.station}.{tr.stats.channel}"
        trace_start = tr.stats.starttime
        trace_end = tr.stats.endtime

        # List to store matching picks
        filtered_picks = []
        items = []
        # Check if the trace ID exists in the pick dictionary
        if trace_id in self.pick_times_imported:
            for item, pick in enumerate(self.pick_times_imported[trace_id]):
                phase_name, pick_time, channel = pick[0], pick[1], pick[2]

                # Ensure channel contains "Z" and pick time falls within the trace's time window
                if trace_start <= pick_time <= trace_end and phase_name == "P":
                    filtered_picks.append(pick)
                    items.append(item)

        return filtered_picks, items


    def optimized_project_processing_pol(self):

        final_filtered_results = []
        details = []

        info = self.project.get_project_basic_info()
        print(info['Start'], info['End'])

        # Parse start and end times
        start_time = UTCDateTime(datetime.strptime(info['Start'], '%Y-%m-%d %H:%M:%S'))
        end_time = UTCDateTime(datetime.strptime(info['End'], '%Y-%m-%d %H:%M:%S'))

        # Generate daily time ranges
        daily_ranges = [(start_time + i * 86400, start_time + (i + 1) * 86400)
                        for i in range(int((end_time - start_time) // 86400))]

        if len(daily_ranges) == 0 and (end_time - start_time) < 86400:
            daily_ranges = [(start_time, end_time)]

        # Prepare arguments for multiprocessing
        tasks = [(self.project, start, end) for start, end in daily_ranges]

        for task in tasks:
            self.process_coincidence_pol(task)

        # no using multiprocess
        # Use multiprocessing to parallelize
        # with Pool(initializer=init_worker) as pool:
        #     results = pool.map(self.process_coincidence_pol, tasks)
        self.write_output()

    def process_coincidence_pol(self, args):
        """Process a single day's data and return events or 'empty'."""
        sp, start, end = args
        # Filter files for the given time range
        filtered_files = sp.filter_time(starttime=start, endtime=end, tol=3600, use_full=True)
        if filtered_files:
            self.run_predict_polarity_day(filtered_files)
        else:
            return "empty"


    def run_predict_polarity_day(self, filtered_files):
        tr = None
        for file in filtered_files:
            try:
                tr = read(file)[0]
                print(tr)
                tr = self.fill_gaps(tr)
                if tr.stats.sampling_rate != 100:
                    tr.resample(100)
            except:
                print("Cannot process day ", file)

            if isinstance(tr, Trace):
                filtered_picks, items = self._filter_trace_by_picks(tr)
                trace_id = f"{tr.stats.station}.{tr.stats.channel}"
                print(filtered_picks)
                data = tr.data

                for pick, item in zip(filtered_picks, items):
                    try:
                        pick_time = pick[1]
                        time_samples = int((pick_time - tr.stats.starttime) * tr.stats.sampling_rate)
                        start_sample = time_samples - 32
                        end_sample = time_samples + 32
                        data_chop = data[start_sample:end_sample]

                        start_sample = time_samples - 200
                        end_sample = time_samples + 200
                        data_full = data[start_sample:end_sample]
                        if len(data_chop) < 64:
                            data_chop = data_chop[0:64]

                        data_full = data_full - np.mean(data_full)

                        data_chop = data_chop - np.mean(data_chop)
                        #data_chop = data_chop / np.max(data_chop)
                        # data_plot = data_chop
                        data_chop = data_chop.reshape(1, -1, 1)
                        data_chop = self.norm(data_chop)
                        y_pred = self.model.predict(data_chop)

                        pol_pred = np.argmax(y_pred[1], axis=1)
                        pred_prob = np.max(y_pred[1], axis=1)
                        predictions = []
                        polarity = ['Negative', 'Positive']
                        for pol, prob in zip(pol_pred, pred_prob):
                            predictions.append((polarity[pol], prob))

                        print(pol_pred, pred_prob, predictions)
                        self._edit_pick_input(trace_id, item, predictions)
                    except:
                        print("Error estimating prediction polarity at :", pick)
                    #self._plot(data_plot, data_full, label=predictions[0])


    def _edit_pick_input(self, trace_id, item, predictions):

        if predictions[0][0] == "Positive" and predictions[0][1] >= self.threshold:
            self.pick_times_imported_modify[trace_id][item][3] = "U"
        elif predictions[0][0] == "Negative" and predictions[0][1] >= self.threshold:
            self.pick_times_imported_modify[trace_id][item][3] = "D"

        print(self.pick_times_imported_modify[trace_id][item])

    def _plot(self, data, data_full, label):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(data_full, label=label)
        #ax.plot(data_full)
        ax.legend()
        plt.show()

    def norm(self, X):
        maxi = np.max(abs(X), axis=1)
        X_ret = X.copy()
        for i in range(X_ret.shape[0]):
            X_ret[i] = X_ret[i] / maxi[i]
        return X_ret

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

    def write_output(self):
        with open(self.output_path, 'w') as file:

            # Write the header line
            header = ("Station_name\tInstrument\tComponent\tP_phase_onset\tP_phase_descriptor\t"
                       "First_Motion\tDate\tHour_min\tSeconds\tErr\tErrMag\tCoda_duration\tAmplitude\tPeriod\n")

            file.write(header)

            for key in self.pick_times_imported_modify.keys():
                id = key.split(".")
                contents = self.pick_times_imported_modify[key]
                station = id[0].ljust(6)  # Station name, left-justified, 6 chars
                instrument = "?".ljust(4)  # Placeholder for Instrument
                component = id[1].ljust(4)  # Placeholder for Component

                for content in contents:
                    time = UTCDateTime(content[1]) # Convert starttime and endtime to pandas-compatible datetime objects
                    date_concatanate = time.strftime("%Y%m%d")
                    p_phase_onset = "?"  # Placeholder for P phase onset
                    phase_descriptor = content[0].ljust(6)  # Phase descriptor (e.g., P, S)
                    first_motion = content[3]  # Placeholder for First Motion
                    date = f"{ date_concatanate}"  # Date in yyyymmdd format
                    hour_min = f"{time.hour:02}{time.minute:02}"  # hhmm
                    seconds = f"{time.second + time.microsecond / 1e6:07.4f}"
                    err = "GAU"  # Error type (GAU)

                    err_mag = f"{content[5]:.2e}"  # Error magnitude in seconds
                    coda_duration = content[6]  # Placeholder for Coda duration
                    amplitude = f"{content[7]:.2e}"  # Amplitude
                    period = content[8]  # Placeholder for Period
                    # Construct the line
                    line = (
                        f"{station} {instrument} {component} {p_phase_onset} {phase_descriptor} {first_motion} "
                        f"{date} {hour_min} {seconds} {err} {err_mag} {coda_duration} {amplitude} {period}\n"
                    )
                    file.write(line)

                    # Add a blank line at the end for NLLoc format compliance
            file.write("\n")


if __name__ == "__main__":

    project_path = "/Users/roberto/Documents/ISP/earthquake_test"

    arrivals_path = ("/Users/roberto/Documents/ISP/isp/earthquakeAnalysis/loc_structure/"
                     "output_original_example.txt")
    model_path = '/Users/roberto/Documents/ISP/isp/PolarCap/PolarCAP.h5'
    output = "./"
    pol = Polarity(project=project_path, model_path=model_path, arrivals_path=arrivals_path, threshold=0.9,
                   output_path="/Users/roberto/Documents/ISP/isp/earthquakeAnalysis/loc_structure/test_focmec.txt")
    print(pol)
    pol.optimized_project_processing_pol()

