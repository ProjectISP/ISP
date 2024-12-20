from obspy import UTCDateTime, read, Trace
import numpy as np
import tensorflow as tf
from datetime import datetime
from isp.Utils import MseedUtil
from multiprocessing import Pool
from surfquakecore.project.surf_project import SurfProject

class Polarity:
    def __init__(self, project_file,  model_path, arrivals_path):

        self.model = tf.keras.models.load_model(model_path)
        self.project = self._read_project(project_file)
        self.model_path = model_path
        self.arrivals_path = arrivals_path
        self.import_picks()

    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        return model_summary

    def _read_project(self, project_path):
        return SurfProject.load_project(project_path)


    def import_picks(self):
        self.pick_times_imported = MseedUtil.get_NLL_phase_picks(input_file=self.arrivals_path)


    def run_predict_polarity_day(self, filtered_files):
        tr = None
        for file in filtered_files:
            try:
                tr = read(file)[0]
            except:
                print("Cannot process day ", file)

            if isinstance(tr, Trace):
                filtered_picks = self._filter_trace_by_picks(tr)
                print(filtered_picks)


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

        # Check if the trace ID exists in the pick dictionary
        if trace_id in self.pick_times_imported:
            for pick in self.pick_times_imported[trace_id]:
                phase_name, pick_time, channel = pick[0], pick[1], pick[2]

                # Ensure channel contains "Z" and pick time falls within the trace's time window
                if "Z" in channel and trace_start <= pick_time <= trace_end and phase_name == "P":
                    filtered_picks.append(pick)

        return filtered_picks


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

        # Use multiprocessing to parallelize
        with Pool() as pool:
            results = pool.map(self.process_coincidence_pol, tasks)

        # Join the output of all days
        # for item in results:
        #     if item[0] is not None and item[1] is not None:
        #         details.extend(item[0])
        #         final_filtered_results.extend(item[1])


        return final_filtered_results, details

    def process_coincidence_pol(self, args):
        """Process a single day's data and return events or 'empty'."""
        sp, start, end = args
        # Filter files for the given time range
        filtered_files = sp.filter_time(starttime=start, endtime=end, tol=3600, use_full=True)
        if filtered_files:
            self.run_predict_polarity_day(filtered_files)
        else:
            return "empty"


    def run_predict_polarity_day(self, filtered_files, model="PolarCap"):
        tr = None
        for file in filtered_files:
            try:
                tr = read(file)[0]
                print(tr)
            except:
                print("Cannot process day ", file)

            if isinstance(tr, Trace):
                filtered_picks = self._filter_trace_by_picks(tr)
                print(filtered_picks)
                data = tr.data

                for pick in filtered_picks:
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
                    data_full = data_full / np.max(data_full)

                    data_chop = data_chop - np.mean(data_chop)
                    #data_chop = data_chop / np.max(data_chop)
                    data_plot = data_chop
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
                    #self._plot(data_plot, data_full, label=predictions[0])



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


if __name__ == "__main__":

    project_path = "/Users/robertocabiecesdiaz/Documents/ISP/earthquake_test"

    arrivals_path = ("/Users/robertocabiecesdiaz/Documents/ISP/isp/earthquakeAnalysis/loc_structure/"
                     "output_original_example.txt")
    model_path = '/Users/robertocabiecesdiaz/Documents/ISP/isp/PolarCap/PolarCAP.h5'
    output = "./"
    pol = Polarity(project_file=project_path, model_path=model_path, arrivals_path=arrivals_path)
    print(pol)
    pol.optimized_project_processing_pol()

