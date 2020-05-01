import os

import numpy as np
from obspy import Stream, Trace

import keras
import tensorflow as tf
from keras import Sequential
from keras.models import model_from_json
from obspy.signal.trigger import trigger_onset

from isp.Utils import ObspyUtil


class CNNPicker:

    def __init__(self):

        keras.backend.set_learning_phase(0)

        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.__model_path = os.path.join(self.root_path, "model", 'red_modelo.json')
        self.__weight_path = os.path.join(self.root_path, "model", "red_pesos.hdf5")

        self.min_proba = 0.99  # Minimum softmax probability for phase detection
        self.decimate_data = True  # If false, assumes data is already 100 Hz samprate
        self.n_shift = 10  # Number of samples to shift the sliding window at a time
        self.win_size = 400
        self.__stream = None
        self.prob_s, self.prob_p, self.prob_n = None, None, None
        self.model = self.load_model()

    def load_model(self, show_summary=False) -> Sequential:
        with open(self.__model_path, 'r') as file:
            model = model_from_json(file.read(), custom_objects={'tf': tf})
            # load weights into new model
            model.load_weights(self.__weight_path)
            if show_summary:
                model.summary()
            print("Loaded model from disk")
        return model

    @property
    def dt(self):
        if self.stats is not None:
            return self.stats.delta
        return 0.

    @property
    def stats(self):
        if self.__stream is not None:
            return self.__stream[0].stats
        return None

    @staticmethod
    def sliding_window(data, size, step_size=1, axis=-1, zero_pad=False):
        stack = []
        for index in range(0, data.shape[axis], step_size):
            window = data[index: index + size:1]

            if window.shape[axis] == size:
                stack.append(window)

            elif window.shape[axis] < size and zero_pad:
                window = np.pad(window, (0, size - window.shape[axis]), "constant", constant_values=0)
                stack.append(window)

        return np.vstack(stack)

    def __create_window(self, tr: Trace):
        return self.sliding_window(tr.data, self.win_size, step_size=self.n_shift)

    def __normalize_data(self):
        tr_win = np.stack(map(self.__create_window, self.__stream), axis=2)
        return tr_win / np.amax(np.abs(tr_win), axis=(1, 2))[:, None, None]

    def validate_stream(self):
        if len(self.__stream) != 3:
            raise ValueError("Stream must have 3 components")

        elif len(self.__stream.get_gaps()) != 0:
            raise ValueError("Your stream has gapes at: {}. Please fix it.".
                             format(self.__stream.get_gaps()))

    def setup_stream(self, stream: Stream, should_copy=True):
        self.__stream = stream.copy() if should_copy else stream  # use a copy of stream by default.
        self.validate_stream()
        self.__stream.sort(['channel'])

        latest_start = np.max([tr.stats.starttime for tr in self.__stream])
        earliest_stop = np.min([tr.stats.endtime for tr in self.__stream])
        self.__stream.trim(latest_start, earliest_stop)
        self.__stream.detrend(type='linear')
        self.__stream.filter(type='bandpass', freqmin=3., freqmax=20.)
        if not ObspyUtil.has_same_sample_rate(self.__stream, 100):
            # self.__stream.resample(100)  # TODO test resample. it gets more events.
            self.__stream.interpolate(100.0)
            print("Resample")

    def predict(self):

        if self.__stream is None:
            raise ValueError("Please call setup_stream(stream) before predict.")

        tr_win = self.__normalize_data()
        ts = self.model.predict(tr_win, verbose=True)
        self.prob_p, self.prob_s, self.prob_n = np.rollaxis(ts, 1)

    def get_time_from_index(self, index):
        return self.stats.starttime + \
               (index * self.n_shift + self.win_size * 0.5) * self.dt

    def __analise_prediction(self, prob):
        triggers = trigger_onset(prob, self.min_proba, 0.1)
        picks = []

        for trigger in triggers:
            index_on, index_off = trigger  # index of when event is On and Off.

            # gets the index where the maximum is between On and Off
            index_of_max = index_on + np.argmax(prob[index_on:index_off]) \
                if index_on < index_off else index_on

            stamp_pick = self.get_time_from_index(index_of_max)
            picks.append(stamp_pick)
            print("{}".format(stamp_pick.isoformat()))

        return picks

    def get_arrivals(self):
        arrival_p, arrival_s = map(self.__analise_prediction, [self.prob_p, self.prob_s])
        return {"p": arrival_p, "s": arrival_s}
