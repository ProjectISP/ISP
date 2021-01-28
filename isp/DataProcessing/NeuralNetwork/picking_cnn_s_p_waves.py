import os
from typing import Dict, List

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import trigger_onset

import keras
import tensorflow as tf
from tensorflow.keras import Sequential
#from keras import Sequential
from keras.models import model_from_json


class CNNPicker:

    def __init__(self, show_model_summary=False):

        keras.backend.set_learning_phase(0)

        # private parameters.
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.__model_path = os.path.join(root_path, "model", 'red_modelo.json')
        self.__weight_path = os.path.join(root_path, "model", "red_pesos.hdf5")
        self.__win_size = 400  # window size,it can't be modify.
        self.__stream = None
        self.__model = self.___load_model(show_summary=show_model_summary)

        # public parameters.
        self.max_proba = 0.99  # Maximum probability for phase detection.
        self.min_proba = 0.1   # Minimum probability for phase detection
        self.resample_freq = 100.  # Frequency to resample if choose to do so. Default=100 Hz.
        self.n_shift = 10  # Number of samples to shift the sliding window at a time
        self.prob_s, self.prob_p, self.prob_n = [], [], []

    def ___load_model(self, show_summary=False) -> Sequential:
        """
        Load  CNN model and its weights.

        :param show_summary: True if you want to display information about the CNN. Default=False.

        :return: The loaded model.
        """
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
        """
        :return: A float of stream delta time.
        """
        if self.stats is not None:
            return self.stats.delta
        return 0.

    @property
    def stats(self):
        """
        :return: The stream status of the first trace. None if stream is not set yet.
        """
        if self.__stream is not None:
            return self.__stream[0].stats
        return None

    @property
    def model(self) -> Sequential:
        """
        :return: The loaded CNN model.
        """
        return self.__model

    @staticmethod
    def sliding_window(data, size, step_size=1, axis=-1, zero_pad=False):
        """
        Calculate a sliding window over a signal.

        :param data: The array to be slided over.
        :param size: The sliding window size.
        :param step_size: The sliding window step size. Default to 1.
        :param axis: The axis to slide over. Defaults to the last axis.
        :param zero_pad: Pad window with zeros if size > data.
        :return: A matrix where row in last dimension consists of one instance
            of the sliding window.

        Examples:
        --------
        >>> a = np.array([1, 2, 3, 4, 5])
        >>> CNNPicker.sliding_window(a, size=3)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5]])
        >>> CNNPicker.sliding_window(a, size=3, stepsize=2)
        array([[1, 2, 3],
               [3, 4, 5]])

        >>> CNNPicker.sliding_window(a, size=4, stepsize=2, zero_pad=True)
        array([[1, 2, 3, 4],
               [3, 4, 5, 0]])
        """
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
        return self.sliding_window(tr.data, self.__win_size, step_size=self.n_shift)

    def __normalize_data(self):
        tr_win = np.stack(list(map(self.__create_window, self.__stream)), axis=2)
        return tr_win / np.amax(np.abs(tr_win), axis=(1, 2))[:, None, None]

    def validate_stream(self):
        if len(self.__stream) != 3:
            raise ValueError("Stream must have 3 components")

        elif len(self.__stream.get_gaps()) != 0:
            raise ValueError("Your stream has gapes at: {}. Please fix it.".
                             format(self.__stream.get_gaps()))

    def setup_stream(self, stream: Stream, freqmin=3., freqmax=20., **kwargs):
        """
        Setup the stream to be analyzed by the CNN. It must contain 3 channels.
            This method will perform a trim, detrend and bandpass filter to the Stream. If
            sample rate is not 100 Hz it will also interpolate it to 100 Hz.

        :param stream: An obspy.Stream with 3 channels inside.
        :param freqmin: Minimum frequency for the bandpass filter. Default=3Hz, filter
            should be set True for this to have an effect.
        :param freqmax: Maximum frequency for the bandpass filter. Default=20Hz, filter
            should be set True for this to have an effect.

        :keyword:
        :keyword resample: Interpolate Stream to self.resample_freq. Default=True. This will
            interpolate to a default frequency of 100Hz. Set to False if you want to avoid it.
            However, it's highly recommend to resample to 100Hz since the CNN is better
            trained for this sample rate.
        :keyword filter: It will apply a bandpass filter using freqmin and freqmax. Default=True.
        :keyword detrend: Detrend stream using linear. Default=True.
        :keyword trim: Trim stream usigin latest_start and earliest_stop from tracers. Default=True.
        :keyword copy: If true it will copy the stream otherwise the
            original stream will be modify. Default = True. Set False if you not gonna use the
            stream after.

        :return:
        """

        copy_stream = kwargs.get("copy", True)
        trim_stream = kwargs.get("trim", True)
        detrend_stream = kwargs.get("detrend", True)
        filter_stream = kwargs.get("filter", True)
        resample_stream = kwargs.get("resample", True)

        self.__stream = stream.copy() if copy_stream else stream  # use a copy of stream by default.
        self.validate_stream()
        self.__stream.sort(['channel'])

        if trim_stream:
            latest_start = np.amax([tr.stats.starttime for tr in self.__stream])
            earliest_stop = np.amin([tr.stats.endtime for tr in self.__stream])
            self.__stream.trim(latest_start, earliest_stop)

        if detrend_stream:
            self.__stream.detrend(type='linear')

        # Filter data plays a big role on predict.
        if filter_stream:
            self.__stream.filter(type='bandpass', freqmin=freqmin, freqmax=freqmax)

        if resample_stream:
            # Works better if interpolate.
            # self.__stream.resample(100)  # TODO test re sample. it gets more events.
            self.__stream.interpolate(self.resample_freq)

    def predict(self, verbose=False):
        """
        Run predict from model. Before calling it be sure to call setup_stream(stream) first. After
            run predict you can call get_arrivals() to get the arrival times of P and S waves.

        :return:
        """

        if self.__stream is None:
            raise ValueError("Please call setup_stream(stream) before predict.")

        tr_win = self.__normalize_data()
        ts = self.model.predict(tr_win, verbose=verbose)
        self.prob_p, self.prob_s, self.prob_n = np.rollaxis(ts, 1)

    def get_time_from_index(self, index):
        return self.stats.starttime + \
               (index * self.n_shift + self.__win_size * 0.5) * self.dt

    def __analise_prediction(self, prob):
        picks = []
        if not (isinstance(prob, np.ndarray) or isinstance(prob, list)):
            print(type(prob))
            raise TypeError("prob must be a list or a numpy.array "
                            "instead it got type {}".format(type(prob)))
        try:
            triggers = trigger_onset(prob, self.max_proba, self.min_proba)

            for trigger in triggers:
                index_on, index_off = trigger  # index of when event is On and Off.

                # gets the index where the maximum is between On and Off
                index_of_max = index_on + np.argmax(prob[index_on:index_off]) \
                    if index_on < index_off else index_on

                stamp_pick = self.get_time_from_index(index_of_max)
                picks.append(stamp_pick)
                print("{}".format(stamp_pick.isoformat()))

        except TypeError:
            # this error happens when pass a empty list of prob.
            pass

        return picks

    def get_arrivals(self) -> Dict[str, List[UTCDateTime]]:
        """
        Get the arrival times of P and S waves.

        :return: A dictionary of keys=("p", "s") arrival times, where arrival times
            is a list of UTCDateTime.
        """
        arrival_p, arrival_s = map(self.__analise_prediction, [self.prob_p, self.prob_s])
        return {"p": arrival_p, "s": arrival_s}
