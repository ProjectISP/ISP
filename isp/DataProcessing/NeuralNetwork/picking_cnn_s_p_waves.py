import os
import numpy as np
from obspy import Stream

import keras
import tensorflow as tf
from keras import Sequential
from keras.models import model_from_json


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
        self.model = self.load_model()
        print(self.model.input_shape)

    def load_model(self, show_summary=False) -> Sequential:
        with open(self.__model_path, 'r') as file:
            model = model_from_json(file.read(), custom_objects={'tf': tf})
            # load weights into new model
            model.load_weights(self.__weight_path)
            if show_summary:
                model.summary()
            print("Loaded model from disk")
        return model

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

    def __normalize_data(self, stream: Stream):
        tr_win = np.array([])
        for index, tr in enumerate(stream):
            win = self.sliding_window(tr.data, self.win_size, step_size=self.n_shift)

            if tr_win.shape[0] == 0:
                tr_win = np.zeros((win.shape[0], self.win_size, 3))

            tr_win[:, :, index] = win

        return tr_win / np.max(np.abs(tr_win), axis=(1, 2))[:, None, None]

    def predict(self, stream: Stream):

        if len(stream) != 3:
            raise ValueError("Stream must have 3 components")

        latest_start = np.max([tr.stats.starttime for tr in stream])
        earliest_stop = np.min([tr.stats.endtime for tr in stream])
        stream.trim(latest_start, earliest_stop)
        stream.detrend(type='linear')
        tr_win = self.__normalize_data(stream)

        return self.model.predict(tr_win, verbose=True)
