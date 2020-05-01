import os
import unittest

from obspy import read, UTCDateTime
import numpy as np

from isp.DataProcessing.NeuralNetwork import CNNPicker


def sliding_window(data, size, step_size=1, axis=-1, zero_pad=True):

    stack = []
    for index in range(0, data.shape[axis], step_size):
        window = data[index: index + size:1]

        if window.shape[axis] == size:
            stack.append(window)

        elif window.shape[axis] < size and zero_pad:
            window = np.pad(window, (0, size - window.shape[axis]), "constant", constant_values=0)
            stack.append(window)

    return np.vstack(stack)


def sliding_window_2(data, size, step_size=1, padded=False, axis=-1, copy=True):
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if step_size < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / step_size - size / step_size + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= step_size
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided


half_dur = 2.00
only_dt = 0.01
n_win = int(half_dur/only_dt)
n_feat = 2*n_win


class TestCNNPicker(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        cls.wave_form_file_z = os.path.join(dir_path, "test_data", "red_example.HHZ.SAC")
        cls.wave_form_file_n = os.path.join(dir_path, "test_data", "red_example.HHN.SAC")
        cls.wave_form_file_e = os.path.join(dir_path, "test_data", "red_example.HHE.SAC")

    def test_cnn(self):
        st = read(self.wave_form_file_e)
        st += read(self.wave_form_file_n)
        st += read(self.wave_form_file_z)

        cnn = CNNPicker()
        cnn.setup_stream(st)  # set stream to use in prediction.
        cnn.predict()
        arrivals = cnn.get_arrivals()

        print(arrivals)
        self.assertEqual(arrivals["p"][0], UTCDateTime("2016-10-20T15:11:51.100000"))
        self.assertEqual(arrivals["s"][0], UTCDateTime("2016-10-20T15:12:52.500000"))

    def test_sliding(self):
        a = np.array([1, 2, 3, 4, 5])
        n, s = 5, 3
        print(sliding_window_2(a, n, s))
        print(sliding_window(a, n, s))


if __name__ == '__main__':
    unittest.main()
