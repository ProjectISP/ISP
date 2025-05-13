import cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hampel(np.ndarray data, int window_size, float n_sigma):
    """
    Applies the Hampel filter to a 1-dimensional numpy array for outlier detection.

    This function replaces outliers in the input data with the median value within a moving window.

    Parameters:
        data (numpy.ndarray): The input 1-dimensional numpy array to be filtered.
        window_size (int, optional): The size of the moving window for outlier detection.
        n_sigma (float, optional): The number of standard deviations for outlier detection.

    Returns:
        Tuple[numpy.ndarray, List[int], numpy.ndarray, numpy.ndarray]: A tuple containing:
            - Filtered data (numpy.ndarray): A copy of the input data with outliers replaced by median values within the specified window.
            - Indices of outliers (List[int]): A list of indices where outliers were found.
            - Local medians (numpy.ndarray): An array containing local median values for each element of the input data.
            - Estimated standard deviations (numpy.ndarray): An array containing estimated standard deviations for each element of the input data.
    """
    # Ensure data is 1D
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array")

    # Convert input to float64 for consistent processing
    if data.dtype != np.float64:
        data = np.ascontiguousarray(data, dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] data_view = data
    cdef int data_len = data_view.shape[0]
    cdef int half_window = window_size // 2

    # Preallocate output arrays
    cdef np.ndarray[np.float64_t, ndim=1] filtered_data = data_view.copy()
    cdef np.ndarray[np.float64_t, ndim=1] thresholds = np.empty(data_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] median_absolute_deviations = np.empty(data_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] medians = np.empty(data_len, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] outlier_indices = np.empty(data_len, dtype=np.int32)

    cdef:
        int i, j, window_length, num_outliers = 0
        np.ndarray[np.float64_t, ndim=1] window
        float median, mad, threshold

    for i in range(half_window, data_len - half_window):
        window = data_view[i - half_window: i + half_window + 1].copy()
        window_length = len(window)
        median = np.median(window)

        for j in range(window_length):
            window[j] = abs(window[j] - median)

        mad = np.median(window)
        threshold = n_sigma * 1.4826 * mad

        thresholds[i] = threshold
        medians[i] = median
        median_absolute_deviations[i] = mad

        if abs(data_view[i] - median) > threshold:
            filtered_data[i] = median
            outlier_indices[num_outliers] = i
            num_outliers += 1

    outlier_indices = outlier_indices[:num_outliers]

    return (
        filtered_data,
        outlier_indices,
        medians,
        median_absolute_deviations,
        thresholds
    )