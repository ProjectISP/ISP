import os

import pandas as pd
from obspy import UTCDateTime


class PickerManager:

    StationName = "Station_name"
    Instrument = "Instrument"
    Component = "Component"
    PPhaseOnset = "P_phase_onset"
    PPhaseDescriptor = "P_phase_descriptor"
    FirstMotion = "First_Motion"
    Date = "Date"
    HourMin = "Hour_min"
    Seconds = "Seconds"
    Err = "Err"
    CodaDuration = "Coda_duration"
    Amplitude = "Amplitude"
    Period = "Period"
    PriorWt = "PriorWt"

    def __init__(self, output_path=None):
        """
        Manager to save pick data in a csv file.

        :param output_path: The full location for the output file. If None a default location will be used.

        Example:

        * Instanciate an object of PickerManager.
        >>> pm = PickerManager()

        * Add data and save it.
        >>> pm.add_data("2009-08-24T00:20:07.500000Z", 122.5, "ST", "Phase")
        >>> pm.add_data("2009-08-24T00:20:21.500000Z", 122.5, "ST", "Phase")
        >>> pm.save()
        """
        self.file_separator = " "
        self.__buffer_data = []
        self.__output_file_name = "output.txt"
        if output_path:
            self.__output_path = output_path
        else:
            self.__output_path = self.__get_default_output_path()

        self.columns = [self.StationName, self.Instrument, self.Component,self.PPhaseOnset,self.PPhaseDescriptor,
                        self.FirstMotion, self.Date, self.HourMin, self.Seconds, self.Err, self.CodaDuration,
                        self.Amplitude, self.Period, self.PriorWt]

        self.__setup_file()

    @property
    def output_path(self):
        return self.__output_path

    @property
    def buffer_data(self):
        return self.__buffer_data

    def __setup_file(self):
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.output_path, sep=self.file_separator, columns=self.columns, index=False)

    def __validate_kwargs(self, v: dict):
        for key in v.keys():
            if key not in self.columns:
                raise ValueError("The key {} must by in {}".format(key, self.columns))

    def __get_default_output_path(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_output")
        if not os.path.isdir(root):
            raise FileNotFoundError("The dir {} doesn't exist.".format(root))

        return os.path.join(root, self.__output_file_name)

    def add_data(self, time, amplitude: float, station: str, p_phase: str, **kwargs):
        """
        Add data to be saved.

        Important: To save data to file you must call the method save.

        :param time: An UTCDateTime or a valid time string.
        :param amplitude: The amplitude of the waveform.
        :param station: The station name
        :param p_phase: The phase.

        :keyword Instrument: The instrument.
        :keyword Component:
        :keyword P_phase_descriptor:
        :keyword First_Motion:
        :keyword Err:
        :keyword Coda_duration:
        :keyword Period:
        :keyword PriorWt:
        :return:
        """

        if not isinstance(time, UTCDateTime):
            time = UTCDateTime(time)

        date = "{year:04d}{month:02d}{day:02d}".format(year=time.date.year, month=time.date.month, day=time.date.day)
        hour_min = "{:02d}{:02d}".format(time.hour, time.minute)
        seconds = "{:02d}".format(time.second)
        amplitude = "{0:.2f}".format(amplitude)

        self.__add_data(Date=date, Hour_min=hour_min, Seconds=seconds, Station_name=station, Amplitude=amplitude,
                        P_phase_onset=p_phase, **kwargs)

    def __add_data(self, **kwargs):
        """
        Add data to be saved. This method creates a dictionary, if kwargs match the columns it sets its value,
        otherwise set a default value equals to ?. See :func:`add_location_data`

        Important: To save data to file you must call the method save.

        :keyword Station_name: The station name.
        :keyword Instrument: The instrument.
        :keyword Component:
        :keyword P_phase_onset:
        :keyword P_phase_descriptor:
        :keyword First_Motion:
        :keyword Date:
        :keyword Hour_min:
        :keyword Seconds:
        :keyword Err:
        :keyword Coda_duration:
        :keyword Amplitude:
        :keyword Period:
        :keyword PriorWt:

        :return:
        """
        self.__validate_kwargs(kwargs)
        data = {key:  kwargs.get(key, "?") for key in self.columns}
        self.__buffer_data.append(data)

    def save(self):
        """
        Save all data in buffer_data. After call this method buffer data will be clear.

        :return:
        """
        out = pd.read_csv(self.output_path, sep=self.file_separator, index_col=None, nrows=None)
        for data in self.__buffer_data:
            out = out.append(data, ignore_index=True)

        out.to_csv(self.output_path, sep=self.file_separator, index=False)
        self.__buffer_data.clear()

    def clear(self):
        """
        Clear file and buffer data.

        :return:
        """
        self.__setup_file()
        self.__buffer_data.clear()
