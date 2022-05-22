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
    GAU = "GAU"
    Err = "Err"
    ErrMag = "ErrMag"
    CodaDuration = "Coda_duration"
    Amplitude = "Amplitude"
    Period = "Period"

    def __init__(self, output_path=None, overwrite=True):
        """
        Manager to save pick data in a csv file.

        :param output_path: The full location for the output file. If None a default location will be used.
        :param overwrite: If True it will overwrite the output file, otherwise will keep it and new data
            will be appended.

        Example:

        * Instanciate an object of PickerManager.
        >>> pm = PickerManager()

        * Add data and save it.
        >>> pm.add_data("2009-08-24T00:20:07.500000Z", 122.5, "ST", "Phase")
        >>> pm.add_data("2009-08-24T00:20:21.500000Z", 122.5, "ST", "Phase")
        >>> pm.save()
        """
        self.file_separator = " "
        if output_path:
            self.__output_path = output_path
        else:
            self.__output_path = self.get_default_output_path()

        self.columns = [self.StationName, self.Instrument, self.Component, self.PPhaseOnset, self.PPhaseDescriptor,
                        self.FirstMotion, self.Date, self.HourMin, self.Seconds, "GAU", self.Err, self.ErrMag,
                        self.CodaDuration, self.Amplitude, self.Period]

        if overwrite:
            self.df = self.__setup_file()
        else:
            self.df = self.__load()

    @property
    def output_path(self):
        return self.__output_path

    def __setup_file(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.output_path, sep=self.file_separator, columns=self.columns, index=False)
        return df

    def __load(self) -> pd.DataFrame:
        out = self.read()
        return out

    def __validate_kwargs(self, v: dict):
        for key in v.keys():
            if key not in self.columns:
                raise ValueError("The key {} must by in {}".format(key, self.columns))

    @staticmethod
    def get_default_output_path():
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_output", "obs")
        if not os.path.isdir(root):
            os.mkdir(root)

        return os.path.join(root, "output.txt")

    @staticmethod
    def __from_utctime_to_datetime(time: UTCDateTime):
        """
        Convert UTCDateTime to date format of the locate file.

        :param time: The UTCDateTime
        :return: A tuple containing date, hour_min, seconds.
        """
        date = "{year:04d}{month:02d}{day:02d}".format(year=time.date.year, month=time.date.month, day=time.date.day)
        hour_min = "{:02d}{:02d}".format(time.hour, time.minute)
        seconds = "{second:02d}.{micro}".format(second=time.second, micro = str(time.microsecond)[0:3])

        return date, hour_min, seconds

    def add_data(self, time, err: float, amplitude: float, station: str, p_phase: str, **kwargs):
        """
        Add data to be saved.

        Important: To save data to file you must call the method save.

        :param time: An UTCDateTime or a valid time string.
        :param err: Associated uncertainty for that pick
        :param amplitude: The amplitude of the waveform.
        :param station: The station name
        :param p_phase: The phase.

        :keyword Instrument: The instrument.
        :keyword Component:
        :keyword First_Motion: The polarization, either "U" or "D"
        :keyword Err: uncertainty
        :keyword ErrMag:
        :keyword Coda_duration:
        :keyword Period:
        :return:
        """

        if not isinstance(time, UTCDateTime):
            time = UTCDateTime(time)

        date, hour_min, seconds = self.__from_utctime_to_datetime(time)
        amplitude = "{0:.2f}".format(amplitude)
        #err = "{0:.2f}".format(err)
        err = format(err, ".2E")
        self.__add_data(Date=date, Hour_min=hour_min, Seconds=seconds, Err=err, Station_name=station, Amplitude=amplitude,
                        P_phase_descriptor=p_phase, **kwargs)

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
        :keyword GAU:
        :keyword Err:
        :keyword ErrMag:
        :keyword Coda_duration:
        :keyword Amplitude:
        :keyword Period:

        :return:
        """
        self.__validate_kwargs(kwargs)
        data = {key: kwargs.get(key, "?") for key in self.columns}  # start a dict with default values equal "?"

        # Override defaults values for some keys
        data["GAU"] = "GAU"
        data[self.Err] = "{:.1f}".format(0) if data[self.Err] == "?" else data[self.Err]
        #data[self.Err] = "GAU" if data[self.Err] == "?" else data[self.Err]
        data[self.ErrMag] = "{:.1f}".format(0) if data[self.ErrMag] == "?" else data[self.ErrMag]
        data[self.CodaDuration] = "{:.1f}".format(0) if data[self.CodaDuration] == "?" else data[self.CodaDuration]
        data[self.Period] = "{:.1f}".format(0) if data[self.Period] == "?" else data[self.Period]

        if data[self.FirstMotion] == "+" and kwargs.get("P_phase_descriptor")[0] == "P":
            data[self.FirstMotion] = "U"
        elif data[self.FirstMotion] == "-" and kwargs.get("P_phase_descriptor")[0] == "P":
            data[self.FirstMotion] = "D"

        if data[self.FirstMotion] == "+" and kwargs.get("P_phase_descriptor")[0] == "S":
            if kwargs.get("Component")[2]=="N" or kwargs.get("Component")[2]=="R":
                data[self.FirstMotion] = "f"

        elif data[self.FirstMotion] == "-" and kwargs.get("P_phase_descriptor")[0] == "S":
            if kwargs.get("Component")[2]=="N" or kwargs.get("Component")[2]=="R":
                data[self.FirstMotion] = "b"

        if data[self.FirstMotion] == "+" and kwargs.get("P_phase_descriptor")[0] == "S":
            if kwargs.get("Component")[2]=="E" or kwargs.get("Component")[2]=="T":
                data[self.FirstMotion] = "r"

        elif data[self.FirstMotion] == "-" and kwargs.get("P_phase_descriptor")[0] == "S":
            if kwargs.get("Component")[2]=="E" or kwargs.get("Component")[2]=="T":
                data[self.FirstMotion] = "l"

        df = pd.DataFrame(data, columns=self.columns, index=[0])
        self.df: pd.DataFrame = self.df.append(df, ignore_index=True)

    def read(self):
        return pd.read_csv(self.output_path, sep=self.file_separator, index_col=None, nrows=None)

    def save(self):
        """
        Save all data in data frame.

        :return:
        """
        self.df.to_csv(self.output_path, sep=self.file_separator, index=False)

    def select_data(self, pick_time: UTCDateTime, station: str) -> pd.DataFrame:
        """
        Select a row in data frame where station and time match.

        :param pick_time: The UTCDateTime to match.
        :param station: The Station name to match.
        :return: The matched data frame.
        """
        date, h_m, s = self.__from_utctime_to_datetime(pick_time)
        df: pd.DataFrame = self.df.loc[(self.df[PickerManager.Date] == date) &
                                       (self.df[PickerManager.HourMin] == h_m) &
                                       (self.df[PickerManager.Seconds] == s) &
                                       (self.df[PickerManager.StationName] == station)]
        return df

    def remove_data(self, pick_time: UTCDateTime, station: str, save=True):
        """
        Remove the data at pick_time and station from data frame. If save=True it will also save the new data frame
        to the file.

        :param pick_time: The UTCDateTime to be removed.
        :param station: The station to be removed.
        :param save: True if you want to save in file, false otherwise.
        :return:
        """
        selected = self.select_data(pick_time, station)
        self.df = self.df.drop(selected.index).reset_index(drop=True)
        if save:
            self.save()

    def clear(self):
        """
        Clear file and data frame.

        :return:
        """
        self.df = self.__setup_file()
